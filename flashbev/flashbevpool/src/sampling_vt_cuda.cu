// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void sampling_vt_pillarpool_kernel(int c, int n_intervals,
  const float *__restrict__ feat,
  const int *__restrict__ ranks_feat,
  const int *__restrict__ ranks_bev,
  const int *__restrict__ interval_starts,
  const int *__restrict__ interval_lengths,
  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float accumulator = 0;
  const float* cur_feat;

  for(int i = 0; i < interval_length; i++){
    cur_feat = feat + (interval_start + i) * c + cur_c;
    accumulator += (*cur_feat);
  }

  const int* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = accumulator;
}

/*
  Function: sampling VT pillar pooling (backward, cuda)
  Args:
    out_grad         : gradient of output bev feature, FloatTensor[b, c, h_out, w_out]
    feat_grad        : gradient of input feature, FloatTensor[n, h, w, c]
    feat             : input features, FloatTensor[n, h, w, c]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Backward equations:
    Forward: out[rank_bev, c] = Σ_i feat[rank_feat[i], c]
    ∂L/∂feat[rank_feat[i], c] = grad_out[rank_bev, c]
*/
__global__ void sampling_vt_pillarpool_grad_kernel(int c, int n_intervals,
  const float *__restrict__ out_grad,
  const float *__restrict__ depth,
  const float *__restrict__ feat,
  const int *__restrict__ ranks_feat,
  const int *__restrict__ ranks_bev,
  const int *__restrict__ interval_starts,
  const int *__restrict__ interval_lengths,
  float* __restrict__ depth_grad,
  float* __restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_intervals) return;
  int interval_start = interval_starts[idx];
  int interval_length = interval_lengths[idx];

  const int* cur_rank;
  const float* cur_out_grad;
  float* cur_feat_grad;

  for(int cur_c = 0; cur_c < c; cur_c++){
    for(int i = 0; i < interval_length; i++){
      cur_rank = ranks_bev + interval_start + i;
      cur_out_grad = out_grad + *cur_rank * c + cur_c;
      cur_feat_grad = feat_grad + (interval_start + i) * c + cur_c;
      atomicAdd(cur_feat_grad, *cur_out_grad);
    }
  }
}

/*
  Function: sampling VT pillar pooling (fused with bilinear sampling and depth weighting)
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,h,w,2]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    u_coords         : FloatTensor[n]
    v_coords         : FloatTensor[n]
    z_coords         : FloatTensor[n]
    batch_camera_indices : IntTensor[n], encoded as batch_idx * num_cameras + camera_idx
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
template <int DEPTH_DISTRIBUTION>
__global__ void sampling_vt_pillarpool_fused_kernel(int c, int n_intervals,
  const float *__restrict__ depth,
  const float *__restrict__ feat,
  const float *__restrict__ u_coords,
  const float *__restrict__ v_coords,
  const float *__restrict__ z_coords,
  const int *__restrict__ batch_camera_indices,
  const int *__restrict__ ranks_bev,
  const int *__restrict__ interval_starts,
  const int *__restrict__ interval_lengths,
  int batch_size,
  int num_cameras,
  int feat_h,
  int feat_w,
  float epsilon,
  float* __restrict__ out) 
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    float accumulator = 0;
    float valid_count = 0.0f;
    
    for(int i = 0; i < interval_length; i++)
    {
      float feat_u = u_coords[interval_start+i];
      float feat_v = v_coords[interval_start+i];
      float img_d = z_coords[interval_start+i];
      int batch_camera_idx = batch_camera_indices[interval_start+i];
      int batch_idx = batch_camera_idx / num_cameras;
      int camera_index = batch_camera_idx % num_cameras;
      
      if (feat_u < 0 || feat_u >= feat_w || feat_v < 0 || feat_v >= feat_h || img_d <= 0.0f) {
        continue;
      }
      
      int u0 = (int)floorf(feat_u);
      int v0 = (int)floorf(feat_v);
      int u1 = min(u0 + 1, feat_w - 1);
      int v1 = min(v0 + 1, feat_h - 1);
      
      float wu = feat_u - u0;
      float wv = feat_v - v0;
      
      int depth_offset = batch_idx * num_cameras * feat_h * feat_w * 2 + 
                         camera_index * feat_h * feat_w * 2;
      
      int feat_offset = batch_idx * num_cameras * feat_h * feat_w * c + 
                        camera_index * feat_h * feat_w * c + 
                        cur_c;
      
      float depth_mean_00 = depth[depth_offset + v0 * feat_w * 2 + u0 * 2 + 0];
      float depth_sigma_00 = depth[depth_offset + v0 * feat_w * 2 + u0 * 2 + 1];
      float depth_mean_01 = depth[depth_offset + v0 * feat_w * 2 + u1 * 2 + 0];
      float depth_sigma_01 = depth[depth_offset + v0 * feat_w * 2 + u1 * 2 + 1];
      float depth_mean_10 = depth[depth_offset + v1 * feat_w * 2 + u0 * 2 + 0];
      float depth_sigma_10 = depth[depth_offset + v1 * feat_w * 2 + u0 * 2 + 1];
      float depth_mean_11 = depth[depth_offset + v1 * feat_w * 2 + u1 * 2 + 0];
      float depth_sigma_11 = depth[depth_offset + v1 * feat_w * 2 + u1 * 2 + 1];
      
      float depth_mean = (1 - wu) * (1 - wv) * depth_mean_00 +
                         wu * (1 - wv) * depth_mean_01 +
                         (1 - wu) * wv * depth_mean_10 +
                         wu * wv * depth_mean_11;
      
      float depth_sigma = (1 - wu) * (1 - wv) * depth_sigma_00 +
                          wu * (1 - wv) * depth_sigma_01 +
                          (1 - wu) * wv * depth_sigma_10 +
                          wu * wv * depth_sigma_11;
      
      float feat_00 = feat[feat_offset + v0 * feat_w * c + u0 * c];
      float feat_01 = feat[feat_offset + v0 * feat_w * c + u1 * c];
      float feat_10 = feat[feat_offset + v1 * feat_w * c + u0 * c];
      float feat_11 = feat[feat_offset + v1 * feat_w * c + u1 * c];
      
      float feature_value = (1 - wu) * (1 - wv) * feat_00 +
                            wu * (1 - wv) * feat_01 +
                            (1 - wu) * wv * feat_10 +
                            wu * wv * feat_11;
      
      float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
      float depth_weight = 0.0f;
      
      if constexpr (DEPTH_DISTRIBUTION == 0) {
        depth_weight = expf(-0.5f * z_score * z_score) / (depth_sigma + epsilon);
      } else {
        depth_weight = 0.5f * expf(-fabsf(z_score)) / (depth_sigma + epsilon);
      }

      accumulator += depth_weight * feature_value;
      valid_count += 1;
    }

    const int* cur_rank = ranks_bev + interval_start;
    float* cur_out = out + *cur_rank * c + cur_c;
    if (valid_count > 0) {
      *cur_out = accumulator / valid_count;
    } else {
      *cur_out = 0.0f;
    }
}

void sampling_vt_pillarpool_fused(int c, int n_intervals,
    const float* depth, const float* feat,
    const float* u_coords, const float* v_coords, const float* z_coords,
    const int* batch_camera_indices, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths,
    int batch_size, int num_cameras, int feat_h, int feat_w, float epsilon, int depth_distribution, float* out) {
  if (depth_distribution == 1) {
    sampling_vt_pillarpool_fused_kernel<1><<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
      c, n_intervals, depth, feat, u_coords, v_coords, z_coords,
      batch_camera_indices, ranks_bev, interval_starts, interval_lengths,
      batch_size, num_cameras, feat_h, feat_w, epsilon, out
    );
  } else {
    sampling_vt_pillarpool_fused_kernel<0><<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
      c, n_intervals, depth, feat, u_coords, v_coords, z_coords,
      batch_camera_indices, ranks_bev, interval_starts, interval_lengths,
      batch_size, num_cameras, feat_h, feat_w, epsilon, out
    );
  }
}

void sampling_vt_pillarpool(int c, int n_intervals, const float* feat,
  const int* ranks_feat, const int* ranks_bev,
  const int* interval_starts, const int* interval_lengths, float* out) {
  sampling_vt_pillarpool_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, feat, ranks_feat, ranks_bev, interval_starts, interval_lengths, out
  );
}

void sampling_vt_pillarpool_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* depth_grad, float* feat_grad) {
  sampling_vt_pillarpool_grad_kernel<<<(int)ceil(((double)n_intervals / 256)), 256>>>(
     c, n_intervals, out_grad, depth, feat, ranks_feat,
     ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}
