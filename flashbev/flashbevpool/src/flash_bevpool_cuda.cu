// Copyright (c) Shunsuke Yokokawa. All rights reserved.

#include <stdio.h>

template <bool USE_SHARED, int DEPTH_DISTRIBUTION, bool OPTIMIZE_Z_PRECOMPUTE = true> // DEPTH_DISTRIBUTION: 0=gaussian, 1=laplace, OPTIMIZE_Z_PRECOMPUTE: optimize z-independent projection
__global__ void flash_bevpool_kernel(
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params, // 12 for 3x4 matrices
    int num_intrinsic_params, // 12 for 3x4 matrices
    float epsilon, // Small value to avoid division by zero
    float depth_weight_threshold, // Threshold for depth weight filtering
    const float *__restrict__ depth_params, // [B,N,H,W,2] - [mean, sigma] for Gaussian depth
    const float *__restrict__ image_feats, // [B,N,H,W,C]
    const float *__restrict__ projection_matrices, // [B,N,3,4]
    const int *__restrict__ feature_size, // [B,N,2]
    const int *__restrict__ image_size, // [2] [image_height, image_width]
    int grid_x, // grid size x
    int grid_y, // grid size y
    int grid_z, // grid size z
    float voxel_size_x, // voxel size x
    float voxel_size_y, // voxel size y
    float voxel_size_z, // voxel size z
    float voxel_x_min, // voxel x min
    float voxel_y_min, // voxel y min
    float voxel_z_min, // voxel z min
    float* __restrict__ out // [B,Y,X,C]
  )
{
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_index = thread_index % num_channels;
  int bev_index = thread_index / num_channels;
  
  int total_bev_locations = batch_size * grid_x * grid_y;
  if (bev_index >= total_bev_locations) {
    return;
  }
  
  int batch_idx = bev_index / (grid_x * grid_y);
  int spatial_idx = bev_index % (grid_x * grid_y);
  int x_idx = spatial_idx % grid_x;
  int y_idx = spatial_idx / grid_x;
  
  float voxel_x = voxel_x_min + (x_idx + 0.5f) * voxel_size_x;
  float voxel_y = voxel_y_min + (y_idx + 0.5f) * voxel_size_y;
  
  int image_height = image_size[0];
  int image_width = image_size[1];

  float accumulator = 0.0f;
  float valid_count = 0.0f;

  const float* proj_base_ptr = nullptr;

  const int num_proj_params = 12;
  const int batch_proj_base = batch_idx * num_cameras * num_proj_params;

  extern __shared__ float smem[];
  float* shmem = nullptr;
  if constexpr (USE_SHARED) {
    shmem = smem;
  }


  for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
    const int proj_camera_offset = cam_idx * num_proj_params;
    
    if constexpr (USE_SHARED) {
      if (threadIdx.x == 0) {
        for (int i = 0; i < num_proj_params; i++) {
          shmem[cam_idx * num_proj_params + i] = projection_matrices[batch_proj_base + proj_camera_offset + i];
        }
      }
      __syncthreads();
      proj_base_ptr = shmem + cam_idx * num_proj_params;
    } else {
      proj_base_ptr = projection_matrices + batch_proj_base + proj_camera_offset;
    }

    float img_u_z0, img_v_z0, img_d_z0;
    float z_coeff_u, z_coeff_v, z_coeff_d;
    
    if constexpr (OPTIMIZE_Z_PRECOMPUTE) {
      img_u_z0 = proj_base_ptr[0] * voxel_x + 
                 proj_base_ptr[1] * voxel_y + 
                 proj_base_ptr[3];
      z_coeff_u = proj_base_ptr[2];
      
      img_v_z0 = proj_base_ptr[4] * voxel_x + 
                 proj_base_ptr[5] * voxel_y + 
                 proj_base_ptr[7];
      z_coeff_v = proj_base_ptr[6];
      
      img_d_z0 = proj_base_ptr[8] * voxel_x + 
                 proj_base_ptr[9] * voxel_y + 
                 proj_base_ptr[11];
      z_coeff_d = proj_base_ptr[10];
    }

    for (int z_idx = 0; z_idx < grid_z; z_idx++) {
      float voxel_z = voxel_z_min + (z_idx + 0.5f) * voxel_size_z;
      
      float img_u, img_v, img_d;
      
      if constexpr (OPTIMIZE_Z_PRECOMPUTE) {
        img_u = img_u_z0 + z_coeff_u * voxel_z;
        img_v = img_v_z0 + z_coeff_v * voxel_z;
        img_d = img_d_z0 + z_coeff_d * voxel_z;
      } else {
        img_u = proj_base_ptr[0] * voxel_x + 
                proj_base_ptr[1] * voxel_y + 
                proj_base_ptr[2] * voxel_z + 
                proj_base_ptr[3];
        img_v = proj_base_ptr[4] * voxel_x + 
                proj_base_ptr[5] * voxel_y + 
                proj_base_ptr[6] * voxel_z + 
                proj_base_ptr[7];
        img_d = proj_base_ptr[8] * voxel_x + 
                proj_base_ptr[9] * voxel_y + 
                proj_base_ptr[10] * voxel_z + 
                proj_base_ptr[11];
      }
      
      if (img_d <= 0.0f) {
        continue;
      }
      
      // Normalize u, v by depth (convert from camera coordinates to image coordinates)
      float clamped_d = (img_d >= 0.0f) ? fmaxf(img_d, epsilon) : fminf(img_d, -epsilon);
      img_u = img_u / clamped_d;
      img_v = img_v / clamped_d;

      int feat_h = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 0];
      int feat_w = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 1];
      
      float feat_u = img_u / image_width * feat_w;
      float feat_v = img_v / image_height * feat_h;
      
      if (feat_u < 0.0f || feat_u >= feat_w || feat_v < 0.0f || feat_v >= feat_h) {
        continue;
      }
      
      int batch_cam_idx = batch_idx * num_cameras + cam_idx;
      int feat_hw = feat_h * feat_w;

      int u0 = static_cast<int>(floorf(feat_u));
      int v0 = static_cast<int>(floorf(feat_v));
      float du = feat_u - u0;
      float dv = feat_v - v0;
      u0 = max(0, min(u0, feat_w - 1));
      int u1 = max(0, min(u0 + 1, feat_w - 1));
      v0 = max(0, min(v0, feat_h - 1));
      int v1 = max(0, min(v0 + 1, feat_h - 1));
      
      float w00 = (1.0f - du) * (1.0f - dv);
      float w01 = (1.0f - du) * dv;
      float w10 = du * (1.0f - dv);
      float w11 = du * dv;

      int depth_base = batch_cam_idx * feat_hw * 2;
      int depth_idx00 = depth_base + v0 * feat_w * 2 + u0 * 2;
      int depth_idx01 = depth_base + v1 * feat_w * 2 + u0 * 2;
      int depth_idx10 = depth_base + v0 * feat_w * 2 + u1 * 2;
      int depth_idx11 = depth_base + v1 * feat_w * 2 + u1 * 2;
      
      float depth_mean = w00 * depth_params[depth_idx00] + 
                         w01 * depth_params[depth_idx01] + 
                         w10 * depth_params[depth_idx10] + 
                         w11 * depth_params[depth_idx11];
      float depth_sigma = w00 * depth_params[depth_idx00 + 1] + 
                          w01 * depth_params[depth_idx01 + 1] + 
                          w10 * depth_params[depth_idx10 + 1] + 
                          w11 * depth_params[depth_idx11 + 1];
      
      float depth_weight = 0.0f;

      if constexpr (DEPTH_DISTRIBUTION == 0) {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = expf(-0.5f * z_score * z_score) / (depth_sigma + epsilon);
      } else {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = 0.5f * expf(-fabsf(z_score)) / (depth_sigma + epsilon);
      }
      
      if (depth_weight < depth_weight_threshold) {
        continue;
      }
      
      int idx00 = batch_cam_idx * feat_hw * num_channels + v0 * feat_w * num_channels + u0 * num_channels;
      int idx01 = batch_cam_idx * feat_hw * num_channels + v1 * feat_w * num_channels + u0 * num_channels;
      int idx10 = batch_cam_idx * feat_hw * num_channels + v0 * feat_w * num_channels + u1 * num_channels;
      int idx11 = batch_cam_idx * feat_hw * num_channels + v1 * feat_w * num_channels + u1 * num_channels;
      
      float feat_value = w00 * image_feats[idx00 + channel_index] + w01 * image_feats[idx01 + channel_index] + 
                         w10 * image_feats[idx10 + channel_index] + w11 * image_feats[idx11 + channel_index];

      accumulator += feat_value * depth_weight;
      valid_count += 1;
    }
    
    if constexpr (USE_SHARED) {
      __syncthreads();
    }
  }

  int out_base = 
  (grid_y * grid_x * num_channels) * batch_idx +
  (grid_x * num_channels) * y_idx +
  (num_channels) * x_idx;

  if (valid_count > 0) {
    out[out_base + channel_index] = accumulator / valid_count;
  } else {
    out[out_base + channel_index] = 0.0f;
  }
}


template <int DEPTH_DISTRIBUTION, bool USE_VECTORIZED_LOAD = false>
__global__ void flash_bevpool_warp_kernel(
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params, // 12 for 3x4 matrices
    int num_intrinsic_params, // 12 for 3x4 matrices
    float epsilon, // Small value to avoid division by zero
    float depth_weight_threshold, // Threshold for depth weight filtering
    const float *__restrict__ depth_params, // [B,N,H,W,2] - [mean, sigma] for Gaussian depth
    const float *__restrict__ image_feats, // [B,N,H,W,C]
    const float *__restrict__ projection_matrices, // [B,N,3,4]
    const int *__restrict__ feature_size, // [B,N,2]
    const int *__restrict__ image_size, // [2] [image_height, image_width]
    int grid_x, // grid size x
    int grid_y, // grid size y
    int grid_z, // grid size z
    float voxel_size_x, // voxel size x
    float voxel_size_y, // voxel size y
    float voxel_size_z, // voxel size z
    float voxel_x_min, // voxel x min
    float voxel_y_min, // voxel y min
    float voxel_z_min, // voxel z min
    float* __restrict__ out // [B,Y,X,C]
  )
{
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int bev_index = thread_index;
  int total_bev_locations = batch_size * grid_x * grid_y;
  
  if (bev_index >= total_bev_locations) {
    return;
  }
  
  int batch_idx = bev_index / (grid_x * grid_y);
  int spatial_idx = bev_index % (grid_x * grid_y);
  int x_idx = spatial_idx % grid_x;
  int y_idx = spatial_idx / grid_x;
  
  float voxel_x = voxel_x_min + (x_idx + 0.5f) * voxel_size_x;
  float voxel_y = voxel_y_min + (y_idx + 0.5f) * voxel_size_y;
  
  int image_height = image_size[0];
  int image_width = image_size[1];

  extern __shared__ float smem[];
  float* accumulators = smem + threadIdx.x * num_channels;
  for (int c = 0; c < num_channels; c++) {
    accumulators[c] = 0.0f;
  }
  int valid_count = 0;

  const int num_proj_params = 12;
  const int batch_proj_base = batch_idx * num_cameras * num_proj_params;

  for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
    const int proj_camera_offset = cam_idx * num_proj_params;
    const float* proj_base_ptr = projection_matrices + batch_proj_base + proj_camera_offset;

    for (int z_idx = 0; z_idx < grid_z; z_idx++) {
      float voxel_z = voxel_z_min + (z_idx + 0.5f) * voxel_size_z;
      
      float img_u = proj_base_ptr[0] * voxel_x + 
                    proj_base_ptr[1] * voxel_y + 
                    proj_base_ptr[2] * voxel_z + 
                    proj_base_ptr[3];
      float img_v = proj_base_ptr[4] * voxel_x + 
                    proj_base_ptr[5] * voxel_y + 
                    proj_base_ptr[6] * voxel_z + 
                    proj_base_ptr[7];
      float img_d = proj_base_ptr[8] * voxel_x + 
                    proj_base_ptr[9] * voxel_y + 
                    proj_base_ptr[10] * voxel_z + 
                    proj_base_ptr[11];
      
      if (img_d <= 0.0f) {
        continue;
      }
      
      float clamped_d = (img_d >= 0.0f) ? fmaxf(img_d, epsilon) : fminf(img_d, -epsilon);
      img_u = img_u / clamped_d;
      img_v = img_v / clamped_d;
      
      int feat_h = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 0];
      int feat_w = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 1];
      
      float feat_u = img_u / image_width * feat_w;
      float feat_v = img_v / image_height * feat_h;
      
      if (feat_u < 0.0f || feat_u >= feat_w || feat_v < 0.0f || feat_v >= feat_h) {
        continue;
      }
      
      int batch_cam_idx = batch_idx * num_cameras + cam_idx;
      int feat_hw = feat_h * feat_w;

      int u0 = static_cast<int>(floorf(feat_u));
      int v0 = static_cast<int>(floorf(feat_v));
      float du = feat_u - u0;
      float dv = feat_v - v0;
      u0 = max(0, min(u0, feat_w - 1));
      int u1 = max(0, min(u0 + 1, feat_w - 1));
      v0 = max(0, min(v0, feat_h - 1));
      int v1 = max(0, min(v0 + 1, feat_h - 1));
      
      float w00 = (1.0f - du) * (1.0f - dv);
      float w01 = (1.0f - du) * dv;
      float w10 = du * (1.0f - dv);
      float w11 = du * dv;

      int depth_base = batch_cam_idx * feat_hw * 2;
      int depth_idx00 = depth_base + v0 * feat_w * 2 + u0 * 2;
      int depth_idx01 = depth_base + v1 * feat_w * 2 + u0 * 2;
      int depth_idx10 = depth_base + v0 * feat_w * 2 + u1 * 2;
      int depth_idx11 = depth_base + v1 * feat_w * 2 + u1 * 2;
      
      float depth_mean = w00 * depth_params[depth_idx00] + 
                         w01 * depth_params[depth_idx01] + 
                         w10 * depth_params[depth_idx10] + 
                         w11 * depth_params[depth_idx11];
      float depth_sigma = w00 * depth_params[depth_idx00 + 1] + 
                          w01 * depth_params[depth_idx01 + 1] + 
                          w10 * depth_params[depth_idx10 + 1] + 
                          w11 * depth_params[depth_idx11 + 1];
      
      float depth_weight = 0.0f;

      if constexpr (DEPTH_DISTRIBUTION == 0) {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = expf(-0.5f * z_score * z_score) / (depth_sigma + epsilon);
      } else {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = 0.5f * expf(-fabsf(z_score)) / (depth_sigma + epsilon);
      }
      
      if (depth_weight < depth_weight_threshold) {
        continue;
      }
      
      int idx00 = batch_cam_idx * feat_hw * num_channels + v0 * feat_w * num_channels + u0 * num_channels;
      int idx01 = batch_cam_idx * feat_hw * num_channels + v1 * feat_w * num_channels + u0 * num_channels;
      int idx10 = batch_cam_idx * feat_hw * num_channels + v0 * feat_w * num_channels + u1 * num_channels;
      int idx11 = batch_cam_idx * feat_hw * num_channels + v1 * feat_w * num_channels + u1 * num_channels;
      
      if constexpr (USE_VECTORIZED_LOAD) {
        for (int channel_index = 0; channel_index < num_channels; channel_index += 4) {
          if (channel_index + 3 < num_channels) {
            const float* ptr00 = &image_feats[idx00 + channel_index];
            const float* ptr01 = &image_feats[idx01 + channel_index];
            const float* ptr10 = &image_feats[idx10 + channel_index];
            const float* ptr11 = &image_feats[idx11 + channel_index];
            
            float4 feat00 = make_float4(ptr00[0], ptr00[1], ptr00[2], ptr00[3]);
            float4 feat01 = make_float4(ptr01[0], ptr01[1], ptr01[2], ptr01[3]);
            float4 feat10 = make_float4(ptr10[0], ptr10[1], ptr10[2], ptr10[3]);
            float4 feat11 = make_float4(ptr11[0], ptr11[1], ptr11[2], ptr11[3]);
            
            float4 feat_value;
            feat_value.x = w00 * feat00.x + w01 * feat01.x + w10 * feat10.x + w11 * feat11.x;
            feat_value.y = w00 * feat00.y + w01 * feat01.y + w10 * feat10.y + w11 * feat11.y;
            feat_value.z = w00 * feat00.z + w01 * feat01.z + w10 * feat10.z + w11 * feat11.z;
            feat_value.w = w00 * feat00.w + w01 * feat01.w + w10 * feat10.w + w11 * feat11.w;
            
            accumulators[channel_index + 0] += feat_value.x * depth_weight;
            accumulators[channel_index + 1] += feat_value.y * depth_weight;
            accumulators[channel_index + 2] += feat_value.z * depth_weight;
            accumulators[channel_index + 3] += feat_value.w * depth_weight;
          } else {
            for (int c = channel_index; c < num_channels; c++) {
              float feat_value = w00 * image_feats[idx00 + c] + 
                                w01 * image_feats[idx01 + c] + 
                                w10 * image_feats[idx10 + c] + 
                                w11 * image_feats[idx11 + c];
              accumulators[c] += feat_value * depth_weight;
            }
          }
        }
      } else {
        for (int channel_index = 0; channel_index < num_channels; channel_index++) {
          float feat_value = w00 * image_feats[idx00 + channel_index] + 
                            w01 * image_feats[idx01 + channel_index] + 
                            w10 * image_feats[idx10 + channel_index] + 
                            w11 * image_feats[idx11 + channel_index];
          accumulators[channel_index] += feat_value * depth_weight;
        }
      }
      valid_count += 1;
    }
  }
  
  int out_base = 
    (grid_y * grid_x * num_channels) * batch_idx +
    (grid_x * num_channels) * y_idx +
    num_channels * x_idx;

  if (valid_count > 0) {
    for (int channel_index = 0; channel_index < num_channels; channel_index++) {
      out[out_base + channel_index] = accumulators[channel_index] / valid_count;
    }
  } else {
    for (int channel_index = 0; channel_index < num_channels; channel_index++) {
      out[out_base + channel_index] = 0.0f;
    }
  }
}

template <int DEPTH_DISTRIBUTION> // DEPTH_DISTRIBUTION: 0=gaussian, 1=laplace
__global__ void flash_bevpool_grad_kernel(
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params,
    int num_intrinsic_params,
    float epsilon,
    float depth_weight_threshold, // Threshold for depth weight filtering
    const float* out_grad,
    const float* depth_params,
    const float* image_feats,
    const float* projection_matrices,
    const int* feature_size,
    const int* image_size,
    const float* roi_range,
    int grid_x, // grid size x
    int grid_y, // grid size y
    int grid_z, // grid size z
    float* depth_params_grad,
    float* image_feats_grad
) {
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_index = thread_index % num_channels;
  int bev_index = thread_index / num_channels;
  
  int batch_idx = bev_index / (grid_x * grid_y);
  int spatial_idx = bev_index % (grid_x * grid_y);
  int x_idx = spatial_idx % grid_x;
  int y_idx = spatial_idx / grid_x;
  
  if (batch_idx >= batch_size) return;
  
  int out_offset = batch_idx * grid_y * grid_x * num_channels + 
                   y_idx * grid_x * num_channels + 
                   x_idx * num_channels + 
                   channel_index;
  float grad_out = out_grad[out_offset];
  
  float voxel_size_x = (roi_range[1] - roi_range[0]) / grid_x;
  float voxel_size_y = (roi_range[3] - roi_range[2]) / grid_y;
  float voxel_size_z = (roi_range[5] - roi_range[4]) / grid_z;
  
  float voxel_x = roi_range[0] + (x_idx + 0.5f) * voxel_size_x;
  float voxel_y = roi_range[2] + (y_idx + 0.5f) * voxel_size_y;
  
  const int num_proj_params = 12;
  const int batch_proj_base = batch_idx * num_cameras * num_proj_params;
  
  float valid_count = 0.0f;
  
  for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
    const int proj_camera_offset = cam_idx * num_proj_params;
    
    for (int z_idx = 0; z_idx < grid_z; z_idx++) {
      float voxel_z = roi_range[4] + (z_idx + 0.5f) * voxel_size_z;
      
      float img_d = projection_matrices[batch_proj_base + proj_camera_offset + 8] * voxel_x + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 9] * voxel_y + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 10] * voxel_z + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 11];
      
      if (img_d <= 0.0f) {
        continue;
      }
      
      float img_u = projection_matrices[batch_proj_base + proj_camera_offset + 0] * voxel_x + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 1] * voxel_y + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 2] * voxel_z + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 3];
      float img_v = projection_matrices[batch_proj_base + proj_camera_offset + 4] * voxel_x + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 5] * voxel_y + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 6] * voxel_z + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 7];
      
      float clamped_d = (img_d >= 0.0f) ? fmaxf(img_d, epsilon) : fminf(img_d, -epsilon);
      img_u = img_u / clamped_d;
      img_v = img_v / clamped_d;
      
      int image_height = image_size[0];
      int image_width = image_size[1];
      
      int feat_h = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 0];
      int feat_w = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 1];
      
      float feat_u = img_u / image_width * feat_w;
      float feat_v = img_v / image_height * feat_h;
      
      if (feat_u < 0.0f || feat_u >= feat_w || feat_v < 0.0f || feat_v >= feat_h) {
        continue;
      }
      
      int batch_cam_idx = batch_idx * num_cameras + cam_idx;
      int feat_hw = feat_h * feat_w;
      int feat_c = num_channels;
      
      int u0 = static_cast<int>(floorf(feat_u));
      int v0 = static_cast<int>(floorf(feat_v));
      float du = feat_u - u0;
      float dv = feat_v - v0;
      u0 = max(0, min(u0, feat_w - 1));
      int u1 = max(0, min(u0 + 1, feat_w - 1));
      v0 = max(0, min(v0, feat_h - 1));
      int v1 = max(0, min(v0 + 1, feat_h - 1));
      
      float w00 = (1.0f - du) * (1.0f - dv);
      float w01 = (1.0f - du) * dv;
      float w10 = du * (1.0f - dv);
      float w11 = du * dv;
      
      int depth_base = batch_cam_idx * feat_hw * 2;
      int depth_idx00 = depth_base + v0 * feat_w * 2 + u0 * 2;
      int depth_idx01 = depth_base + v1 * feat_w * 2 + u0 * 2;
      int depth_idx10 = depth_base + v0 * feat_w * 2 + u1 * 2;
      int depth_idx11 = depth_base + v1 * feat_w * 2 + u1 * 2;
      
      float depth_mean = w00 * depth_params[depth_idx00] + 
                         w01 * depth_params[depth_idx01] + 
                         w10 * depth_params[depth_idx10] + 
                         w11 * depth_params[depth_idx11];
      float depth_sigma = w00 * depth_params[depth_idx00 + 1] + 
                          w01 * depth_params[depth_idx01 + 1] + 
                          w10 * depth_params[depth_idx10 + 1] + 
                          w11 * depth_params[depth_idx11 + 1];
      
      float depth_weight = 0.0f;
      if constexpr (DEPTH_DISTRIBUTION == 0) {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = expf(-0.5f * z_score * z_score) / (depth_sigma + epsilon);
      } else {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = 0.5f * expf(-fabsf(z_score)) / (depth_sigma + epsilon);
      }
      
      if (depth_weight < depth_weight_threshold) {
        continue;
      }
      
      valid_count += 1.0f;
    }
  }
  
  if (valid_count == 0.0f) {
    return;
  }
  
  float grad_out_normalized = grad_out / valid_count;
  
  for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
    const int proj_camera_offset = cam_idx * num_proj_params;
    
    for (int z_idx = 0; z_idx < grid_z; z_idx++) {
      float voxel_z = roi_range[4] + (z_idx + 0.5f) * voxel_size_z;
      
      float img_d = projection_matrices[batch_proj_base + proj_camera_offset + 8] * voxel_x + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 9] * voxel_y + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 10] * voxel_z + 
                     projection_matrices[batch_proj_base + proj_camera_offset + 11];
      
      if (img_d <= 0.0f) {
        continue;
      }
      
      float img_u = projection_matrices[batch_proj_base + proj_camera_offset + 0] * voxel_x + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 1] * voxel_y + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 2] * voxel_z + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 3];
      float img_v = projection_matrices[batch_proj_base + proj_camera_offset + 4] * voxel_x + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 5] * voxel_y + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 6] * voxel_z + 
                    projection_matrices[batch_proj_base + proj_camera_offset + 7];
      
      float clamped_d = (img_d >= 0.0f) ? fmaxf(img_d, epsilon) : fminf(img_d, -epsilon);
      img_u = img_u / clamped_d;
      img_v = img_v / clamped_d;
      
      int image_height = image_size[0];
      int image_width = image_size[1];
      
      int feat_h = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 0];
      int feat_w = feature_size[batch_idx * num_cameras * 2 + cam_idx * 2 + 1];
      
      float feat_u = img_u / image_width * feat_w;
      float feat_v = img_v / image_height * feat_h;
      
      if (feat_u < 0.0f || feat_u >= feat_w || feat_v < 0.0f || feat_v >= feat_h) {
        continue;
      }
      
      int batch_cam_idx = batch_idx * num_cameras + cam_idx;
      int feat_hw = feat_h * feat_w;
      int feat_c = num_channels;
      
      int u0 = static_cast<int>(floorf(feat_u));
      int v0 = static_cast<int>(floorf(feat_v));
      float du = feat_u - u0;
      float dv = feat_v - v0;
      u0 = max(0, min(u0, feat_w - 1));
      int u1 = max(0, min(u0 + 1, feat_w - 1));
      v0 = max(0, min(v0, feat_h - 1));
      int v1 = max(0, min(v0 + 1, feat_h - 1));
      
      float w00 = (1.0f - du) * (1.0f - dv);
      float w01 = (1.0f - du) * dv;
      float w10 = du * (1.0f - dv);
      float w11 = du * dv;
      
      int depth_base = batch_cam_idx * feat_hw * 2;
      int depth_idx00 = depth_base + v0 * feat_w * 2 + u0 * 2;
      int depth_idx01 = depth_base + v1 * feat_w * 2 + u0 * 2;
      int depth_idx10 = depth_base + v0 * feat_w * 2 + u1 * 2;
      int depth_idx11 = depth_base + v1 * feat_w * 2 + u1 * 2;
      
      float depth_mean = w00 * depth_params[depth_idx00] + 
                         w01 * depth_params[depth_idx01] + 
                         w10 * depth_params[depth_idx10] + 
                         w11 * depth_params[depth_idx11];
      float depth_sigma = w00 * depth_params[depth_idx00 + 1] + 
                          w01 * depth_params[depth_idx01 + 1] + 
                          w10 * depth_params[depth_idx10 + 1] + 
                          w11 * depth_params[depth_idx11 + 1];
      
      float depth_weight = 0.0f;
      if constexpr (DEPTH_DISTRIBUTION == 0) {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = expf(-0.5f * z_score * z_score) / (depth_sigma + epsilon);
      } else {
        float z_score = (img_d - depth_mean) / (depth_sigma + epsilon);
        depth_weight = 0.5f * expf(-fabsf(z_score)) / (depth_sigma + epsilon);
      }
      
      if (depth_weight < depth_weight_threshold) {
        continue;
      }
      
      int idx00 = batch_cam_idx * feat_hw * feat_c + v0 * feat_w * feat_c + u0 * feat_c + channel_index;
      int idx01 = batch_cam_idx * feat_hw * feat_c + v1 * feat_w * feat_c + u0 * feat_c + channel_index;
      int idx10 = batch_cam_idx * feat_hw * feat_c + v0 * feat_w * feat_c + u1 * feat_c + channel_index;
      int idx11 = batch_cam_idx * feat_hw * feat_c + v1 * feat_w * feat_c + u1 * feat_c + channel_index;
      
      float feat_value = w00 * image_feats[idx00] + w01 * image_feats[idx01] + 
                         w10 * image_feats[idx10] + w11 * image_feats[idx11];
      
      float weight = depth_weight * grad_out_normalized;
      
      atomicAdd(&depth_params_grad[depth_idx00], weight * feat_value * w00);
      atomicAdd(&depth_params_grad[depth_idx00 + 1], weight * feat_value * w00);
      atomicAdd(&depth_params_grad[depth_idx01], weight * feat_value * w01);
      atomicAdd(&depth_params_grad[depth_idx01 + 1], weight * feat_value * w01);
      atomicAdd(&depth_params_grad[depth_idx10], weight * feat_value * w10);
      atomicAdd(&depth_params_grad[depth_idx10 + 1], weight * feat_value * w10);
      atomicAdd(&depth_params_grad[depth_idx11], weight * feat_value * w11);
      atomicAdd(&depth_params_grad[depth_idx11 + 1], weight * feat_value * w11);
      
      atomicAdd(&image_feats_grad[idx00], weight * w00);
      atomicAdd(&image_feats_grad[idx01], weight * w01);
      atomicAdd(&image_feats_grad[idx10], weight * w10);
      atomicAdd(&image_feats_grad[idx11], weight * w11);
    }
  }
}

void flash_bevpool_warp(
    int total_bev_locations,
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params,
    int num_intrinsic_params,
    int depth_distribution,
    bool use_vectorized_load,
    float epsilon,
    float depth_weight_threshold,
    const float* depth_params,
    const float* image_feats,
    const float* projection_matrices,
    const int* feature_size,
    const int* image_size,
    int grid_x,
    int grid_y,
    int grid_z,
    float voxel_size_x,
    float voxel_size_y,
    float voxel_size_z,
    float voxel_x_min,
    float voxel_y_min,
    float voxel_z_min,
    float* out
) {
  const size_t max_shmem_per_block = 48 * 1024;
  int threads_per_block = 256;
  size_t shmem_size = static_cast<size_t>(threads_per_block) * num_channels * sizeof(float);
  
  if (shmem_size > max_shmem_per_block) {
    threads_per_block = static_cast<int>(max_shmem_per_block / (num_channels * sizeof(float)));
    if (threads_per_block < 32) {
      threads_per_block = 32;
    }
    threads_per_block = (threads_per_block + 31) / 32 * 32;
    shmem_size = static_cast<size_t>(threads_per_block) * num_channels * sizeof(float);
  }
  
  int num_blocks = (total_bev_locations + threads_per_block - 1) / threads_per_block;
  
  if (depth_distribution == 1) {
    if (use_vectorized_load) {
      flash_bevpool_warp_kernel<1, true><<<num_blocks, threads_per_block, shmem_size>>>(
        num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
        epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
        feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
      );
    } else {
      flash_bevpool_warp_kernel<1, false><<<num_blocks, threads_per_block, shmem_size>>>(
        num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
        epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
        feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
      );
    }
  } else {
    if (use_vectorized_load) {
      flash_bevpool_warp_kernel<0, true><<<num_blocks, threads_per_block, shmem_size>>>(
        num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
        epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
        feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
      );
    } else {
      flash_bevpool_warp_kernel<0, false><<<num_blocks, threads_per_block, shmem_size>>>(
        num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
        epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
        feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
      );
    }
  }
}

void flash_bevpool(
    int total_threads,
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params,
    int num_intrinsic_params,
    int depth_distribution,
    bool use_shared_memory,
    bool optimize_z_precompute,
    bool use_warp_kernel,
    bool use_vectorized_load,
    float epsilon,
    float depth_weight_threshold,
    const float* depth_params,
    const float* image_feats,
    const float* projection_matrices,
    const int* feature_size,
    const int* image_size,
    int grid_x,
    int grid_y,
    int grid_z,
    float voxel_size_x,
    float voxel_size_y,
    float voxel_size_z,
    float voxel_x_min,
    float voxel_y_min,
    float voxel_z_min,
    float* out
) {
  
  if (use_warp_kernel) {
    int total_bev_locations = batch_size * grid_x * grid_y;
    flash_bevpool_warp(
      total_bev_locations, num_channels, num_cameras, batch_size,
      num_extrinsic_params, num_intrinsic_params, depth_distribution,
      use_vectorized_load, epsilon, depth_weight_threshold, depth_params, image_feats,
      projection_matrices, feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
    );
    return;
  }
  
  int threads_per_block = 256;
  int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
  const int num_proj_params = 12;

  if (use_shared_memory) {
    size_t shmem = static_cast<size_t>(num_cameras) * num_proj_params * sizeof(float);
    if (depth_distribution == 1) {
      if (optimize_z_precompute) {
        flash_bevpool_kernel<true, 1, true><<<num_blocks, threads_per_block, shmem>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      } else {
        flash_bevpool_kernel<true, 1, false><<<num_blocks, threads_per_block, shmem>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      }
    } else {
      if (optimize_z_precompute) {
        flash_bevpool_kernel<true, 0, true><<<num_blocks, threads_per_block, shmem>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      } else {
        flash_bevpool_kernel<true, 0, false><<<num_blocks, threads_per_block, shmem>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      }
    }
  } else {
    if (depth_distribution == 1) {
      if (optimize_z_precompute) {
        flash_bevpool_kernel<false, 1, true><<<num_blocks, threads_per_block, 0>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      } else {
        flash_bevpool_kernel<false, 1, false><<<num_blocks, threads_per_block, 0>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      }
    } else {
      if (optimize_z_precompute) {
        flash_bevpool_kernel<false, 0, true><<<num_blocks, threads_per_block, 0>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      } else {
        flash_bevpool_kernel<false, 0, false><<<num_blocks, threads_per_block, 0>>>(
          num_channels, num_cameras, batch_size, num_extrinsic_params, num_intrinsic_params,
          epsilon, depth_weight_threshold, depth_params, image_feats, projection_matrices,
          feature_size, image_size, grid_x, grid_y, grid_z, voxel_size_x, voxel_size_y, voxel_size_z, voxel_x_min, voxel_y_min, voxel_z_min, out
        );
      }
    }
  }
}

void flash_bevpool_grad(
    int total_threads,
    int num_channels,
    int num_cameras,
    int batch_size,
    int num_extrinsic_params,
    int num_intrinsic_params,
    int depth_distribution,
    float epsilon,
    float depth_weight_threshold,
    const float* out_grad,
    const float* depth_params,
    const float* image_feats,
    const float* projection_matrices,
    const int* feature_size,
    const int* image_size,
    const float* roi_range,
    int grid_x,
    int grid_y,
    int grid_z,
    float* depth_params_grad,
    float* image_feats_grad
) {
  int threads_per_block = 256;
  int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
  
  if (depth_distribution == 1) {
    flash_bevpool_grad_kernel<1><<<num_blocks, threads_per_block>>>(
      num_channels,
      num_cameras,
      batch_size,
      num_extrinsic_params,
      num_intrinsic_params,
      epsilon,
      depth_weight_threshold,
      out_grad,
      depth_params,
      image_feats,
      projection_matrices,
      feature_size,
      image_size,
      roi_range,
      grid_x,
      grid_y,
      grid_z,
      depth_params_grad,
      image_feats_grad
    );
  } else {
    flash_bevpool_grad_kernel<0><<<num_blocks, threads_per_block>>>(
      num_channels,
      num_cameras,
      batch_size,
      num_extrinsic_params,
      num_intrinsic_params,
      epsilon,
      depth_weight_threshold,
      out_grad,
      depth_params,
      image_feats,
      projection_matrices,
      feature_size,
      image_size,
      roi_range,
      grid_x,
      grid_y,
      grid_z,
      depth_params_grad,
      image_feats_grad
    );
  }
}
