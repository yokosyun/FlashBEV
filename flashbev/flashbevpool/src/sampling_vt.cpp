// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

void sampling_vt_pillarpool_fused(int c, int n_intervals,
    const float* depth, const float* feat,
    const float* u_coords, const float* v_coords, const float* z_coords,
    const int* batch_camera_indices, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths,
    int batch_size, int num_cameras, int feat_h, int feat_w, float epsilon, int depth_distribution, float* out);

void sampling_vt_pillarpool_fused_forward(
  const at::Tensor _depth,
  const at::Tensor _feat,
  const at::Tensor _u_coords,
  const at::Tensor _v_coords,
  const at::Tensor _z_coords,
  const at::Tensor _batch_camera_indices,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  at::Tensor _out,
  int batch_size,
  int num_cameras,
  int feat_h,
  int feat_w,
  float epsilon,
  int depth_distribution
) {
  int c = _feat.size(-1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat));
  
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const float* u_coords = _u_coords.data_ptr<float>();
  const float* v_coords = _v_coords.data_ptr<float>();
  const float* z_coords = _z_coords.data_ptr<float>();
  const int* batch_camera_indices = _batch_camera_indices.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  float* out = _out.data_ptr<float>();
  
  sampling_vt_pillarpool_fused(
    c, n_intervals, depth, feat, u_coords, v_coords, z_coords,
    batch_camera_indices, ranks_bev, interval_starts, interval_lengths,
    batch_size, num_cameras, feat_h, feat_w, epsilon, depth_distribution, out
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sampling_vt_pillarpool_fused_forward", &sampling_vt_pillarpool_fused_forward, "sampling_vt_pillarpool_fused_forward");
}
