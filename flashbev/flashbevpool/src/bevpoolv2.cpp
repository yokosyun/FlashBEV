// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void bev_pool_v2_classification_nearest(int c, int n_intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_v2_classification_nearest_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_depth, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths,
  float* depth_grad, float* feat_grad);

void bev_pool_v2_regression_nearest(int c, int n_intervals, const float* depth, const float* feat,
  const int* ranks_feat, const int* ranks_bev,
  const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_v2_regression_nearest_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths,
  float* depth_grad, float* feat_grad);

void bev_pool_v2_classification_nearest_forward(
  const at::Tensor _depth,
  const at::Tensor _feat,
  at::Tensor _out,
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _feat.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_depth = _ranks_depth.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  float* out = _out.data_ptr<float>();
  bev_pool_v2_classification_nearest(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

void bev_pool_v2_classification_nearest_backward(
  const at::Tensor _out_grad,
  at::Tensor _depth_grad,
  at::Tensor _feat_grad,
  const at::Tensor _depth,
  const at::Tensor _feat,
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  float* depth_grad = _depth_grad.data_ptr<float>();
  float* feat_grad = _feat_grad.data_ptr<float>();
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_depth = _ranks_depth.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  bev_pool_v2_classification_nearest_grad(
    c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}

void bev_pool_v2_regression_nearest_forward(
  const at::Tensor _depth,
  const at::Tensor _feat,
  at::Tensor _out,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _feat.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  float* out = _out.data_ptr<float>();
  bev_pool_v2_regression_nearest(
    c, n_intervals, depth, feat, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

void bev_pool_v2_regression_nearest_backward(
  const at::Tensor _out_grad,
  at::Tensor _depth_grad,
  at::Tensor _feat_grad,
  const at::Tensor _depth,
  const at::Tensor _feat,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  float* depth_grad = _depth_grad.data_ptr<float>();
  float* feat_grad = _feat_grad.data_ptr<float>();
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  bev_pool_v2_regression_nearest_grad(
    c, n_intervals, out_grad, depth, feat, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_v2_classification_nearest_forward", &bev_pool_v2_classification_nearest_forward, "bev_pool_v2_classification_nearest_forward");
  m.def("bev_pool_v2_classification_nearest_backward", &bev_pool_v2_classification_nearest_backward, "bev_pool_v2_classification_nearest_backward");
  m.def("bev_pool_v2_regression_nearest_forward", &bev_pool_v2_regression_nearest_forward, "bev_pool_v2_regression_nearest_forward");
  m.def("bev_pool_v2_regression_nearest_backward", &bev_pool_v2_regression_nearest_backward, "bev_pool_v2_regression_nearest_backward");
}
