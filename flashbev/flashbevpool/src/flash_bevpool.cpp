// Copyright (c) Shunsuke Yokokawa. All rights reserved.
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

// CUDA function declarations
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
);

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
);

/*
  Function: BEV Pool V3 Fused (forward, cuda)
  Args:
    depth_params      : Gaussian depth parameters [B,N,H,W,2] - [mean, sigma]
    image_feats       : Image features [B,N,H,W,C]
    projection_matrices: Camera projection matrices [B,N,3,4]
    feature_size      : Feature dimensions [B,N,2]
    grid_size         : Grid dimensions [3] [grid_x,grid_y,grid_z]
    voxel_size_x      : Voxel size in x direction
    voxel_size_y      : Voxel size in y direction
    voxel_size_z      : Voxel size in z direction
    voxel_x_min       : Minimum x coordinate
    voxel_y_min       : Minimum y coordinate
    voxel_z_min       : Minimum z coordinate
    out               : Output BEV features [B,X,Y,C]
*/
void flash_bevpool_forward(
  const at::Tensor _depth_params,
  const at::Tensor _image_feats,
  const at::Tensor _projection_matrices,
  const at::Tensor _feature_size,
  const at::Tensor _image_size,
  const at::Tensor _grid_size,
  float voxel_size_x,
  float voxel_size_y,
  float voxel_size_z,
  float voxel_x_min,
  float voxel_y_min,
  float voxel_z_min,
  at::Tensor _out,
  int total_threads,
  int depth_distribution,
  bool use_shared_memory,
  bool optimize_z_precompute,
  bool use_warp_kernel,
  bool use_vectorized_load,
  float epsilon = 1e-6f,
  float depth_weight_threshold = 1e-6f
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth_params));
  
  // Get tensor dimensions
  int batch_size = _depth_params.size(0);
  int num_cameras = _depth_params.size(1);
  int num_channels = _image_feats.size(-1);
  
  // Get tensor data pointers
  const float* depth_params = _depth_params.data_ptr<float>();
  const float* image_feats = _image_feats.data_ptr<float>();
  const float* projection_matrices = _projection_matrices.data_ptr<float>();
  const int* feature_size = _feature_size.data_ptr<int>();
  const int* image_size = _image_size.data_ptr<int>();
  float* out = _out.data_ptr<float>();
  
  int grid_x, grid_y, grid_z;
  if (_grid_size.is_cuda()) {
    int grid_size_host[3];
    const int* grid_size = _grid_size.data_ptr<int>();
    cudaMemcpy(grid_size_host, grid_size, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    grid_x = grid_size_host[0];
    grid_y = grid_size_host[1];
    grid_z = grid_size_host[2];
  } else {
    const int* grid_size = _grid_size.data_ptr<int>();
    grid_x = grid_size[0];
    grid_y = grid_size[1];
    grid_z = grid_size[2];
  }
  
  // Matrix dimensions (3x4 matrices)
  int num_extrinsic_params = 12;
  int num_intrinsic_params = 12;
  
  flash_bevpool(
    total_threads,
    num_channels,
    num_cameras, 
    batch_size, 
    num_extrinsic_params, 
    num_intrinsic_params,
    depth_distribution,
    use_shared_memory,
    optimize_z_precompute,
    use_warp_kernel,
    use_vectorized_load,
    epsilon,
    depth_weight_threshold,
    depth_params, 
    image_feats, 
    projection_matrices, 
    feature_size,
    image_size,
    grid_x,
    grid_y,
    grid_z,
    voxel_size_x,
    voxel_size_y,
    voxel_size_z,
    voxel_x_min,
    voxel_y_min,
    voxel_z_min,
    out
  );
}

void flash_bevpool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _depth_params,
  const at::Tensor _image_feats,
  const at::Tensor _projection_matrices,
  const at::Tensor _feature_size,
  const at::Tensor _image_size,
  const at::Tensor _roi_range,
  const at::Tensor _grid_size,
  at::Tensor _depth_params_grad,
  at::Tensor _image_feats_grad,
  int total_threads,
  int depth_distribution,
  float epsilon = 1e-6f,
  float depth_weight_threshold = 1e-6f
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  
  int batch_size = _depth_params.size(0);
  int num_cameras = _depth_params.size(1);
  int num_channels = _image_feats.size(-1);
  
  const float* out_grad = _out_grad.data_ptr<float>();
  const float* depth_params = _depth_params.data_ptr<float>();
  const float* image_feats = _image_feats.data_ptr<float>();
  const float* projection_matrices = _projection_matrices.data_ptr<float>();
  const int* feature_size = _feature_size.data_ptr<int>();
  const int* image_size = _image_size.data_ptr<int>();
  const float* roi_range = _roi_range.data_ptr<float>();
  float* depth_params_grad = _depth_params_grad.data_ptr<float>();
  float* image_feats_grad = _image_feats_grad.data_ptr<float>();
  
  int grid_x, grid_y, grid_z;
  if (_grid_size.is_cuda()) {
    int grid_size_host[3];
    const int* grid_size = _grid_size.data_ptr<int>();
    cudaMemcpy(grid_size_host, grid_size, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    grid_x = grid_size_host[0];
    grid_y = grid_size_host[1];
    grid_z = grid_size_host[2];
  } else {
    const int* grid_size = _grid_size.data_ptr<int>();
    grid_x = grid_size[0];
    grid_y = grid_size[1];
    grid_z = grid_size[2];
  }
  
  int num_extrinsic_params = 12;
  int num_intrinsic_params = 12;
  
  flash_bevpool_grad(
    total_threads,
    num_channels,
    num_cameras,
    batch_size,
    num_extrinsic_params,
    num_intrinsic_params,
    depth_distribution,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_bevpool_forward", &flash_bevpool_forward, "flash_bevpool_forward");
  m.def("flash_bevpool_backward", &flash_bevpool_backward, "flash_bevpool_backward");
}


