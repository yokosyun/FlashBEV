# Copyright (c) Shunsuke Yokokawa. All rights reserved.

import torch

try:
    from . import flashbevpool_ext
except ImportError:
    import flashbevpool_ext


class FlashBEVPool(torch.autograd.Function):
    r"""FlashBEVPool implementation with Gaussian/Laplace depth distribution and fused kernel.
    
    This implementation uses a fused CUDA kernel that performs:
    - 3D to 2D projection with 4x4 matrices
    - Gaussian or Laplace depth distribution (mean, sigma)
    - Bilinear interpolation
    - Reduction over cameras (N) and height (Z)
    """
    
    @staticmethod
    def forward(ctx, depth_params, image_feats, projection_matrices,
                feature_size, image_size, roi_range, grid_size, depth_distribution=0, use_shared_memory=True, optimize_z_precompute=True, use_warp_kernel=False, use_vectorized_load=False, epsilon=1e-6, depth_weight_threshold=1e-6):
        """
        Forward pass for FlashBEVPool fused kernel.
        
        Args:
            depth_params: Depth parameters [B,N,H,W,2] - [mean, sigma] or [mean, b]
            image_feats: Image features [B,N,H,W,C]
            projection_matrices: Camera projection matrices [B,N,4,4]
            feature_size: Feature dimensions [B,N,2]
            image_size: Image dimensions [2] [height, width]
            roi_range: ROI bounds [6] [min_x,max_x,min_y,max_y,min_z,max_z]
            grid_size: Grid dimensions [3] [grid_x,grid_y,grid_z]
            depth_distribution: 0=gaussian, 1=laplace
            use_shared_memory: Whether to use shared memory for projection matrices
            optimize_z_precompute: Whether to optimize z-independent projection computation
            use_warp_kernel: Whether to use warp-optimized kernel (channels loop inside)
            use_vectorized_load: Whether to use vectorized (float4) loads (requires use_warp_kernel=True)
            epsilon: Small value for numerical stability
            depth_weight_threshold: Threshold for depth weight filtering
            
        Returns:
            BEV features [B,X,Y,C]
        """
  
        depth_params = depth_params.contiguous().float()
        image_feats = image_feats.contiguous().float()
        projection_matrices = projection_matrices.contiguous().float()
        feature_size = feature_size.contiguous().int()
        image_size = image_size.contiguous().int()
        roi_range = roi_range.contiguous().float()
        grid_size = grid_size.contiguous().int()
        
        batch_size = depth_params.size(0)
        num_channels = image_feats.size(-1)
        
        grid_x = grid_size[0].item()
        grid_y = grid_size[1].item()
        grid_z = grid_size[2].item()
        
        voxel_size_x = (roi_range[1] - roi_range[0]) / grid_x
        voxel_size_y = (roi_range[3] - roi_range[2]) / grid_y
        voxel_size_z = (roi_range[5] - roi_range[4]) / grid_z
        voxel_x_min = roi_range[0].item()
        voxel_y_min = roi_range[2].item()
        voxel_z_min = roi_range[4].item()
        
        out = image_feats.new_zeros((batch_size, grid_size[0], grid_size[1], num_channels))
        
        total_threads = out.numel()
        flashbevpool_ext.flash_bevpool_forward(
            depth_params,
            image_feats,
            projection_matrices,
            feature_size,
            image_size,
            grid_size,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            voxel_x_min,
            voxel_y_min,
            voxel_z_min,
            out,
            total_threads,
            depth_distribution,
            use_shared_memory,
            optimize_z_precompute,
            use_warp_kernel,
            use_vectorized_load,
            epsilon,
            depth_weight_threshold
        )
        
        ctx.save_for_backward(
            depth_params, image_feats, projection_matrices,
            feature_size, image_size, roi_range, grid_size)
        ctx.total_threads = total_threads
        ctx.depth_distribution = depth_distribution
        ctx.optimize_z_precompute = optimize_z_precompute
        ctx.epsilon = epsilon
        ctx.depth_weight_threshold = depth_weight_threshold
        
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        """
        Backward pass for FlashBEVPool fused kernel.
        """
        (depth_params, image_feats, projection_matrices, feature_size, image_size, roi_range,
         grid_size) = ctx.saved_tensors
        
        depth_params_grad = torch.zeros_like(depth_params)
        image_feats_grad = torch.zeros_like(image_feats)
        
        flashbevpool_ext.flash_bevpool_backward(
            out_grad,
            depth_params,
            image_feats,
            projection_matrices,
            feature_size,
            image_size,
            roi_range,
            grid_size,
            depth_params_grad,
            image_feats_grad,
            ctx.total_threads,
            ctx.depth_distribution,
            ctx.epsilon,
            ctx.depth_weight_threshold
        )
        
        return (depth_params_grad, image_feats_grad, None, None, None, None,
                None, None, None, None, None, None, None, None)


def flash_bevpool(image_feats, depth_params, projection_matrices,
                     feature_size, image_size, roi_range, grid_size, depth_distribution=0, use_shared_memory=True, optimize_z_precompute=True, use_warp_kernel=False, use_vectorized_load=False, epsilon=1e-6, depth_weight_threshold=1e-6):
    """
    FlashBEVPool fused function wrapper.
    
    Args:
        depth_params: Depth parameters [B,N,H,W,2] - [mean, sigma] or [mean, b]
        image_feats: Image features [B,N,H,W,C]
        projection_matrices: Camera projection matrices [B,N,4,4]
        feature_size: Feature dimensions [B,N,2]
        image_size: Image dimensions [2] [height, width]
        roi_range: ROI bounds [6] [min_x,max_x,min_y,max_y,min_z,max_z]
        grid_size: Grid dimensions [3] [grid_x,grid_y,grid_z]
        depth_distribution: 0=gaussian, 1=laplace
        use_shared_memory: Whether to use shared memory for projection matrices
        optimize_z_precompute: Whether to optimize z-independent projection computation
        use_warp_kernel: Whether to use warp-optimized kernel (channels loop inside)
        use_vectorized_load: Whether to use vectorized (float4) loads (requires use_warp_kernel=True)
        epsilon: Small value for numerical stability
        depth_weight_threshold: Threshold for depth weight filtering
        
    Returns:
        BEV features [B,X,Y,C]
    """
    return FlashBEVPool.apply(
            depth_params, 
            image_feats, 
            projection_matrices,
            feature_size,
            image_size,
            roi_range,
            grid_size,
            depth_distribution,
            use_shared_memory,
            optimize_z_precompute,
            use_warp_kernel,
            use_vectorized_load,
            epsilon,
            depth_weight_threshold
        )
