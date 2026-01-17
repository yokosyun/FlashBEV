# Copyright (c) Shunsuke Yokokawa. All rights reserved.

"""
Utilities for calculating intermediate tensor memory usage for FlashBEV.

This module provides theoretical memory calculations derived from first
principles, making it easy for reviewers and users to verify and understand
the memory footprint of different view transform methods.
"""

from typing import Dict


def _calculate_view_transform_memory(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    feature_h: int,
    feature_w: int,
    bytes_per_element: int = 4,
    store_intermediate_tensors: bool = True,
) -> Dict[str, float]:
    """
    Shared memory calculation for view transform methods.

    Calculates an upper bound estimate of peak memory by summing all tensors
    as if they exist simultaneously. This is NOT HBM memory traffic (read/write
    counts), but rather the maximum memory footprint at any point.

    The key difference between FlashBEV and Dense PyTorch is whether
    intermediate tensors are stored (Dense PyTorch) or computed on-the-fly
    (FlashBEV).

    Note: Actual peak memory may be lower due to memory reuse/deallocation,
    but this provides a conservative upper bound.

    Args:
        feature_h: Feature map height
        feature_w: Feature map width
        bytes_per_element: Bytes per float32 element (default 4)
        store_intermediate_tensors: If True, include intermediate tensor
            memory (Dense PyTorch)
    """
    B = batch_size
    X = grid_x
    Y = grid_y
    Z = grid_z
    N = num_cameras
    C = num_channels
    H = feature_h
    W = feature_w
    
    memory = {}
    
    # Input tensors (same for both methods)
    # features_pv: [B*N, C, H, W] - perspective view features
    # depths: [B*N, H, W, 2] - depth parameters
    # projection_matrices: [B, N, 3, 4] - projection matrices
    memory["input_features"] = B * N * C * H * W * bytes_per_element
    memory["input_depths"] = B * N * 2 * H * W * bytes_per_element
    memory["projection_matrices"] = B * N * 3 * 4 * bytes_per_element
    
    if store_intermediate_tensors:
        # Dense PyTorch intermediate tensors
        
        # image_uvd: [B*N, X*Y*Z, 3] - projected coordinates (u, v, d)
        # Note: image_u, image_v, image_d, feat_u, feat_v are views/computed
        # on-the-fly, not separate tensors
        memory["image_uvd"] = B * N * X * Y * Z * 3 * bytes_per_element
        
        # FOV masks: [B*N, X*Y*Z] - boolean (stored as float in PyTorch)
        memory["fov_masks"] = B * N * X * Y * Z * bytes_per_element
        
        # Normalized coordinates: [B*N, X*Y*Z, 2]
        memory["coords_uv"] = B * N * X * Y * Z * 2 * bytes_per_element
        
        # Grid sample outputs
        # features_3d: [B*N, C, X*Y*Z] - interpolated features
        memory["features_3d"] = B * N * C * X * Y * Z * bytes_per_element
        
        # depths_3d: [B*N, 2, X*Y*Z] - interpolated depths
        # Note: depth_mu, depth_sigma are views/slices, not separate tensors
        memory["depths_3d"] = B * N * 2 * X * Y * Z * bytes_per_element
        
        # Z-score: [B*N, X*Y*Z]
        memory["z_score"] = B * N * X * Y * Z * bytes_per_element
        
        # Depth probability: [B*N, X*Y*Z]
        memory["depth_prob_3d"] = B * N * X * Y * Z * bytes_per_element
        
        # Valid mask: [B*N, X*Y*Z]
        memory["valid_mask"] = B * N * X * Y * Z * bytes_per_element
        
        # Weighted features: [B*N, C, X*Y*Z]
        # (in-place modification, but counted separately)
        memory["features_3d_weighted"] = (
            B * N * C * X * Y * Z * bytes_per_element
        )
        
        # Valid counts: [B, Y, X]
        memory["valid_counts"] = B * Y * X * bytes_per_element
    else:
        # FlashBEV: No intermediate tensors stored
        # Only temporary values computed per channel on-the-fly
        memory["intermediate_tensors"] = 0.0
    
    # Output tensor: [B, Y, X, C] or [B, C, Y, X]
    memory["output_bev"] = B * Y * X * C * bytes_per_element
    
    # Calculate totals
    if store_intermediate_tensors:
        intermediate_memory = (
            memory.get("image_uvd", 0) +
            memory.get("fov_masks", 0) +
            memory.get("coords_uv", 0) +
            memory.get("features_3d", 0) +
            memory.get("depths_3d", 0) +
            memory.get("z_score", 0) +
            memory.get("depth_prob_3d", 0) +
            memory.get("valid_mask", 0) +
            memory.get("features_3d_weighted", 0) +
            memory.get("valid_counts", 0)
        )
    else:
        intermediate_memory = 0.0
    
    input_memory = (
        memory["input_features"] +
        memory["input_depths"] +
        memory["projection_matrices"]
    )
    
    memory["intermediate_total"] = intermediate_memory
    memory["input_total"] = input_memory
    memory["output_total"] = memory["output_bev"]
    
    # Total tensor size: sum of all tensors (compiler-independent)
    memory["total_tensor_size"] = (
        input_memory + intermediate_memory + memory["output_bev"]
    )
    memory["total_tensor_size_mb"] = memory["total_tensor_size"] / (1024 * 1024)
    
    # Max tensor size: largest single tensor (compiler-independent)
    all_tensor_sizes = [
        memory["input_features"],
        memory["input_depths"],
        memory["projection_matrices"],
        memory["output_bev"],
    ]
    if store_intermediate_tensors:
        all_tensor_sizes.extend([
            memory.get("image_uvd", 0),
            memory.get("fov_masks", 0),
            memory.get("coords_uv", 0),
            memory.get("features_3d", 0),
            memory.get("depths_3d", 0),
            memory.get("z_score", 0),
            memory.get("depth_prob_3d", 0),
            memory.get("valid_mask", 0),
            memory.get("features_3d_weighted", 0),
            memory.get("valid_counts", 0),
        ])
    memory["max_tensor_size"] = max(all_tensor_sizes)
    memory["max_tensor_size_mb"] = memory["max_tensor_size"] / (1024 * 1024)
    
    # For backward compatibility
    memory["peak_memory"] = memory["total_tensor_size"]
    memory["peak_memory_mb"] = memory["total_tensor_size_mb"]
    
    return memory


def calculate_flashbevpool_memory(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    feature_h: int,
    feature_w: int,
    bytes_per_element: int = 4,
) -> Dict[str, float]:
    """
    Calculate theoretical memory usage for flash_bevpool_kernel.

    FlashBEV computes everything on-the-fly per channel, so no intermediate
    tensors are stored. Only input and output tensors.
    """
    return _calculate_view_transform_memory(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        feature_h=feature_h,
        feature_w=feature_w,
        bytes_per_element=bytes_per_element,
        store_intermediate_tensors=False,
    )


def calculate_dense_pytorch_sampling_vt_memory(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    feature_h: int,
    feature_w: int,
    bytes_per_element: int = 4,
) -> Dict[str, float]:
    """
    Calculate theoretical memory usage for Dense PyTorch Sampling-VT.

    Dense PyTorch stores all intermediate tensors, leading to higher peak
    memory.
    """
    return _calculate_view_transform_memory(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        feature_h=feature_h,
        feature_w=feature_w,
        bytes_per_element=bytes_per_element,
        store_intermediate_tensors=True,
    )


if __name__ == "__main__":
    batch_size = 1
    grid_x = 400
    grid_y = 400
    grid_z = 12
    num_cameras = 6
    num_channels = 128
    feature_h = 16
    feature_w = 44
    
    print("Memory Analysis Comparison")
    print("=" * 80)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Grid: {grid_x} x {grid_y} x {grid_z}")
    print(f"  Cameras: {num_cameras}")
    print(f"  Channels: {num_channels}")
    print(f"  Feature size: {feature_h} x {feature_w}")
    print()
    
    memory_flashbev = calculate_flashbevpool_memory(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        feature_h=feature_h,
        feature_w=feature_w,
    )
    
    memory_dense = calculate_dense_pytorch_sampling_vt_memory(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        feature_h=feature_h,
        feature_w=feature_w,
    )
    
    print("=" * 145)
    msg = "Side-by-Side Memory Comparison "
    msg += "(Dense PyTorch Sampling-VT = Baseline)"
    print(msg)
    print("=" * 145)
    print()
    symbols = ("Symbols: B=batch_size, X=grid_x, Y=grid_y, Z=grid_z, "
               "N=num_cameras, C=num_channels")
    print(symbols)
    print()

    header = (f"{'Tensor':<35s} | {'Shape':<35s} | "
              f"{'Dense PyTorch':>18s} | {'FlashBEV':>18s} | "
              f"{'Savings':>18s}")
    subheader = (f"{'':<35s} | {'':<35s} | {'(MB)':>18s} | "
                 f"{'(MB)':>18s} | {'(MB)':>18s}")
    separator = "─" * 145
    row_separator = "-" * 145
    
    print(header)
    print(subheader)
    print(separator)
    
    all_tensors = [
        ("Input Features", "input_features", "B*N*C*H*W"),
        ("Input Depths", "input_depths", "B*N*2*H*W"),
        ("Projection Matrices", "projection_matrices", "B*N*3*4"),
        ("Image UVD", "image_uvd", "B*N*X*Y*Z*3"),
        ("FOV Masks", "fov_masks", "B*N*X*Y*Z"),
        ("Coords UV", "coords_uv", "B*N*X*Y*Z*2"),
        ("Features 3D", "features_3d", "B*N*C*X*Y*Z"),
        ("Depths 3D", "depths_3d", "B*N*2*X*Y*Z"),
        ("Z-Score", "z_score", "B*N*X*Y*Z"),
        ("Depth Prob 3D", "depth_prob_3d", "B*N*X*Y*Z"),
        ("Valid Mask", "valid_mask", "B*N*X*Y*Z"),
        ("Features 3D Weighted", "features_3d_weighted", "B*N*C*X*Y*Z"),
        ("Valid Counts", "valid_counts", "B*Y*X"),
        ("Output BEV", "output_bev", "B*Y*X*C"),
    ]
    
    total_dense = 0.0
    total_flashbev = 0.0
    
    for tensor_name, key, shape_str in all_tensors:
        dense_value = 0.0
        flashbev_value = 0.0
        
        if key in memory_dense:
            dense_value = memory_dense[key] / (1024 * 1024)
            total_dense += memory_dense[key]
        
        if key in memory_flashbev:
            flashbev_value = memory_flashbev[key] / (1024 * 1024)
            total_flashbev += memory_flashbev[key]
        
        if dense_value > 0 and flashbev_value == 0:
            savings = dense_value
            savings_str = f"{savings:>16.2f} MB"
        elif dense_value > 0 and flashbev_value > 0:
            savings = dense_value - flashbev_value
            if abs(savings) < 0.01:
                savings_str = f"{'0.00':>16} MB"
            else:
                savings_str = f"{savings:>16.2f} MB"
        else:
            savings_str = f"{'—':>17}"
        
        dense_str = f"{dense_value:>16.2f} MB" if dense_value > 0 else f"{'—':>17}"
        flashbev_str = f"{flashbev_value:>16.2f} MB" if flashbev_value > 0 else f"{'—':>17}"
        
        print(f"   {tensor_name:<33s} | {shape_str:<33s} | "
              f"{dense_str} | {flashbev_str} | {savings_str}")

    print(separator)
    total_savings = (
        (memory_dense['total_tensor_size'] - memory_flashbev['total_tensor_size']) /
        (1024 * 1024)
    )
    max_tensor_savings = (
        (memory_dense['max_tensor_size'] - memory_flashbev['max_tensor_size']) /
        (1024 * 1024)
    )
    print(f"{'TOTAL TENSOR SIZE':<35s} | {'Sum of all tensors':<33s} | "
          f"{memory_dense['total_tensor_size_mb']:>16.2f} MB | "
          f"{memory_flashbev['total_tensor_size_mb']:>16.2f} MB | "
          f"{total_savings:>16.2f} MB")
    print(f"{'MAX TENSOR SIZE':<35s} | {'Largest single tensor':<33s} | "
          f"{memory_dense['max_tensor_size_mb']:>16.2f} MB | "
          f"{memory_flashbev['max_tensor_size_mb']:>16.2f} MB | "
          f"{max_tensor_savings:>16.2f} MB")
    print()
    
    total_reduction = (
        (1 - memory_flashbev['total_tensor_size'] /
         memory_dense['total_tensor_size']) * 100
    )
    max_reduction = (
        (1 - memory_flashbev['max_tensor_size'] /
         memory_dense['max_tensor_size']) * 100
    )
    print(f"   TOTAL TENSOR SIZE Reduction: {total_reduction:.1f}%")
    print(f"   MAX TENSOR SIZE Reduction: {max_reduction:.1f}%")
    print()

