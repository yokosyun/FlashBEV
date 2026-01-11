"""
Utilities for calculating FLOPs (Floating Point Operations) for FlashBEV kernels.

This module provides theoretical FLOP calculations derived from first principles,
making it easy for reviewers and users to verify and understand the computational
complexity of different view transform methods.
"""

from typing import Dict, Optional, Tuple




def _calculate_view_transform_flops(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    depth_distribution: str,
    multiply_channel_independent_by_c: bool,
    average_valid_voxel_ratio: float = 1.0,
) -> Dict[str, float]:
    """
    Shared FLOP calculation for view transform methods.
    
    The only difference between FlashBEV and Dense PyTorch is whether channel-independent
    operations are multiplied by C (FlashBEV recomputes them per channel) or not.
    
    Args:
        multiply_channel_independent_by_c: If True, multiply channel-independent ops by C (FlashBEV)
    """
    B = batch_size
    X = grid_x
    Y = grid_y
    Z = grid_z
    N = num_cameras
    C = num_channels
    
    # FLOPs per operation (channel-independent base)
    projection_flops = 12
    normalization_flops = 3
    coordinate_computation_flops = 4
    bilinear_weights_flops = 4
    depth_interpolation_flops = 14
    depth_weight_calc_flops = 5 if depth_distribution == "laplace" else 6
    feature_interp_flops = 7
    
    base_per_voxel_camera = (
        projection_flops
        + normalization_flops
        + coordinate_computation_flops
        + bilinear_weights_flops
        + depth_interpolation_flops
        + depth_weight_calc_flops
    )
    
    # Channel-independent operations
    # FlashBEV: B × N × X × Y × Z × base_per_voxel_camera × C (recomputed per channel)
    # Dense PyTorch: B × N × X × Y × Z × base_per_voxel_camera (computed once)
    channel_independent_mult = C if multiply_channel_independent_by_c else 1
    channel_independent_base_flops = B * N * X * Y * Z * base_per_voxel_camera * channel_independent_mult
    
    # Channel-dependent operations
    grid_sample_features_flops = B * N * X * Y * Z * C * feature_interp_flops
    # grid_sample_depths is only for Dense PyTorch (FlashBEV does depth interpolation inline)
    grid_sample_depths_flops = B * N * X * Y * Z * 2 * feature_interp_flops if not multiply_channel_independent_by_c else 0
    total_voxel_camera_combos = B * N * X * Y * Z
    total_valid_voxels = int(total_voxel_camera_combos * average_valid_voxel_ratio)
    feature_weighting_flops = total_valid_voxels * C * 1
    masking_flops = total_valid_voxels * 1 if not multiply_channel_independent_by_c else total_valid_voxels * C * 1
    num_elements_per_output = N * Z
    sum_over_cameras_z_flops = B * X * Y * C * (num_elements_per_output - 1)
    final_division_flops = B * X * Y * C * 1
    
    channel_dependent_flops = (
        grid_sample_features_flops
        + grid_sample_depths_flops
        + feature_weighting_flops
        + masking_flops
        + sum_over_cameras_z_flops
        + final_division_flops
    )
    
    # Build detailed breakdown (using unified key names)
    channel_independent_mult = C if multiply_channel_independent_by_c else 1
    flops = {
        "projection": B * N * X * Y * Z * channel_independent_mult * projection_flops,
        "normalization": B * N * X * Y * Z * channel_independent_mult * normalization_flops,
        "coordinate_computation": B * N * X * Y * Z * channel_independent_mult * coordinate_computation_flops,
        "bilinear_weights": B * N * X * Y * Z * channel_independent_mult * bilinear_weights_flops,
        "depth_interpolation": B * N * X * Y * Z * channel_independent_mult * depth_interpolation_flops,
        "depth_weight_calc": B * N * X * Y * Z * channel_independent_mult * depth_weight_calc_flops,
        "feature_interpolation": grid_sample_features_flops,
        "feature_weighting": feature_weighting_flops,
        "bev_accumulation": sum_over_cameras_z_flops,
        "valid_count": masking_flops,
        "final_division": final_division_flops,
    }
    
    if not multiply_channel_independent_by_c:
        flops["grid_sample_depths"] = grid_sample_depths_flops
    
    if multiply_channel_independent_by_c:
        flops["weighted_accumulation"] = feature_weighting_flops + sum_over_cameras_z_flops
        flops["recompute_flops"] = channel_independent_base_flops + masking_flops
        flops["non_recompute_flops"] = grid_sample_features_flops + feature_weighting_flops + sum_over_cameras_z_flops + final_division_flops
    else:
        flops["non_recompute_flops"] = channel_independent_base_flops + channel_dependent_flops
    
    flops["total"] = channel_independent_base_flops + channel_dependent_flops
    flops["total_gflops"] = flops["total"] / 1e9
    
    return flops


def calculate_flashbevpool_flops(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    depth_distribution: str = "laplace",
    average_valid_voxels_per_bev: Optional[float] = None,
    average_valid_cameras_per_voxel: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate theoretical FLOPs for flash_bevpool_kernel.
    
    Equation-based computation: FlashBEV multiplies channel-independent operations by C.
    """
    if average_valid_voxels_per_bev is None:
        average_valid_voxels_per_bev = grid_z
    average_valid_voxel_ratio = average_valid_voxels_per_bev / grid_z if grid_z > 0 else 1.0
    
    return _calculate_view_transform_flops(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        depth_distribution=depth_distribution,
        multiply_channel_independent_by_c=True,
        average_valid_voxel_ratio=average_valid_voxel_ratio,
    )


def calculate_dense_pytorch_sampling_vt_flops(
    batch_size: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    num_channels: int,
    depth_distribution: str = "laplace",
    average_valid_voxel_ratio: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate theoretical FLOPs for Dense PyTorch Sampling-VT.
    
    Equation-based computation: Channel-independent operations computed once (NOT × C).
    """
    return _calculate_view_transform_flops(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        depth_distribution=depth_distribution,
        multiply_channel_independent_by_c=False,
        average_valid_voxel_ratio=average_valid_voxel_ratio,
    )


if __name__ == "__main__":
    batch_size = 1
    grid_x = 400
    grid_y = 400
    grid_z = 12
    num_cameras = 6
    num_channels = 128
    
    print("FLOP Analysis Comparison")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Grid: {grid_x} x {grid_y} x {grid_z}")
    print(f"  Cameras: {num_cameras}")
    print(f"  Channels: {num_channels}")
    print()
    
    flops_flashbev = calculate_flashbevpool_flops(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        depth_distribution="laplace",
    )
    
    flops_dense = calculate_dense_pytorch_sampling_vt_flops(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        depth_distribution="laplace",
        average_valid_voxel_ratio=1.0,
    )
    
    
    print("=" * 145)
    print("Side-by-Side FLOPs Comparison (Dense PyTorch Sampling-VT = Baseline)")
    print("=" * 145)
    print()
    print("Symbols: B=batch_size, X=grid_x, Y=grid_y, Z=grid_z, N=num_cameras, C=num_channels")
    print()
    
    header = f"{'Operation':<35s} | {'Formula':<35s} | {'Recomp.':<8s} | {'Dense PyTorch':>18s} | {'FlashBEV':>18s} | {'Overhead':>18s}"
    subheader = f"{'':<35s} | {'':<35s} | {'×C':<8s} | {'(Baseline)':>18s} | {'(Total)':>18s} | {'(Wasted)':>18s}"
    separator = "─" * 145
    row_separator = "-" * 145
    
    print(header)
    print(subheader)
    print(separator)
    
    # Keys are now unified, so we can use them directly
    
    all_operations = [
        ("Projection", "projection", True, "B*N*X*Y*Z*12"),
        ("Normalization", "normalization", True, "B*N*X*Y*Z*3"),
        ("Coordinate Computation", "coordinate_computation", True, "B*N*X*Y*Z*4"),
        ("Bilinear Weights", "bilinear_weights", True, "B*N*X*Y*Z*4"),
        ("Depth Interpolation", "depth_interpolation", True, "B*N*X*Y*Z*14"),
        ("Grid Sample Depths", "grid_sample_depths", False, "B*N*X*Y*Z*2*7"),
        ("Depth Weight Calc", "depth_weight_calc", True, "B*N*X*Y*Z*7"),
        ("Feature Interpolation", "feature_interpolation", False, "B*N*X*Y*Z*C*7"),
        ("Depth Likelihood Weighting", "feature_weighting", False, "B*N*X*Y*Z*C*1"),
        ("BEV Accumulation", "bev_accumulation", False, "B*X*Y*C*(N*Z-1)"),
        ("Valid Count", "valid_count", True, "B*N*X*Y*Z*1"),
        ("Final Division", "final_division", False, "B*X*Y*C*1"),
    ]
    
    recomputed_overhead = 0.0
    total_dense = 0.0
    total_flashbev = 0.0
    recomputed_ops_count = 0
    
    for op_name, canonical_key, is_recomputed, formula in all_operations:
        dense_value = 0.0
        flashbev_value = 0.0
        
        if canonical_key in flops_dense:
            dense_value = flops_dense[canonical_key] / 1e9
            total_dense += flops_dense[canonical_key]
        
        if canonical_key in flops_flashbev:
            flashbev_value = flops_flashbev[canonical_key] / 1e9
            total_flashbev += flops_flashbev[canonical_key]
        
        if is_recomputed:
            recomputed_ops_count += 1
            overhead = flashbev_value - dense_value
            recomputed_overhead += overhead * 1e9
            diff_str = f"{overhead:>16.2f} G"
        elif dense_value > 0 and flashbev_value > 0:
            diff = flashbev_value - dense_value
            if abs(diff) < 0.01:
                diff_str = f"{'0.00':>16} G"
            else:
                diff_str = f"{diff:>16.2f} G"
        else:
            if dense_value > 0:
                diff_str = f"{'—':>17}"
            elif flashbev_value > 0:
                diff_str = f"{'—':>17}"
            else:
                diff_str = f"{'—':>17}"
        
        dense_str = f"{dense_value:>16.2f} G" if dense_value > 0 else f"{'—':>17}"
        flashbev_str = f"{flashbev_value:>16.2f} G" if flashbev_value > 0 else f"{'—':>17}"
        
        # Show formula (Recomp. ×C column indicates if recomputed)
        formula_str = formula
        
        # Mark recomputed operations with checkbox
        recomp_marker = "✓" if is_recomputed else " "
        
        print(f"   {op_name:<33s} | {formula_str:<33s} | {recomp_marker:<8s} | {dense_str} | {flashbev_str} | {diff_str}")
        if is_recomputed and recomputed_ops_count < 6:
            print(row_separator)
    
    print(separator)
    total_diff = flops_flashbev['total'] / 1e9 - flops_dense['total'] / 1e9
    print(f"{'TOTAL':<35s} | {'Sum of above':<33s} | {'':<8s} | {flops_dense['total_gflops']:>16.2f} G | {flops_flashbev['total_gflops']:>16.2f} G | {total_diff:>16.2f} G")
    print()
    
    print(f"   Total Ratio: {flops_flashbev['total_gflops'] / flops_dense['total_gflops']:.2f}x")
    print()
    
