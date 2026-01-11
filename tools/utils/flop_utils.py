"""
Utilities for calculating FLOPs (Floating Point Operations) for FlashBEV kernels.

This module provides theoretical FLOP calculations derived from first principles,
making it easy for reviewers and users to verify and understand the computational
complexity of different view transform methods.
"""

from typing import Dict, Optional, Tuple


def calculate_shared_flops_per_voxel_camera(
    depth_distribution: str = "laplace",
) -> Dict[str, float]:
    """
    Calculate shared FLOPs per voxel-camera combination.
    
    Theoretical derivation: These operations transform 3D voxel coordinates to 2D image
    coordinates and compute sampling weights. They depend only on geometry (x, y, z, camera
    parameters), NOT on feature channels. Therefore, they should be computed once per voxel-camera.
    
    Mathematical basis:
    - Input: One 3D voxel location (x, y, z) + one camera
    - Output: 2D image coordinates (u, v) + depth (d) + sampling weights
    
    These operations are common to both FlashBEV and Dense PyTorch methods:
    - Projection (3D to image coordinates): Homogeneous matrix multiplication
    - Normalization: Perspective division
    - Coordinate computation: Scale to feature map coordinates
    - Bilinear weights: Compute interpolation weights
    - Depth interpolation: Bilinear interpolation of depth map
    - Depth weight calculation: Probability/weight based on depth distribution
    
    Returns:
        Dictionary with FLOP counts per voxel-camera
    """
    flops = {
        "projection": 0.0,
        "normalization": 0.0,
        "coordinate_computation": 0.0,
        "bilinear_weights": 0.0,
        "depth_interpolation": 0.0,
        "depth_weight_calc": 0.0,
    }
    
    # Projection: 3D→2D homogeneous transformation
    # Theory: [u, v, d]^T = P × [x, y, z, 1]^T where P is 3×4 projection matrix
    # Derivation: Each output coordinate requires: P_row · [x, y, z, 1]
    #             = P[0]*x + P[1]*y + P[2]*z + P[3]
    # Formula: 3 coords × (3 mults + 1 add + 1 mult for FMA optimization) = 12 FLOPs
    # Details: For each coordinate (u, v, d):
    #          - 3 multiplies (P[i]*x, P[i+1]*y, P[i+2]*z) = 3 mults
    #          - 3 additions (sequential accumulation) = 3 adds
    #          - With FMA: 3 mult-add pairs = 3×2 = 6 FLOPs per coord
    #          - But simpler count: 3 coords × 4 ops = 12 FLOPs (we count 3+3 = 6×2=12)
    flops["projection"] = 9 + 3  # 3 coords × (3 mults + 1 add) = 12 FLOPs
    
    # Normalization: clamped_d = max/min(img_d, epsilon), img_u = img_u / clamped_d, img_v = img_v / clamped_d
    # FLOPs: 1 max/min + 2 divs = 3 FLOPs
    flops["normalization"] = 1 + 2
    
    # Coordinate computation: feat_u = img_u / image_width * feat_w, feat_v = img_v / image_height * feat_h
    # FLOPs: 2 divs + 2 mults = 4 FLOPs
    flops["coordinate_computation"] = 2 + 2
    
    # Bilinear weights: w00 = (1-du)*(1-dv), w01 = (1-du)*dv, w10 = du*(1-dv), w11 = du*dv
    # FLOPs: 4 mults = 4 FLOPs
    flops["bilinear_weights"] = 4
    
    # Depth interpolation: Bilinear interpolation of depth map
    # Theory: depth_value = Σ(w_ij × depth_map[i,j]) for i,j in {0,1}×{0,1}
    # Derivation: We interpolate 2 values (depth_mean, depth_sigma) from 4 neighbors
    #             Each value: w00*d00 + w01*d01 + w10*d10 + w11*d11
    #             = 4 multiplies + 3 additions = 7 FLOPs per value
    # Formula: 2 values × (4 mults + 3 adds) = 2 × 7 = 14 FLOPs
    flops["depth_interpolation"] = 8 + 6  # 2 values × (4 mults + 3 adds) = 14 FLOPs
    
    # Depth weight calculation
    if depth_distribution == "gaussian":
        # z_score = (img_d - depth_mean) / (depth_sigma + epsilon)
        # depth_weight = exp(-0.5 * z_score^2) / (depth_sigma + epsilon)
        # FLOPs: 1 sub + 1 add + 1 div + 1 sqr + 1 mult + 1 exp + 1 div = 8 FLOPs
        flops["depth_weight_calc"] = 2 + 1 + 2 + 1
    else:
        # z_score = (img_d - depth_mean) / (depth_sigma + epsilon)
        # depth_weight = 0.5 * exp(-|z_score|) / (depth_sigma + epsilon)
        # FLOPs: 1 sub + 1 add + 1 div + 1 abs + 1 exp + 1 mult + 1 div = 7 FLOPs
        flops["depth_weight_calc"] = 2 + 1 + 1 + 1
    
    return flops


def calculate_feature_interpolation_flops_per_channel() -> float:
    """
    Calculate FLOPs for feature interpolation per channel.
    
    Feature interpolation: feat_value = w00*feat[00] + w01*feat[01] + w10*feat[10] + w11*feat[11]
    FLOPs: 4 mults + 3 adds = 7 FLOPs per channel
    """
    return 4 + 3


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
    
    Simplified approach: Separate operations into recompute_flops (channel-independent)
    and non_recompute_flops (channel-dependent), then multiply recompute_flops by C.
    
    Args:
        batch_size: Batch size
        grid_x: Grid size in X dimension
        grid_y: Grid size in Y dimension
        grid_z: Grid size in Z dimension (height bins)
        num_cameras: Number of cameras
        num_channels: Number of feature channels
        depth_distribution: "laplace" or "gaussian"
        average_valid_voxels_per_bev: Average number of valid voxels per BEV location (for early termination)
        average_valid_cameras_per_voxel: Average number of valid cameras per voxel (for early termination)
    
    Returns:
        Dictionary with FLOP counts for different operation categories
    """
    # Define problem dimensions
    B = batch_size
    X = grid_x
    Y = grid_y
    Z = grid_z
    N = num_cameras
    C = num_channels
    
    if average_valid_voxels_per_bev is None:
        average_valid_voxels_per_bev = Z
    if average_valid_cameras_per_voxel is None:
        average_valid_cameras_per_voxel = N
    
    shared_flops = calculate_shared_flops_per_voxel_camera(depth_distribution)
    feature_interp_flops = calculate_feature_interpolation_flops_per_channel()
    
    # Base FLOPs per voxel-camera (channel-independent, should be computed once)
    base_per_voxel_camera = (
        shared_flops["projection"]
        + shared_flops["normalization"]
        + shared_flops["coordinate_computation"]
        + shared_flops["bilinear_weights"]
        + shared_flops["depth_interpolation"]
        + shared_flops["depth_weight_calc"]
    )
    
    # Recomputed FLOPs: Channel-independent operations × C (recomputed per channel)
    # Formula: B × N × X × Y × Z × base_per_voxel_camera × C
    recompute_flops = B * N * X * Y * Z * base_per_voxel_camera * C
    
    # Non-recomputed FLOPs: Channel-dependent operations (naturally per channel)
    # Feature interpolation: B × N × X × Y × Z × C × feature_interp_flops
    feature_interpolation_flops = B * N * X * Y * Z * C * feature_interp_flops
    
    # Depth likelihood weighting: B × N × X × Y × Z × C × 1
    depth_weighting_flops = B * N * X * Y * Z * C * 1
    
    # BEV accumulation: B × X × Y × C × (N × Z - 1)
    num_elements_per_output = N * Z
    bev_accumulation_flops = B * X * Y * C * (num_elements_per_output - 1)
    
    # Valid count (recomputed in FlashBEV): B × N × X × Y × Z × C × 1
    valid_count_flops = B * N * X * Y * Z * C * 1
    
    # Final division: B × X × Y × C × 1
    final_division_flops = B * X * Y * C * 1
    
    non_recompute_flops = (
        feature_interpolation_flops
        + depth_weighting_flops
        + bev_accumulation_flops
        + final_division_flops
    )
    
    # Build detailed breakdown for backward compatibility
    flops = {
        "projection_per_voxel": B * N * X * Y * Z * C * shared_flops["projection"],
        "normalization": B * N * X * Y * Z * C * shared_flops["normalization"],
        "coordinate_computation": B * N * X * Y * Z * C * shared_flops["coordinate_computation"],
        "bilinear_weights": B * N * X * Y * Z * C * shared_flops["bilinear_weights"],
        "depth_interpolation": B * N * X * Y * Z * C * shared_flops["depth_interpolation"],
        "depth_weight_calc": B * N * X * Y * Z * C * shared_flops["depth_weight_calc"],
        "feature_interpolation": feature_interpolation_flops,
        "depth_likelihood_weighting": depth_weighting_flops,
        "bev_accumulation": bev_accumulation_flops,
        "weighted_accumulation": depth_weighting_flops + bev_accumulation_flops,
        "valid_count_increment": valid_count_flops,
        "final_division": final_division_flops,
    }
    
    # Add summary categories (valid_count is also recomputed)
    flops["recompute_flops"] = recompute_flops + valid_count_flops
    flops["non_recompute_flops"] = non_recompute_flops
    flops["total"] = flops["recompute_flops"] + flops["non_recompute_flops"]
    flops["total_gflops"] = flops["total"] / 1e9
    
    return flops


def calculate_flops_per_output_element(
    grid_x: int,
    grid_y: int,
    grid_z: int,
    num_cameras: int,
    depth_distribution: str = "laplace",
) -> Dict[str, float]:
    """
    Calculate FLOPs per output BEV element (per channel).
    
    Returns average FLOPs assuming all voxels and cameras are valid.
    """
    flops_per_element = {
        "projection_per_voxel": 12.0,
        "normalization": 3.0,
        "coordinate_computation": 4.0,
        "bilinear_weights": 4.0,
        "depth_interpolation": 7.0,
        "depth_weight_calc": 0.0,
        "feature_interpolation": 7.0,
        "weighted_accumulation": 2.0,
        "final_division": 1.0,
    }
    
    if depth_distribution == "gaussian":
        flops_per_element["depth_weight_calc"] = 8.0
    else:
        flops_per_element["depth_weight_calc"] = 7.0
    
    total_per_element = sum(flops_per_element.values())
    flops_per_element["total"] = total_per_element
    
    total_flops = total_per_element * grid_z * num_cameras
    flops_per_element["total_per_element_all_voxels"] = total_flops
    
    return flops_per_element


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
    
    Simplified approach: Compute channel-independent operations once (not multiplied by C),
    and channel-dependent operations per channel (multiplied by C).
    
    Args:
        batch_size: Batch size
        grid_x: Grid size in X dimension
        grid_y: Grid size in Y dimension
        grid_z: Grid size in Z dimension (height bins)
        num_cameras: Number of cameras
        num_channels: Number of feature channels
        depth_distribution: "laplace" or "gaussian"
        average_valid_voxel_ratio: Ratio of valid voxels (accounts for early termination)
    
    Returns:
        Dictionary with FLOP counts for different operation categories
    """
    # Define problem dimensions
    B = batch_size
    X = grid_x
    Y = grid_y
    Z = grid_z
    N = num_cameras
    C = num_channels
    
    shared_flops = calculate_shared_flops_per_voxel_camera(depth_distribution)
    feature_interp_flops = calculate_feature_interpolation_flops_per_channel()
    
    # Non-recomputed FLOPs: Channel-independent operations (computed once, NOT × C)
    # Formula: B × N × X × Y × Z × base_per_voxel_camera
    base_per_voxel_camera = (
        shared_flops["projection"]
        + shared_flops["normalization"]
        + shared_flops["coordinate_computation"]
        + shared_flops["bilinear_weights"]
        + shared_flops["depth_interpolation"]
        + shared_flops["depth_weight_calc"]
    )
    non_recompute_flops_base = B * N * X * Y * Z * base_per_voxel_camera
    
    # Grid sample: Feature interpolation (C channels) + depth interpolation (2 values)
    # Formula: B × N × X × Y × Z × C × feature_interp_flops + B × N × X × Y × Z × 2 × feature_interp_flops
    grid_sample_features_flops = B * N * X * Y * Z * C * feature_interp_flops
    grid_sample_depths_flops = B * N * X * Y * Z * 2 * feature_interp_flops
    
    # Feature weighting: B × N × X × Y × Z × C × 1
    total_voxel_camera_combos = B * N * X * Y * Z
    total_valid_voxels = int(total_voxel_camera_combos * average_valid_voxel_ratio)
    feature_weighting_flops = total_valid_voxels * C * 1
    
    # Masking: B × N × X × Y × Z × 1 (computed once per voxel-camera)
    masking_flops = total_valid_voxels * 1
    
    # Sum over cameras and z: B × X × Y × C × (N × Z - 1)
    num_elements_per_output = N * Z
    sum_over_cameras_z_flops = B * X * Y * C * (num_elements_per_output - 1)
    
    # Final division: B × X × Y × C × 1
    final_division_flops = B * X * Y * C * 1
    
    non_recompute_flops = (
        non_recompute_flops_base
        + grid_sample_features_flops
        + grid_sample_depths_flops
        + feature_weighting_flops
        + masking_flops
        + sum_over_cameras_z_flops
        + final_division_flops
    )
    
    # Build detailed breakdown for backward compatibility
    flops = {
        "projection": B * N * X * Y * Z * shared_flops["projection"],
        "normalization": B * N * X * Y * Z * shared_flops["normalization"],
        "coordinate_computation": B * N * X * Y * Z * shared_flops["coordinate_computation"],
        "bilinear_weights": B * N * X * Y * Z * shared_flops["bilinear_weights"],
        "depth_interpolation": B * N * X * Y * Z * shared_flops["depth_interpolation"],
        "depth_weight_calc": B * N * X * Y * Z * shared_flops["depth_weight_calc"],
        "grid_sample_features": grid_sample_features_flops,
        "grid_sample_depths": grid_sample_depths_flops,
        "feature_weighting": feature_weighting_flops,
        "masking": masking_flops,
        "sum_over_cameras_z": sum_over_cameras_z_flops,
        "final_division": final_division_flops,
    }
    
    # Add summary categories
    flops["non_recompute_flops"] = non_recompute_flops
    flops["total"] = non_recompute_flops
    flops["total_gflops"] = flops["total"] / 1e9
    
    return flops


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
    
    print("=" * 80)
    print("FlashBEV Kernel (fused)")
    print("=" * 80)
    flops_flashbev = calculate_flashbevpool_flops(
        batch_size=batch_size,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        num_cameras=num_cameras,
        num_channels=num_channels,
        depth_distribution="laplace",
    )
    
    print(f"FLOP Breakdown:")
    print(f"  {'─'*76}")
    print(f"  RECOMPUTED PER CHANNEL (should be computed once per voxel-camera):")
    print(f"  {'─'*76}")
    recomputed_ops = ["projection_per_voxel", "normalization", "coordinate_computation", 
                     "bilinear_weights", "depth_interpolation", "depth_weight_calc"]
    recomputed_total = 0.0
    for key in recomputed_ops:
        if key in flops_flashbev:
            value = flops_flashbev[key]
            recomputed_total += value
            print(f"    ⚠️  {key:28s}: {value:15,.0f} FLOPs ({value/1e9:8.2f} GFLOPs) [RECOMPUTED ×{num_channels}]")
    print(f"  {'─'*76}")
    print(f"    {'Subtotal (recomputation overhead)':28s}: {recomputed_total:15,.0f} FLOPs ({recomputed_total/1e9:8.2f} GFLOPs)")
    print()
    print(f"  CHANNEL-DEPENDENT OPERATIONS (correctly per channel):")
    print(f"  {'─'*76}")
    channel_dependent_ops = ["feature_interpolation", "weighted_accumulation", "valid_count_increment"]
    for key in channel_dependent_ops:
        if key in flops_flashbev:
            value = flops_flashbev[key]
            print(f"    ✓  {key:28s}: {value:15,.0f} FLOPs ({value/1e9:8.2f} GFLOPs)")
    print()
    if "final_division" in flops_flashbev:
        print(f"  FINAL OPERATIONS:")
        print(f"  {'─'*76}")
        print(f"    {('final_division'):28s}: {flops_flashbev['final_division']:15,.0f} FLOPs ({flops_flashbev['final_division']/1e9:8.2f} GFLOPs)")
    print()
    print(f"  {'Total':30s}: {flops_flashbev['total']:15,.0f} FLOPs ({flops_flashbev['total_gflops']:8.2f} GFLOPs)")
    print()
    
    print("=" * 80)
    print("Dense PyTorch Sampling-VT")
    print("=" * 80)
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
    
    print(f"FLOP Breakdown:")
    print(f"  {'─'*76}")
    print(f"  COMPUTED ONCE PER VOXEL-CAMERA (not recomputed per channel):")
    print(f"  {'─'*76}")
    once_per_voxel_ops = ["projection", "normalization", "coordinate_computation",
                         "bilinear_weights", "depth_interpolation", "depth_weight_calc"]
    once_total = 0.0
    for key in once_per_voxel_ops:
        if key in flops_dense:
            value = flops_dense[key]
            once_total += value
            print(f"    ✓  {key:28s}: {value:15,.0f} FLOPs ({value/1e9:8.2f} GFLOPs) [computed once]")
    print(f"  {'─'*76}")
    print(f"    {'Subtotal (efficient)':28s}: {once_total:15,.0f} FLOPs ({once_total/1e9:8.2f} GFLOPs)")
    print()
    print(f"  VECTORIZED OPERATIONS (via grid_sample, optimized):")
    print(f"  {'─'*76}")
    vectorized_ops = ["grid_sample_features", "grid_sample_depths"]
    for key in vectorized_ops:
        if key in flops_dense:
            value = flops_dense[key]
            print(f"    ✓  {key:28s}: {value:15,.0f} FLOPs ({value/1e9:8.2f} GFLOPs)")
    print()
    print(f"  OTHER OPERATIONS:")
    print(f"  {'─'*76}")
    other_ops = ["feature_weighting", "masking", "sum_over_cameras_z", "final_division"]
    for key in other_ops:
        if key in flops_dense:
            value = flops_dense[key]
            print(f"    ✓  {key:28s}: {value:15,.0f} FLOPs ({value/1e9:8.2f} GFLOPs)")
    print()
    print(f"  {'Total':30s}: {flops_dense['total']:15,.0f} FLOPs ({flops_dense['total_gflops']:8.2f} GFLOPs)")
    print()
    
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
    
    all_operations = [
        ("Projection", "projection_per_voxel", "projection", True, "B*N*X*Y*Z*12", "B*X*Y*Z*N*C*12"),
        ("Normalization", "normalization", "normalization", True, "B*N*X*Y*Z*3", "B*X*Y*Z*N*C*3"),
        ("Coordinate Computation", "coordinate_computation", "coordinate_computation", True, "B*N*X*Y*Z*4", "B*X*Y*Z*N*C*4"),
        ("Bilinear Weights", "bilinear_weights", "bilinear_weights", True, "B*N*X*Y*Z*4", "B*X*Y*Z*N*C*4"),
        ("Depth Interpolation", "depth_interpolation", "grid_sample_depths", True, "B*N*X*Y*Z*14", "B*X*Y*Z*N*C*14"),
        ("Depth Weight Calc", "depth_weight_calc", "depth_weight_calc", True, "B*N*X*Y*Z*7", "B*X*Y*Z*N*C*7"),
        ("Feature Interpolation", "feature_interpolation", "grid_sample_features", False, "B*N*X*Y*Z*C*7", "B*X*Y*Z*N*C*7"),
        ("Depth Likelihood Weighting", "depth_likelihood_weighting", "feature_weighting", False, "B*N*X*Y*Z*C*1", "B*X*Y*Z*N*C*1"),
        ("BEV Accumulation", "bev_accumulation", "sum_over_cameras_z", False, "B*X*Y*C*(N*Z-1)", "B*X*Y*C*(N*Z-1)"),
        ("Valid Count", "valid_count_increment", "masking", True, "B*N*X*Y*Z*1", "B*X*Y*Z*N*C*1"),
        ("Final Division", "final_division", "final_division", False, "B*X*Y*C*1", "B*X*Y*C*1"),
    ]
    
    recomputed_overhead = 0.0
    total_dense = 0.0
    total_flashbev = 0.0
    recomputed_ops_count = 0
    
    for op_name, flashbev_key, dense_key, is_recomputed, dense_formula, flashbev_formula in all_operations:
        dense_value = 0.0
        flashbev_value = 0.0
        
        if dense_key:
            if isinstance(dense_key, list):
                for key in dense_key:
                    if key in flops_dense:
                        dense_value += flops_dense[key] / 1e9
                        total_dense += flops_dense[key]
            elif dense_key in flops_dense:
                dense_value = flops_dense[dense_key] / 1e9
                total_dense += dense_value * 1e9
        
        if flashbev_key and flashbev_key in flops_flashbev:
            flashbev_value = flops_flashbev[flashbev_key] / 1e9
            total_flashbev += flashbev_value * 1e9
        
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
        
        # Show base formula (Recomp. ×C column indicates if recomputed)
        if is_recomputed:
            # For recomputed operations, show the base formula (without ×C notation)
            # since the Recomp. ×C column already indicates recomputation
            if "B*N*X*Y*Z*C*" in dense_formula:
                # Already has C in formula, show full formula (both compute per channel)
                formula_str = dense_formula
            elif "B*N*X*Y*Z*" in dense_formula:
                # No C in formula, show base formula (Recomp. ×C column shows it's recomputed)
                formula_str = dense_formula
            else:
                formula_str = dense_formula
        else:
            # Show the formula (both should be the same for non-recomputed)
            formula_str = dense_formula if dense_value > 0 else flashbev_formula
        
        # Mark recomputed operations with checkbox
        recomp_marker = "✓" if is_recomputed else " "
        
        print(f"   {op_name:<33s} | {formula_str:<33s} | {recomp_marker:<8s} | {dense_str} | {flashbev_str} | {diff_str}")
        if is_recomputed and recomputed_ops_count < 6:
            print(row_separator)
    
    print(separator)
    total_diff = flops_flashbev['total'] / 1e9 - flops_dense['total'] / 1e9
    print(f"{'TOTAL':<35s} | {'Sum of above':<33s} | {'':<8s} | {flops_dense['total_gflops']:>16.2f} G | {flops_flashbev['total_gflops']:>16.2f} G | {total_diff:>16.2f} G")
    print()
    
    print("=" * 80)
    print("Key Insights")
    print("=" * 80)
    overhead_percentage = (recomputed_overhead / flops_flashbev['total']) * 100
    overhead_vs_dense = recomputed_overhead / flops_dense['total']
    print(f"⚠️  FlashBEV wastes {recomputed_overhead/1e9:.2f} GFLOPs ({overhead_percentage:.1f}% of total) recomputing")
    print(f"    channel-independent operations {num_channels}× times per voxel-camera.")
    print(f"    This overhead is {overhead_vs_dense:.2f}x ({overhead_vs_dense * 100:.1f}%) of Dense PyTorch's total!")
    print()
    print(f"✓  Operations marked ⚠️ RECOMPUTED are computed once in Dense PyTorch")
    print(f"   but recomputed {num_channels}× times in FlashBEV (one thread per channel).")
    print()
    print(f"   Total Ratio: {flops_flashbev['total_gflops'] / flops_dense['total_gflops']:.2f}x")
    print()
    
    print("=" * 145)
    print("Theoretical FLOP Derivation (Step-by-Step)")
    print("=" * 145)
    print()
    print("We derive FLOPs from first principles based on the algorithm structure:")
    print()
    
    # Derive formulas step by step
    total_voxel_camera_combos = batch_size * num_cameras * grid_x * grid_y * grid_z
    total_output_elements = batch_size * grid_x * grid_y * num_channels
    
    print("Step 1: Define problem dimensions")
    print(f"  • Total voxel-camera combinations: B × N × X × Y × Z = {batch_size} × {num_cameras} × {grid_x} × {grid_y} × {grid_z} = {total_voxel_camera_combos:,}")
    print(f"  • Total output elements: B × X × Y × C = {batch_size} × {grid_x} × {grid_y} × {num_channels} = {total_output_elements:,}")
    print()
    
    print("Step 2: Derive FLOPs for channel-independent operations")
    print("  These operations depend only on geometry (3D→2D projection), not on features:")
    print()
    
    shared_flops = calculate_shared_flops_per_voxel_camera("laplace")
    print(f"  a) Projection (3D world → image coordinates):")
    print(f"     • Per voxel-camera: {shared_flops['projection']} FLOPs (matrix multiply)")
    print(f"     • Dense PyTorch: B × N × X × Y × Z × {shared_flops['projection']} = {total_voxel_camera_combos * shared_flops['projection'] / 1e9:.2f} GFLOPs")
    print(f"     • FlashBEV: Same formula × C (recomputed per channel)")
    print()
    
    print(f"  b) Normalization, Coordinate Computation, Bilinear Weights:")
    print(f"     • Per voxel-camera: {shared_flops['normalization'] + shared_flops['coordinate_computation'] + shared_flops['bilinear_weights']} FLOPs")
    print(f"     • Dense PyTorch: B × N × X × Y × Z × FLOPs = computed once")
    print(f"     • FlashBEV: Same × C (recomputed per channel)")
    print()
    
    print(f"  c) Depth Interpolation & Depth Weight Calculation:")
    print(f"     • Per voxel-camera: {shared_flops['depth_interpolation'] + shared_flops['depth_weight_calc']} FLOPs")
    print(f"     • Dense PyTorch: B × N × X × Y × Z × FLOPs = computed once")
    print(f"     • FlashBEV: Same × C (recomputed per channel)")
    print()
    
    print("Step 3: Derive FLOPs for channel-dependent operations")
    print("  These operations must process each channel separately:")
    print()
    
    feature_interp_flops = calculate_feature_interpolation_flops_per_channel()
    print(f"  a) Feature Interpolation (bilinear sampling):")
    print(f"     • Per voxel-camera-channel: {feature_interp_flops} FLOPs (4 mults + 3 adds)")
    print(f"     • Both methods: B × N × X × Y × Z × C × {feature_interp_flops} = {total_voxel_camera_combos * num_channels * feature_interp_flops / 1e9:.2f} GFLOPs")
    print()
    
    print(f"  b) Depth Likelihood Weighting (features × depth weights):")
    print(f"     • Per voxel-camera-channel: 1 FLOP (multiply)")
    print(f"     • Both methods: B × N × X × Y × Z × C × 1 = {total_voxel_camera_combos * num_channels / 1e9:.2f} GFLOPs")
    print()
    
    print("Step 4: Derive FLOPs for accumulation/reduction")
    print("  Accumulating features over cameras and z-dimensions:")
    print()
    
    num_elements_to_sum = num_cameras * grid_z
    print(f"  • For each output element (BEV location, channel), we sum over:")
    print(f"    - {num_cameras} cameras × {grid_z} z-values = {num_elements_to_sum} elements")
    print(f"  • Tree reduction: ({num_elements_to_sum} - 1) = {num_elements_to_sum - 1} additions")
    print(f"  • Both methods: B × X × Y × C × ({num_elements_to_sum} - 1) = {total_output_elements * (num_elements_to_sum - 1) / 1e9:.2f} GFLOPs")
    print()
    
    print("Step 5: Summary - Key Insight")
    print()
    print("  Dense PyTorch total:")
    print(f"    = Channel-independent ops × 1 (computed once)")
    print(f"    + Channel-dependent ops × C (naturally per channel)")
    print(f"    = {flops_dense['total_gflops']:.2f} GFLOPs")
    print()
    print("  FlashBEV total:")
    print(f"    = Channel-independent ops × C (unnecessarily recomputed)")
    print(f"    + Channel-dependent ops × C (naturally per channel)")
    print(f"    = {flops_flashbev['total_gflops']:.2f} GFLOPs")
    print()
    print(f"  Overhead: {flops_flashbev['total_gflops'] - flops_dense['total_gflops']:.2f} GFLOPs")
    print(f"            = Channel-independent ops × (C - 1)")
    print(f"            = {recomputed_overhead/1e9:.2f} GFLOPs")
    print()

