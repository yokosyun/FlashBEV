#!/usr/bin/env python3
"""
Benchmark script for evaluating memory peak and latency of different view transform methods.

Usage:
    # Kernel-only mode (no mmdet3d required):
    python benchmark_view_transform.py --kernel-only --num-height-bins 8,10,16,20,40
    
    # Full system mode (requires mmdet3d):
    python benchmark_view_transform.py --load-calib calib.json
    
Note:
    - Kernel-only mode (--kernel-only): Benchmarks FlashBEV kernel directly, no mmdet3d dependency
    - Full system mode: Requires mmdet3d.models.necks.view_transformer_back.SamplingVT
      If running from flashbev directory, it will try to import from parent repository.
"""

import argparse
import json
import time
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Try to import SamplingVT from flashbevpool (standalone)
SamplingVT = None

from flashbevpool import SamplingVT, flash_bevpool


def load_calibration_params(input_path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load calibration parameters from JSON file."""
    with open(input_path, "r") as f:
        calib_params = json.load(f)
    
    # Convert lists back to tensors
    result = {
        "sensor2ego": torch.tensor(calib_params["sensor2ego"], device=device, dtype=torch.float32),
        "ego2global": torch.tensor(calib_params["ego2global"], device=device, dtype=torch.float32),
        "camera2imgs": torch.tensor(calib_params["camera2imgs"], device=device, dtype=torch.float32),
        "post_rots": torch.tensor(calib_params["post_rots"], device=device, dtype=torch.float32),
        "post_trans": torch.tensor(calib_params["post_trans"], device=device, dtype=torch.float32),
        "bda": torch.tensor(calib_params["bda"], device=device, dtype=torch.float32),
    }
    
    print(f"Calibration parameters loaded from {input_path}")
    return result


def create_dummy_input(
    batch_size: int,
    num_cameras: int,
    in_channels: int,
    feature_h: int,
    feature_w: int,
    device: str = "cuda",
    calib_params: Dict[str, torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], Dict]:
    """Create dummy input for view transformer.
    
    Args:
        batch_size: Batch size
        num_cameras: Number of cameras
        in_channels: Input channels
        feature_h: Feature map height
        feature_w: Feature map width
        device: Device to use
        calib_params: Optional calibration parameters dict. If provided, uses these
                     instead of creating dummy identity matrices.
    """
    # Image features
    img_feat = torch.randn(
        batch_size, num_cameras, in_channels, feature_h, feature_w,
        device=device, dtype=torch.float32
    )
    
    if calib_params is not None:
        # Use provided calibration parameters
        sensor2ego = calib_params["sensor2ego"]
        ego2global = calib_params["ego2global"]
        camera2imgs = calib_params["camera2imgs"]
        post_rots = calib_params["post_rots"]
        post_trans = calib_params["post_trans"]
        bda = calib_params["bda"]
        
        # Ensure batch size matches
        if sensor2ego.shape[0] != batch_size:
            # Expand or slice to match batch_size
            if sensor2ego.shape[0] == 1:
                sensor2ego = sensor2ego.expand(batch_size, -1, -1, -1)
                ego2global = ego2global.expand(batch_size, -1, -1, -1)
                camera2imgs = camera2imgs.expand(batch_size, -1, -1, -1)
                post_rots = post_rots.expand(batch_size, -1, -1, -1)
                post_trans = post_trans.expand(batch_size, -1, -1)
                bda = bda.expand(batch_size, -1, -1)
            else:
                sensor2ego = sensor2ego[:batch_size]
                ego2global = ego2global[:batch_size]
                camera2imgs = camera2imgs[:batch_size]
                post_rots = post_rots[:batch_size]
                post_trans = post_trans[:batch_size]
                bda = bda[:batch_size]
        
        # Ensure number of cameras matches
        if sensor2ego.shape[1] != num_cameras:
            # Slice to match num_cameras
            sensor2ego = sensor2ego[:, :num_cameras]
            ego2global = ego2global[:, :num_cameras]
            camera2imgs = camera2imgs[:, :num_cameras]
            post_rots = post_rots[:, :num_cameras]
            post_trans = post_trans[:, :num_cameras]
    else:
        # Camera parameters (identity matrices for simplicity)
        sensor2ego = torch.eye(4, device=device).view(1, 1, 4, 4).expand(
            batch_size, num_cameras, 4, 4
        )
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).expand(
            batch_size, num_cameras, 4, 4
        )
        
        # Intrinsic matrices (typical camera intrinsics)
        camera2imgs = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size, num_cameras, 3, 3
        )
        camera2imgs[..., 0, 0] = 700.0  # fx
        camera2imgs[..., 1, 1] = 700.0  # fy
        camera2imgs[..., 0, 2] = feature_w / 2  # cx
        camera2imgs[..., 1, 2] = feature_h / 2  # cy
        
        # Post-rotation and post-translation (identity)
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size, num_cameras, 3, 3
        )
        post_trans = torch.zeros(batch_size, num_cameras, 3, device=device)
        
        # BDA (Bird's Eye View Data Augmentation) - identity
        bda = torch.eye(4, device=device).view(1, 4, 4).expand(batch_size, 4, 4)
    
    input_list = [
        img_feat,
        sensor2ego,
        ego2global,
        camera2imgs,
        post_rots,
        post_trans,
        bda,
    ]
    
    return input_list, {}


def create_view_transformer(
    grid_config: Dict,
    sample_grid_z: Tuple[float,float,float],
    input_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    downsample: int,
    fuse_projection: bool,
    use_bev_pool: bool,
    use_shared_memory: bool,
    depth_regression: bool,
    use_bilinear: bool,
    fuse_bilinear: bool,
    device: str = "cuda",
    depth_distribution: str = "laplace",
    optimize_z_precompute: bool = True,
    use_warp_kernel: bool = False,
    use_vectorized_load: bool = False,
    depth_weight_threshold: float = 0.0,
):
    """Create view transformer with specified configuration."""
    transformer = SamplingVT(
        grid_config=grid_config,
        input_size=input_size,
        downsample=downsample,
        in_channels=in_channels,
        out_channels=out_channels,
        fuse_projection=fuse_projection,
        depth_regression=depth_regression,
        use_bev_pool=use_bev_pool,
        use_shared_memory=use_shared_memory,
        use_bilinear=use_bilinear,
        fuse_bilinear=fuse_bilinear,
        sample_grid_z=sample_grid_z,
        depth_distribution=depth_distribution,
        optimize_z_precompute=optimize_z_precompute,
        use_warp_kernel=use_warp_kernel,
        use_vectorized_load=use_vectorized_load,
        depth_weight_threshold=depth_weight_threshold,
    )
    
    transformer = transformer.to(device)
    transformer.eval()
    
    return transformer


def create_projection_matrix(
    sensor2ego: torch.Tensor,
    ego2global: torch.Tensor,
    camera2imgs: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    bda: torch.Tensor,
) -> torch.Tensor:
    """Create 4x4 projection matrices from camera parameters."""
    B, N, _, _ = camera2imgs.shape
    
    extrinsic_matrices = sensor2ego
    extrinsic_matrices = torch.inverse(extrinsic_matrices)
    
    intrinsic_matrices = torch.eye(4, device=camera2imgs.device).repeat(B, N, 1, 1)
    intrinsic_matrices[..., :3, :3] = torch.matmul(post_rots, camera2imgs)
    intrinsic_matrices[..., :3, 2] += post_trans
    
    projection_matrices = torch.matmul(intrinsic_matrices, extrinsic_matrices)
    
    return projection_matrices


def create_flashbevpool_data(
    batch_size: int,
    num_cameras: int,
    in_channels: int,
    feature_h: int,
    feature_w: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    roi_range: List[float],
    device: str = "cuda",
    calib_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Create dummy data for FlashBEVPool kernel-only benchmarking."""
    image_feats = torch.randn(
        batch_size, num_cameras, feature_h, feature_w, in_channels,
        device=device, dtype=torch.float32
    )
    
    depth_params = torch.randn(
        batch_size, num_cameras, feature_h, feature_w, 2,
        device=device, dtype=torch.float32
    )
    depth_params[..., 0] = depth_params[..., 0].abs() * 10.0 + 5.0
    depth_params[..., 1] = depth_params[..., 1].abs() * 2.0 + 0.5
    
    if calib_params is not None:
        sensor2ego = calib_params["sensor2ego"]
        ego2global = calib_params["ego2global"]
        camera2imgs = calib_params["camera2imgs"]
        post_rots = calib_params["post_rots"]
        post_trans = calib_params["post_trans"]
        bda = calib_params["bda"]
        
        if sensor2ego.shape[0] != batch_size:
            if sensor2ego.shape[0] == 1:
                sensor2ego = sensor2ego.expand(batch_size, -1, -1, -1)
                ego2global = ego2global.expand(batch_size, -1, -1, -1)
                camera2imgs = camera2imgs.expand(batch_size, -1, -1, -1)
                post_rots = post_rots.expand(batch_size, -1, -1, -1)
                post_trans = post_trans.expand(batch_size, -1, -1)
                bda = bda.expand(batch_size, -1, -1)
            else:
                sensor2ego = sensor2ego[:batch_size]
                ego2global = ego2global[:batch_size]
                camera2imgs = camera2imgs[:batch_size]
                post_rots = post_rots[:batch_size]
                post_trans = post_trans[:batch_size]
                bda = bda[:batch_size]
        
        if sensor2ego.shape[1] != num_cameras:
            sensor2ego = sensor2ego[:, :num_cameras]
            ego2global = ego2global[:, :num_cameras]
            camera2imgs = camera2imgs[:, :num_cameras]
            post_rots = post_rots[:, :num_cameras]
            post_trans = post_trans[:, :num_cameras]
    else:
        sensor2ego = torch.eye(4, device=device).view(1, 1, 4, 4).expand(batch_size, num_cameras, 4, 4)
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).expand(batch_size, num_cameras, 4, 4)
        camera2imgs = torch.eye(3, device=device).view(1, 1, 3, 3).expand(batch_size, num_cameras, 3, 3)
        camera2imgs[..., 0, 0] = 700.0
        camera2imgs[..., 1, 1] = 700.0
        camera2imgs[..., 0, 2] = feature_w / 2
        camera2imgs[..., 1, 2] = feature_h / 2
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).expand(batch_size, num_cameras, 3, 3)
        post_trans = torch.zeros(batch_size, num_cameras, 3, device=device)
        bda = torch.eye(4, device=device).view(1, 4, 4).expand(batch_size, 4, 4)
    
    projection_matrices = create_projection_matrix(
        sensor2ego, ego2global, camera2imgs, post_rots, post_trans, bda
    )
    
    feature_size = torch.tensor(
        [[feature_h, feature_w]] * batch_size * num_cameras,
        device=device, dtype=torch.int32
    ).view(batch_size, num_cameras, 2)
    
    image_size = torch.tensor([feature_h * 16, feature_w * 16], device=device, dtype=torch.int32)
    roi_range_tensor = torch.tensor(roi_range, device=device, dtype=torch.float32)
    grid_size = torch.tensor([grid_x, grid_y, grid_z], device=device, dtype=torch.int32)
    
    return {
        "image_feats": image_feats,
        "depth_params": depth_params,
        "projection_matrices": projection_matrices,
        "feature_size": feature_size,
        "image_size": image_size,
        "roi_range": roi_range_tensor,
        "grid_size": grid_size,
    }


def benchmark_flashbevpool_kernel(
    data: Dict[str, torch.Tensor],
    depth_distribution: int = 1,
    use_shared_memory: bool = False,
    optimize_z_precompute: bool = True,
    use_warp_kernel: bool = False,
    use_vectorized_load: bool = False,
    epsilon: float = 1e-6,
    depth_weight_threshold: float = 0.0,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = flash_bevpool(
                image_feats=data["image_feats"],
                depth_params=data["depth_params"],
                projection_matrices=data["projection_matrices"],
                feature_size=data["feature_size"],
                image_size=data["image_size"],
                roi_range=data["roi_range"],
                grid_size=data["grid_size"],
                depth_distribution=depth_distribution,
                use_shared_memory=use_shared_memory,
                optimize_z_precompute=optimize_z_precompute,
                use_warp_kernel=use_warp_kernel,
                use_vectorized_load=use_vectorized_load,
                epsilon=epsilon,
                depth_weight_threshold=depth_weight_threshold,
            )
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = flash_bevpool(
                image_feats=data["image_feats"],
                depth_params=data["depth_params"],
                projection_matrices=data["projection_matrices"],
                feature_size=data["feature_size"],
                image_size=data["image_size"],
                roi_range=data["roi_range"],
                grid_size=data["grid_size"],
                depth_distribution=depth_distribution,
                use_shared_memory=use_shared_memory,
                optimize_z_precompute=optimize_z_precompute,
                use_warp_kernel=use_warp_kernel,
                use_vectorized_load=use_vectorized_load,
                epsilon=epsilon,
                depth_weight_threshold=depth_weight_threshold,
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)
    
    peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    latencies = np.array(latencies)
    stats = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "peak_memory_allocated_mb": float(peak_memory_allocated_mb),
        "peak_memory_reserved_mb": float(peak_memory_reserved_mb),
    }
    
    return stats


def benchmark_method(
    transformer: SamplingVT,
    input_list: List[torch.Tensor],
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark a view transform method (requires mmdet3d)."""
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = transformer(input=input_list)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = transformer(input=input_list)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)

    peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    latencies = np.array(latencies)
    stats = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "peak_memory_allocated_mb": float(peak_memory_allocated_mb),
        "peak_memory_reserved_mb": float(peak_memory_reserved_mb),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark view transform methods"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--num-cameras", type=int, default=6,
        help="Number of cameras (default: 6)"
    )
    parser.add_argument("--feature-h", type=int, default=16)
    parser.add_argument("--feature-w", type=int, default=44)
    parser.add_argument("--in-channels", type=int, default=256)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--input-h", type=int, default=256)
    parser.add_argument("--input-w", type=int, default=704)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--grid-config", type=str, default=None)
    parser.add_argument("--load-calib", type=str, default=None,
                        help="Path to calibration JSON file (optional for kernel-only mode)")
    parser.add_argument("--num-height-bins", type=str, default=None,
                        help="Comma-separated list of num_height_bins to test (e.g., '8,10,16,20,40')")
    parser.add_argument("--num-cameras-list", type=str, default=None,
                        help="Comma-separated list of num_cameras to test (e.g., '1,2,3,4,5,6')")
    parser.add_argument("--plot-output", type=str, default=None,
                        help="Output path for plot (auto-determined if not specified)")
    parser.add_argument("--depth-distribution", type=str, default="laplace",
                        choices=["laplace", "gaussian"],
                        help="Depth distribution type: 'laplace' or 'gaussian' (default: laplace)")
    parser.add_argument("--depth-weight-threshold", type=float, default=0.0,
                        help="Threshold for depth weight filtering")
    parser.add_argument("--depth-weight-threshold-list", type=str, default=None,
                        help="Comma-separated list of depth_weight_threshold values to test (e.g., '1e-6,1e-5,1e-4,1e-3')")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Benchmark FlashBEVPool kernel only (no mmdet3d dependency). "
                             "Only FlashBEV methods will be benchmarked.")
    
    args = parser.parse_args()
    
    # Default grid config (NuScenes-like)
    if args.grid_config:
        grid_config = json.loads(args.grid_config)
    else:
        grid_config = {
            "x": [-51.2, 51.2, 0.8],
            "y": [-51.2, 51.2, 0.4],
            "z": [-5.0, 3.0, 8.0],
            "depth": [1.0, 60.0, 1.0],
        }
    
    # Z range for sample_grid_z
    z_min, z_max = -5.0, 3.0
    z_range = z_max - z_min  # 8.0
    
    # Determine experiment type
    has_height_bins_exp = args.num_height_bins is not None
    has_cameras_exp = args.num_cameras_list is not None
    has_depth_threshold_exp = args.depth_weight_threshold_list is not None
    
    if sum([has_height_bins_exp, has_cameras_exp, has_depth_threshold_exp]) > 1:
        raise ValueError("Cannot run multiple experiments simultaneously. "
                       "Please specify only one: --num-height-bins OR --num-cameras-list OR --depth-weight-threshold-list")
    
    if not has_height_bins_exp and not has_cameras_exp and not has_depth_threshold_exp:
        # Default to single height bins experiment
        num_height_bins_list = [10]
        has_height_bins_exp = True
    
    # Parse num_height_bins if provided
    if has_height_bins_exp:
        if args.num_height_bins:
            num_height_bins_list = [int(x.strip()) for x in args.num_height_bins.split(",")]
        else:
            num_height_bins_list = [10]  # Default single value
        # Convert num_height_bins to z_resolutions
        z_resolutions = [z_range / num_bins for num_bins in num_height_bins_list]
        num_cameras_list = [args.num_cameras]  # Use fixed num_cameras
        depth_weight_threshold_list = [args.depth_weight_threshold]  # Use fixed threshold
    elif has_depth_threshold_exp:
        # Depth threshold experiment
        depth_weight_threshold_list = [float(x.strip()) for x in args.depth_weight_threshold_list.split(",")]
        num_height_bins_list = [10]  # Default fixed value
        z_resolutions = [z_range / num_height_bins_list[0]]
        num_cameras_list = [args.num_cameras]  # Use fixed num_cameras
    else:
        # Camera-wise experiment
        num_cameras_list = [int(x.strip()) for x in args.num_cameras_list.split(",")]
        num_height_bins_list = [10]  # Default fixed value
        z_resolutions = [z_range / num_height_bins_list[0]]
        depth_weight_threshold_list = [args.depth_weight_threshold]  # Use fixed threshold
    
    # Define methods to benchmark
    all_methods = [
        {
            "name": "FlashBEV(baseline)",
            "fuse_projection": True,
            "use_bev_pool": False,
            "use_shared_memory": False,
            "depth_regression": True,
            "use_bilinear": True,
            "optimize_z_precompute": False,
        },
        {
            "name": "FlashBEV(warp_kernel)",
            "fuse_projection": True,
            "use_bev_pool": False,
            "use_shared_memory": False,
            "depth_regression": True,
            "use_bilinear": True,
            "optimize_z_precompute": False,
            "use_warp_kernel": True,
            "use_vectorized_load": False,
        },
        {
            "name": "Sampling-VT(FullPytorch)",
            "fuse_projection": False,
            "use_bev_pool": False,
            "depth_regression": True,
            "use_bilinear": True,
        },
        {
            "name": "Sampling-VT(PillarPool)",
            "fuse_projection": False,
            "use_bev_pool": True,
            "depth_regression": True,
            "use_bilinear": True,
            "fuse_bilinear": True,
        },
    ]
    
    # Filter methods based on mode
    if args.kernel_only:
        methods = [m for m in all_methods if m["fuse_projection"] and not m.get("use_bev_pool", False)]
        if not methods:
            raise ValueError("No FlashBEV methods found for kernel-only benchmarking.")
        print("Kernel-only mode: Only benchmarking FlashBEV methods (no mmdet3d required)")
    else:
        methods = all_methods
    
    # Calculate grid dimensions
    grid_x = int((grid_config["x"][1] - grid_config["x"][0]) / grid_config["x"][2])
    grid_y = int((grid_config["y"][1] - grid_config["y"][0]) / grid_config["y"][2])
    roi_range = [
        grid_config["x"][0], grid_config["x"][1],
        grid_config["y"][0], grid_config["y"][1],
        grid_config["z"][0], grid_config["z"][1],
    ]
        
    input_size = (args.input_h, args.input_w)
    
    # Load calibration parameters if provided
    calib_params = None
    if args.load_calib:
        calib_params = load_calibration_params(args.load_calib, device=args.device)
    elif not args.kernel_only:
        print("Warning: --load-calib not provided. Using dummy calibration parameters.")
    
    # Store results
    all_results = []
    
    depth_distribution_int = 1 if args.depth_distribution == "laplace" else 0
    
    if has_depth_threshold_exp:
        # Depth weight threshold experiment
        memory_data = {method["name"]: {"depth_weight_threshold": [], "memory_mb": [], "latency_ms": []} for method in methods}
        x_axis_label = "depth_weight_threshold"
        x_axis_values = depth_weight_threshold_list
        plot_title = "Memory Usage vs Depth Weight Threshold"
        default_plot_name = "memory_vs_depth_weight_threshold.png"
        latency_plot_title = "Latency vs Depth Weight Threshold"
        latency_plot_name = "latency_vs_depth_weight_threshold.png"
        
        print("\n" + "=" * 80)
        print("Benchmarking View Transform Methods (Depth Weight Threshold Experiment)")
        print("=" * 80)
        print(f"Batch size: {args.batch_size}")
        print(f"Num cameras: {args.num_cameras} (fixed)")
        print(f"Feature size: {args.feature_h} x {args.feature_w}")
        print(f"Input size: {args.input_h} x {args.input_w}")
        print(f"Grid config: {grid_config}")
        print(f"Num height bins: {num_height_bins_list[0]} (fixed)")
        print(f"Depth weight thresholds to test: {depth_weight_threshold_list}")
        print(f"Warmup iterations: {args.num_warmup}")
        print(f"Benchmark iterations: {args.num_iterations}")
        print("=" * 80 + "\n")
        
        # Create dummy input only if needed for full system benchmarks
        input_list = None
        sample_grid_z = None
        if not args.kernel_only and any(not (m["fuse_projection"] and not m.get("use_bev_pool", False)) for m in methods):
            print("Creating dummy input for full system benchmarks...")
            input_list, _ = create_dummy_input(
                batch_size=args.batch_size,
                num_cameras=args.num_cameras,
                in_channels=args.in_channels,
                feature_h=args.feature_h,
                feature_w=args.feature_w,
                device=args.device,
                calib_params=calib_params,
            )
            sample_grid_z = [z_min, z_max, z_resolutions[0]]
        
        # Loop over depth_weight_threshold values
        for depth_threshold in depth_weight_threshold_list:
            print(f"\n{'='*80}")
            print(f"Testing Depth Weight Threshold: {depth_threshold}")
            print(f"{'='*80}\n")
            
            results = []
            
            for method_config in methods:
                method_name = method_config["name"]
                print(f"Benchmarking {method_name} (depth_weight_threshold={depth_threshold})...")
                
                try:
                    is_flashbev = method_config["fuse_projection"] and not method_config.get("use_bev_pool", False)
                    
                    if is_flashbev and args.kernel_only:
                        grid_z = num_height_bins_list[0]
                        data = create_flashbevpool_data(
                            batch_size=args.batch_size,
                            num_cameras=args.num_cameras,
                            in_channels=args.in_channels,
                            feature_h=args.feature_h,
                            feature_w=args.feature_w,
                            grid_x=grid_x,
                            grid_y=grid_y,
                            grid_z=grid_z,
                            roi_range=roi_range,
                            device=args.device,
                            calib_params=calib_params,
                        )
                        
                        stats = benchmark_flashbevpool_kernel(
                            data=data,
                            depth_distribution=depth_distribution_int,
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            epsilon=1e-6,
                            depth_weight_threshold=depth_threshold,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    else:
                        if input_list is None:
                            input_list, _ = create_dummy_input(
                                batch_size=args.batch_size,
                                num_cameras=args.num_cameras,
                                in_channels=args.in_channels,
                                feature_h=args.feature_h,
                                feature_w=args.feature_w,
                                device=args.device,
                                calib_params=calib_params,
                            )
                        if sample_grid_z is None:
                            sample_grid_z = [z_min, z_max, z_resolutions[0]]
                        
                        transformer = create_view_transformer(
                            grid_config=grid_config,
                            sample_grid_z=sample_grid_z,
                            input_size=input_size,
                            in_channels=args.in_channels,
                            out_channels=args.out_channels,
                            downsample=args.downsample,
                            fuse_projection=method_config["fuse_projection"],
                            use_bev_pool=method_config["use_bev_pool"],
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            depth_regression=method_config["depth_regression"],
                            use_bilinear=method_config["use_bilinear"],
                            fuse_bilinear=method_config.get("fuse_bilinear"),
                            device=args.device,
                            depth_distribution=method_config.get("depth_distribution", args.depth_distribution),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            depth_weight_threshold=depth_threshold,
                        )
                        
                        stats = benchmark_method(
                            transformer=transformer,
                            input_list=input_list,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    
                    # Store results
                    result = {
                        "method": method_name,
                        "num_height_bins": num_height_bins_list[0],
                        "num_cameras": args.num_cameras,
                        "z_resolution": z_resolutions[0],
                        "depth_weight_threshold": depth_threshold,
                        **method_config,
                        **stats,
                    }
                    results.append(result)
                    all_results.append(result)
                    
                    # Store for plotting
                    memory_data[method_name]["depth_weight_threshold"].append(depth_threshold)
                    memory_data[method_name]["memory_mb"].append(stats['peak_memory_allocated_mb'])
                    memory_data[method_name]["latency_ms"].append(stats['latency_mean_ms'])
                    
                    print(f"  ✓ Latency: {stats['latency_mean_ms']:.2f} ± {stats['latency_std_ms']:.2f} ms")
                    print(f"  ✓ Peak Memory (allocated): {stats['peak_memory_allocated_mb']:.2f} MB")
                    print(f"  ✓ Peak Memory (reserved): {stats['peak_memory_reserved_mb']:.2f} MB")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                print()
    
    elif has_height_bins_exp:
        # Height bins experiment
        memory_data = {method["name"]: {"num_height_bins": [], "memory_mb": [], "latency_ms": []} for method in methods}
        x_axis_label = "num_height_bins"
        x_axis_values = num_height_bins_list
        plot_title = "Memory Usage vs Num Height Bins"
        default_plot_name = "memory_vs_num_height_bins.png"
        latency_plot_title = "Latency vs Num Height Bins"
        latency_plot_name = "latency_vs_num_height_bins.png"
        
        print("\n" + "=" * 80)
        print("Benchmarking View Transform Methods (Height Bins Experiment)")
        print("=" * 80)
        print(f"Batch size: {args.batch_size}")
        print(f"Num cameras: {args.num_cameras} (fixed)")
        print(f"Feature size: {args.feature_h} x {args.feature_w}")
        print(f"Input size: {args.input_h} x {args.input_w}")
        print(f"Grid config: {grid_config}")
        print(f"Num height bins to test: {num_height_bins_list}")
        print(f"Warmup iterations: {args.num_warmup}")
        print(f"Benchmark iterations: {args.num_iterations}")
        print("=" * 80 + "\n")
        
        # Create dummy input only if needed for full system benchmarks
        input_list = None
        if not args.kernel_only and any(not (m["fuse_projection"] and not m.get("use_bev_pool", False)) for m in methods):
            print("Creating dummy input for full system benchmarks...")
            input_list, _ = create_dummy_input(
                batch_size=args.batch_size,
                num_cameras=args.num_cameras,
                in_channels=args.in_channels,
                feature_h=args.feature_h,
                feature_w=args.feature_w,
                device=args.device,
                calib_params=calib_params,
            )
        
        # Loop over num_height_bins
        for num_bins, z_res in zip(num_height_bins_list, z_resolutions):
            sample_grid_z = [z_min, z_max, z_res]
            print(f"\n{'='*80}")
            print(f"Testing Num Height Bins: {num_bins} (z_res={z_res:.4f})")
            print(f"{'='*80}\n")
            
            results = []
            
            for method_config in methods:
                method_name = method_config["name"]
                print(f"Benchmarking {method_name} (num_height_bins={num_bins})...")
                
                try:
                    # Check if this is a FlashBEV method (kernel-only)
                    is_flashbev = method_config["fuse_projection"] and not method_config.get("use_bev_pool", False)
                    
                    if is_flashbev and args.kernel_only:
                        # Kernel-only benchmarking
                        grid_z = num_bins
                        data = create_flashbevpool_data(
                            batch_size=args.batch_size,
                            num_cameras=args.num_cameras,
                            in_channels=args.in_channels,
                            feature_h=args.feature_h,
                            feature_w=args.feature_w,
                            grid_x=grid_x,
                            grid_y=grid_y,
                            grid_z=grid_z,
                            roi_range=roi_range,
                            device=args.device,
                            calib_params=calib_params,
                        )
                        
                        stats = benchmark_flashbevpool_kernel(
                            data=data,
                            depth_distribution=depth_distribution_int,
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            epsilon=1e-6,
                            depth_weight_threshold=args.depth_weight_threshold,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    else:
                        # Full system benchmarking (requires mmdet3d)
                        transformer = create_view_transformer(
                            grid_config=grid_config,
                            sample_grid_z=sample_grid_z,
                            input_size=input_size,
                            in_channels=args.in_channels,
                            out_channels=args.out_channels,
                            downsample=args.downsample,
                            fuse_projection=method_config["fuse_projection"],
                            use_bev_pool=method_config["use_bev_pool"],
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            depth_regression=method_config["depth_regression"],
                            use_bilinear=method_config["use_bilinear"],
                            fuse_bilinear=method_config.get("fuse_bilinear"),
                            device=args.device,
                            depth_distribution=method_config.get("depth_distribution", args.depth_distribution),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            depth_weight_threshold=args.depth_weight_threshold,
                        )
                        
                        stats = benchmark_method(
                            transformer=transformer,
                            input_list=input_list,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    
                    # Store results
                    result = {
                        "method": method_name,
                        "num_height_bins": num_bins,
                        "num_cameras": args.num_cameras,
                        "z_resolution": z_res,
                        "depth_weight_threshold": args.depth_weight_threshold,
                        **method_config,
                        **stats,
                    }
                    results.append(result)
                    all_results.append(result)
                    
                    # Store for plotting
                    memory_data[method_name]["num_height_bins"].append(num_bins)
                    memory_data[method_name]["memory_mb"].append(stats['peak_memory_allocated_mb'])
                    memory_data[method_name]["latency_ms"].append(stats['latency_mean_ms'])
                    
                    print(f"  ✓ Latency: {stats['latency_mean_ms']:.2f} ± {stats['latency_std_ms']:.2f} ms")
                    print(f"  ✓ Peak Memory (allocated): {stats['peak_memory_allocated_mb']:.2f} MB")
                    print(f"  ✓ Peak Memory (reserved): {stats['peak_memory_reserved_mb']:.2f} MB")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                print()
    
    else:
        # Camera-wise experiment
        memory_data = {method["name"]: {"num_cameras": [], "memory_mb": [], "latency_ms": []} for method in methods}
        x_axis_label = "num_cameras"
        x_axis_values = num_cameras_list
        plot_title = "Memory Usage vs Num Cameras"
        default_plot_name = "memory_vs_num_cameras.png"
        latency_plot_title = "Latency vs Num Cameras"
        latency_plot_name = "latency_vs_num_cameras.png"
        
        print("\n" + "=" * 80)
        print("Benchmarking View Transform Methods (Camera-wise Experiment)")
        print("=" * 80)
        print(f"Batch size: {args.batch_size}")
        print(f"Num cameras to test: {num_cameras_list}")
        print(f"Feature size: {args.feature_h} x {args.feature_w}")
        print(f"Input size: {args.input_h} x {args.input_w}")
        print(f"Grid config: {grid_config}")
        print(f"Num height bins: {num_height_bins_list[0]} (fixed)")
        print(f"Warmup iterations: {args.num_warmup}")
        print(f"Benchmark iterations: {args.num_iterations}")
        print("=" * 80 + "\n")
        
        # Fixed z_resolution for camera experiment
        sample_grid_z = [z_min, z_max, z_resolutions[0]]
        
        # Loop over num_cameras
        for num_cams in num_cameras_list:
            print(f"\n{'='*80}")
            print(f"Testing Num Cameras: {num_cams}")
            print(f"{'='*80}\n")
            
            # Create dummy input only if needed for full system benchmarks
            input_list = None
            if not args.kernel_only and any(not (m["fuse_projection"] and not m.get("use_bev_pool", False)) for m in methods):
                print(f"Creating dummy input for {num_cams} cameras...")
                input_list, _ = create_dummy_input(
                    batch_size=args.batch_size,
                    num_cameras=num_cams,
                    in_channels=args.in_channels,
                    feature_h=args.feature_h,
                    feature_w=args.feature_w,
                    device=args.device,
                    calib_params=calib_params,
                )
            
            results = []
            
            for method_config in methods:
                method_name = method_config["name"]
                print(f"Benchmarking {method_name} (num_cameras={num_cams})...")
                
                try:
                    is_flashbev = method_config["fuse_projection"] and not method_config.get("use_bev_pool", False)
                    
                    if is_flashbev and args.kernel_only:
                        grid_z = num_height_bins_list[0]
                        data = create_flashbevpool_data(
                            batch_size=args.batch_size,
                            num_cameras=num_cams,
                            in_channels=args.in_channels,
                            feature_h=args.feature_h,
                            feature_w=args.feature_w,
                            grid_x=grid_x,
                            grid_y=grid_y,
                            grid_z=grid_z,
                            roi_range=roi_range,
                            device=args.device,
                            calib_params=calib_params,
                        )
                        
                        stats = benchmark_flashbevpool_kernel(
                            data=data,
                            depth_distribution=depth_distribution_int,
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            epsilon=1e-6,
                            depth_weight_threshold=args.depth_weight_threshold,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    else:
                        if input_list is None:
                            input_list, _ = create_dummy_input(
                                batch_size=args.batch_size,
                                num_cameras=num_cams,
                                in_channels=args.in_channels,
                                feature_h=args.feature_h,
                                feature_w=args.feature_w,
                                device=args.device,
                                calib_params=calib_params,
                            )
                        if sample_grid_z is None:
                            sample_grid_z = [z_min, z_max, z_res]
                        
                        transformer = create_view_transformer(
                            grid_config=grid_config,
                            sample_grid_z=sample_grid_z,
                            input_size=input_size,
                            in_channels=args.in_channels,
                            out_channels=args.out_channels,
                            downsample=args.downsample,
                            fuse_projection=method_config["fuse_projection"],
                            use_bev_pool=method_config["use_bev_pool"],
                            use_shared_memory=method_config.get("use_shared_memory", False),
                            depth_regression=method_config["depth_regression"],
                            use_bilinear=method_config["use_bilinear"],
                            fuse_bilinear=method_config.get("fuse_bilinear"),
                            device=args.device,
                            depth_distribution=method_config.get("depth_distribution", args.depth_distribution),
                            optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                            use_warp_kernel=method_config.get("use_warp_kernel", False),
                            use_vectorized_load=method_config.get("use_vectorized_load", False),
                            depth_weight_threshold=args.depth_weight_threshold,
                        )
                        
                        stats = benchmark_method(
                            transformer=transformer,
                            input_list=input_list,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=args.device,
                        )
                    
                    # Store results
                    result = {
                        "method": method_name,
                        "num_height_bins": num_height_bins_list[0],
                        "num_cameras": num_cams,
                        "z_resolution": z_resolutions[0],
                        "depth_weight_threshold": args.depth_weight_threshold,
                        **method_config,
                        **stats,
                    }
                    results.append(result)
                    all_results.append(result)
                    
                    # Store for plotting
                    memory_data[method_name]["num_cameras"].append(num_cams)
                    memory_data[method_name]["memory_mb"].append(stats['peak_memory_allocated_mb'])
                    memory_data[method_name]["latency_ms"].append(stats['latency_mean_ms'])
                    
                    print(f"  ✓ Latency: {stats['latency_mean_ms']:.2f} ± {stats['latency_std_ms']:.2f} ms")
                    print(f"  ✓ Peak Memory (allocated): {stats['peak_memory_allocated_mb']:.2f} MB")
                    print(f"  ✓ Peak Memory (reserved): {stats['peak_memory_reserved_mb']:.2f} MB")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                print()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if len(x_axis_values) == 1:
        # Single value - show detailed table
        if has_height_bins_exp:
            current_results = [r for r in all_results if r["num_height_bins"] == x_axis_values[0]]
        elif has_depth_threshold_exp:
            current_results = [r for r in all_results if r["depth_weight_threshold"] == x_axis_values[0]]
        else:
            current_results = [r for r in all_results if r["num_cameras"] == x_axis_values[0]]
        table_data = []
        for result in current_results:
            table_data.append([
                result["method"],
                f"{result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f}",
                f"{result['latency_p95_ms']:.2f}",
                f"{result['latency_p99_ms']:.2f}",
                f"{result['peak_memory_allocated_mb']:.2f}",
                f"{result['peak_memory_reserved_mb']:.2f}",
            ])
        
        headers = [
            "Method",
            "Latency (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Peak Mem Alloc (MB)",
            "Peak Mem Resv (MB)",
        ]
        
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # Simple table without tabulate
            col_widths = [max(len(str(row[i])) for row in table_data + [headers]) for i in range(len(headers))]
            print(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
            print("-" * sum(col_widths) + "-" * (len(headers) - 1) * 3)
            for row in table_data:
                print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    else:
        # Multiple values - show summary
        if has_height_bins_exp:
            print(f"Tested {len(x_axis_values)} num_height_bins: {x_axis_values}")
        elif has_depth_threshold_exp:
            print(f"Tested {len(x_axis_values)} depth_weight_thresholds: {x_axis_values}")
        else:
            print(f"Tested {len(x_axis_values)} num_cameras: {x_axis_values}")
        print(f"Total results: {len(all_results)}")
    print()
    
    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Determine plot output paths
    if args.plot_output:
        # If user provided a custom path, use it (but if it's just a filename, put it in outputs/)
        if os.path.dirname(args.plot_output):
            plot_output = args.plot_output
        else:
            plot_output = str(outputs_dir / args.plot_output)
        # For latency plot, replace "memory" with "latency" in filename if custom name provided
        latency_plot_output = plot_output.replace("memory", "latency")
    else:
        # Default: save to outputs directory
        plot_output = str(outputs_dir / default_plot_name)
        latency_plot_output = str(outputs_dir / latency_plot_name)
    
    # Plot memory vs x_axis
    if len(x_axis_values) > 1 and plot_output:
        print(f"Generating memory plot: {plot_output}...")
        plt.figure(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for idx, (method_name, data) in enumerate(memory_data.items()):
            if len(data[x_axis_label]) > 0:
                x_values = data[x_axis_label]
                memory = data["memory_mb"]
                # Sort by x_axis for clean line plot
                sorted_pairs = sorted(zip(x_values, memory))
                x_sorted, memory_sorted = zip(*sorted_pairs)
                
                plt.plot(
                    x_sorted, memory_sorted,
                    marker=markers[idx % len(markers)],
                    label=method_name,
                    linewidth=2,
                    markersize=8,
                    color=colors[idx % len(colors)]
                )
        
        if has_height_bins_exp:
            xlabel = "Num Height Bins"
        elif has_depth_threshold_exp:
            xlabel = "Depth Weight Threshold"
            plt.xscale('log')
        else:
            xlabel = "Num Cameras"
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Peak Memory Allocated (MB)", fontsize=12)
        plt.title(plot_title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_output, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {plot_output}")
        print()
    
    # Plot latency vs x_axis
    if len(x_axis_values) > 1 and latency_plot_output:
        print(f"Generating latency plot: {latency_plot_output}...")
        plt.figure(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for idx, (method_name, data) in enumerate(memory_data.items()):
            if len(data[x_axis_label]) > 0:
                x_values = data[x_axis_label]
                latency = data["latency_ms"]
                # Sort by x_axis for clean line plot
                sorted_pairs = sorted(zip(x_values, latency))
                x_sorted, latency_sorted = zip(*sorted_pairs)
                
                plt.plot(
                    x_sorted, latency_sorted,
                    marker=markers[idx % len(markers)],
                    label=method_name,
                    linewidth=2,
                    markersize=8,
                    color=colors[idx % len(colors)]
                )
        
        if has_height_bins_exp:
            xlabel = "Num Height Bins"
        elif has_depth_threshold_exp:
            xlabel = "Depth Weight Threshold"
            plt.xscale('log')
        else:
            xlabel = "Num Cameras"
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Latency (ms)", fontsize=12)
        plt.title(latency_plot_title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(latency_plot_output, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to {latency_plot_output}")
        print()
    
    # Save results
    if args.output_csv:
        print(f"Saving results to {args.output_csv}...")
        with open(args.output_csv, "w", newline="") as f:
            if len(all_results) > 0:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print("  ✓ Saved CSV")
    
    if args.output_json:
        print(f"Saving results to {args.output_json}...")
        with open(args.output_json, "w") as f:
            json.dump({
                "config": {
                    "batch_size": args.batch_size,
                    "num_cameras": args.num_cameras,
                    "feature_h": args.feature_h,
                    "feature_w": args.feature_w,
                    "in_channels": args.in_channels,
                    "out_channels": args.out_channels,
                    "input_h": args.input_h,
                    "input_w": args.input_w,
                    "downsample": args.downsample,
                    "num_warmup": args.num_warmup,
                    "num_iterations": args.num_iterations,
                    "device": args.device,
                    "grid_config": grid_config,
                    "num_height_bins": num_height_bins_list,
                    "num_cameras_list": num_cameras_list if has_cameras_exp else None,
                    "depth_weight_threshold_list": depth_weight_threshold_list if has_depth_threshold_exp else None,
                    "z_resolutions": z_resolutions,
                    "experiment_type": "height_bins" if has_height_bins_exp else ("depth_threshold" if has_depth_threshold_exp else "cameras"),
                    "methods": methods,
                },
                "results": all_results,
            }, f, indent=2)
        print("  ✓ Saved JSON")
    
    print("\nBenchmark complete!")
    
    # Optional: Compare outputs (only if not kernel-only or if mmdet3d available)
    if not args.kernel_only:
        print("\n" + "=" * 80)  
        print("Output Comparison")
        print("=" * 80)
        
        bev_feat_list = []
        num_cams = args.num_cameras
        input_list, _ = create_dummy_input(
            batch_size=args.batch_size,
            num_cameras=num_cams,
            in_channels=args.in_channels,
            feature_h=args.feature_h,
            feature_w=args.feature_w,
            device=args.device,
            calib_params=calib_params,
        )
        
        for method_config in methods:
            method_name = method_config["name"]
            print(f"Computing output for {method_name}...")
            sample_grid_z = [z_min, z_max, z_resolutions[0]]
            
            try:
                transformer = create_view_transformer(
                    grid_config=grid_config,
                    sample_grid_z=sample_grid_z,
                    input_size=input_size,
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    downsample=args.downsample,
                    fuse_projection=method_config["fuse_projection"],
                    use_bev_pool=method_config["use_bev_pool"],
                    use_shared_memory=method_config.get("use_shared_memory", False),
                    depth_regression=method_config["depth_regression"],
                    use_bilinear=method_config["use_bilinear"],
                    fuse_bilinear=method_config.get("fuse_bilinear"),
                    device=args.device,
                    depth_distribution=method_config.get("depth_distribution", args.depth_distribution),
                    optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                    use_warp_kernel=method_config.get("use_warp_kernel", False),
                    use_vectorized_load=method_config.get("use_vectorized_load", False),
                    depth_weight_threshold=args.depth_weight_threshold,
                )
                
                bev_feat, _ = transformer(input=input_list)
                print(f"  Shape: {bev_feat.shape}")
                bev_feat_list.append(bev_feat)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        if len(bev_feat_list) > 1:
            print("\nComparing outputs...")
            for i, (bev_feat, method) in enumerate(zip(bev_feat_list, methods)):
                if i == 0:
                    continue
                diff = torch.max((bev_feat_list[0] - bev_feat).abs())
                print(f"{method['name']} vs {methods[0]['name']}:")
                print(f"  Max diff: {diff.item():.6e}")
                print(f"  Range: [{bev_feat.min().item():.6f}, {bev_feat.max().item():.6f}]")
    elif args.kernel_only:
        print("\nNote: Output comparison skipped in kernel-only mode (requires mmdet3d for full system methods).")

if __name__ == "__main__":
    main()

