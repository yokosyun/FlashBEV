#!/usr/bin/env python3
"""
Benchmark script for evaluating memory peak and latency of different view transform methods.

Usage:
    # Kernel-only mode (no mmdet3d required):
    python benchmark.py --kernel-only --num-height-bins 8,10,16,20,40
    
    # Full system mode (requires mmdet3d):
    python benchmark.py --load-calib calib.json
"""

import argparse
import csv
import json
from typing import Dict, List

import torch

from utils.benchmark_utils import (
    benchmark_flashbevpool_kernel,
    benchmark_method,
)
from utils.config import DEFAULT_GRID_CONFIG, DEFAULT_METHODS, Z_MAX, Z_MIN, Z_RANGE
from utils.data_utils import (
    create_dummy_input,
    create_flashbevpool_data,
    load_calibration_params,
)
from utils.plotting import (
    get_plot_paths,
    plot_latency_vs_x,
    plot_memory_vs_x,
    setup_output_directory,
)
from utils.transformer_utils import create_view_transformer

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark view transform methods")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num-cameras", type=int, default=6, help="Number of cameras (default: 6)")
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
                        help="Comma-separated list of depth_weight_threshold values to test")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Benchmark FlashBEVPool kernel only (no mmdet3d dependency).")
    return parser.parse_args()


def determine_experiment_type(args):
    """Determine experiment type and parse parameters."""
    has_height_bins_exp = args.num_height_bins is not None
    has_cameras_exp = args.num_cameras_list is not None
    has_depth_threshold_exp = args.depth_weight_threshold_list is not None
    
    if sum([has_height_bins_exp, has_cameras_exp, has_depth_threshold_exp]) > 1:
        raise ValueError("Cannot run multiple experiments simultaneously. "
                       "Please specify only one: --num-height-bins OR --num-cameras-list OR --depth-weight-threshold-list")
    
    if not has_height_bins_exp and not has_cameras_exp and not has_depth_threshold_exp:
        num_height_bins_list = [10]
        has_height_bins_exp = True
    elif has_height_bins_exp:
        num_height_bins_list = [int(x.strip()) for x in args.num_height_bins.split(",")]
    else:
        num_height_bins_list = [10]
    
    if has_depth_threshold_exp:
        depth_weight_threshold_list = [float(x.strip()) for x in args.depth_weight_threshold_list.split(",")]
    else:
        depth_weight_threshold_list = [args.depth_weight_threshold]
    
    if has_cameras_exp:
        num_cameras_list = [int(x.strip()) for x in args.num_cameras_list.split(",")]
    else:
        num_cameras_list = [args.num_cameras]
    
    z_resolutions = [Z_RANGE / num_bins for num_bins in num_height_bins_list]
    
    return {
        "has_height_bins_exp": has_height_bins_exp,
        "has_cameras_exp": has_cameras_exp,
        "has_depth_threshold_exp": has_depth_threshold_exp,
        "num_height_bins_list": num_height_bins_list,
        "num_cameras_list": num_cameras_list,
        "depth_weight_threshold_list": depth_weight_threshold_list,
        "z_resolutions": z_resolutions,
    }


def run_single_benchmark(
    method_config: Dict,
    args,
    grid_config: Dict,
    grid_x: int,
    grid_y: int,
    roi_range: List[float],
    input_size: tuple,
    calib_params,
    num_bins: int,
    z_res: float,
    num_cams: int,
    depth_threshold: float,
    depth_distribution_int: int,
    input_list,
    sample_grid_z,
    is_kernel_only: bool,
):
    """Run a single benchmark for a method."""
    method_name = method_config["name"]
    is_flashbev = method_config["fuse_projection"] and not method_config.get("use_bev_pool", False)
    
    try:
        if is_flashbev and is_kernel_only:
            data = create_flashbevpool_data(
                batch_size=args.batch_size,
                num_cameras=num_cams,
                in_channels=args.in_channels,
                feature_h=args.feature_h,
                feature_w=args.feature_w,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=num_bins,
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
                    num_cameras=num_cams,
                    in_channels=args.in_channels,
                    feature_h=args.feature_h,
                    feature_w=args.feature_w,
                    device=args.device,
                    calib_params=calib_params,
                )
            
            transformer = create_view_transformer(
                grid_config=grid_config,
                sample_grid_z=sample_grid_z or [Z_MIN, Z_MAX, z_res],
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
        
        print(f"  ✓ Latency: {stats['latency_mean_ms']:.2f} ± {stats['latency_std_ms']:.2f} ms")
        print(f"  ✓ Peak Memory (allocated): {stats['peak_memory_allocated_mb']:.2f} MB")
        print(f"  ✓ Peak Memory (reserved): {stats['peak_memory_reserved_mb']:.2f} MB")
        
        return {
            "method": method_name,
            "num_height_bins": num_bins,
            "num_cameras": num_cams,
            "z_resolution": z_res,
            "depth_weight_threshold": depth_threshold,
            **method_config,
            **stats,
        }
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment(
    args,
    exp_config: Dict,
    grid_config: Dict,
    methods: List[Dict],
    calib_params,
    depth_distribution_int: int,
):
    """Run the benchmark experiment."""
    grid_x = int((grid_config["x"][1] - grid_config["x"][0]) / grid_config["x"][2])
    grid_y = int((grid_config["y"][1] - grid_config["y"][0]) / grid_config["y"][2])
    roi_range = [
        grid_config["x"][0], grid_config["x"][1],
        grid_config["y"][0], grid_config["y"][1],
        grid_config["z"][0], grid_config["z"][1],
    ]
    input_size = (args.input_h, args.input_w)
    
    all_results = []
    memory_data = {method["name"]: {exp_config["x_axis_label"]: [], "memory_mb": [], "latency_ms": []} 
                   for method in methods}
    
    if exp_config["has_depth_threshold_exp"]:
        values_to_test = exp_config["depth_weight_threshold_list"]
        value_key = "depth_weight_threshold"
    elif exp_config["has_height_bins_exp"]:
        values_to_test = exp_config["num_height_bins_list"]
        value_key = "num_height_bins"
    else:
        values_to_test = exp_config["num_cameras_list"]
        value_key = "num_cameras"
    
    for idx, value in enumerate(values_to_test):
        if exp_config["has_depth_threshold_exp"]:
            depth_threshold = value
            num_bins = exp_config["num_height_bins_list"][0]
            num_cams = exp_config["num_cameras_list"][0]
            z_res = exp_config["z_resolutions"][0]
        elif exp_config["has_height_bins_exp"]:
            num_bins = value
            z_res = exp_config["z_resolutions"][idx]
            num_cams = exp_config["num_cameras_list"][0]
            depth_threshold = exp_config["depth_weight_threshold_list"][0]
        else:
            num_cams = value
            num_bins = exp_config["num_height_bins_list"][0]
            z_res = exp_config["z_resolutions"][0]
            depth_threshold = exp_config["depth_weight_threshold_list"][0]
        
        print(f"\n{'='*80}")
        print(f"Testing {value_key}: {value}")
        print(f"{'='*80}\n")
        
        sample_grid_z = [Z_MIN, Z_MAX, z_res]
        input_list = None
        
        if not args.kernel_only and any(not (m["fuse_projection"] and not m.get("use_bev_pool", False)) for m in methods):
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
            print(f"Benchmarking {method_name} ({value_key}={value})...")
            
            result = run_single_benchmark(
                method_config, args, grid_config, grid_x, grid_y, roi_range,
                input_size, calib_params, num_bins, z_res, num_cams,
                depth_threshold, depth_distribution_int, input_list, sample_grid_z,
                args.kernel_only,
            )
            
            if result:
                all_results.append(result)
                memory_data[method_name][exp_config["x_axis_label"]].append(value)
                memory_data[method_name]["memory_mb"].append(result['peak_memory_allocated_mb'])
                memory_data[method_name]["latency_ms"].append(result['latency_mean_ms'])
            
            print()
    
    return all_results, memory_data, exp_config


def print_summary(all_results, exp_config, x_axis_values):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if len(x_axis_values) == 1:
        if exp_config["has_height_bins_exp"]:
            current_results = [r for r in all_results if r["num_height_bins"] == x_axis_values[0]]
        elif exp_config["has_depth_threshold_exp"]:
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
        
        headers = ["Method", "Latency (ms)", "P95 (ms)", "P99 (ms)", 
                  "Peak Mem Alloc (MB)", "Peak Mem Resv (MB)"]
        
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            col_widths = [max(len(str(row[i])) for row in table_data + [headers]) for i in range(len(headers))]
            print(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
            print("-" * sum(col_widths) + "-" * (len(headers) - 1) * 3)
            for row in table_data:
                print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    else:
        print(f"Tested {len(x_axis_values)} values: {x_axis_values}")
        print(f"Total results: {len(all_results)}")
    print()


def save_results(all_results, args, grid_config, exp_config):
    """Save results to CSV/JSON files."""
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
                    "num_height_bins": exp_config["num_height_bins_list"],
                    "num_cameras_list": exp_config["num_cameras_list"] if exp_config["has_cameras_exp"] else None,
                    "depth_weight_threshold_list": exp_config["depth_weight_threshold_list"] if exp_config["has_depth_threshold_exp"] else None,
                    "z_resolutions": exp_config["z_resolutions"],
                    "experiment_type": "height_bins" if exp_config["has_height_bins_exp"] else ("depth_threshold" if exp_config["has_depth_threshold_exp"] else "cameras"),
                },
                "results": all_results,
            }, f, indent=2)
        print("  ✓ Saved JSON")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    grid_config = json.loads(args.grid_config) if args.grid_config else DEFAULT_GRID_CONFIG.copy()
    exp_config = determine_experiment_type(args)
    
    if exp_config["has_height_bins_exp"]:
        exp_config["x_axis_label"] = "num_height_bins"
        exp_config["plot_title"] = "Memory Usage vs Num Height Bins"
        exp_config["default_plot_name"] = "memory_vs_num_height_bins.png"
        exp_config["latency_plot_title"] = "Latency vs Num Height Bins"
        exp_config["latency_plot_name"] = "latency_vs_num_height_bins.png"
        exp_config["x_axis_values"] = exp_config["num_height_bins_list"]
    elif exp_config["has_depth_threshold_exp"]:
        exp_config["x_axis_label"] = "depth_weight_threshold"
        exp_config["plot_title"] = "Memory Usage vs Depth Weight Threshold"
        exp_config["default_plot_name"] = "memory_vs_depth_weight_threshold.png"
        exp_config["latency_plot_title"] = "Latency vs Depth Weight Threshold"
        exp_config["latency_plot_name"] = "latency_vs_depth_weight_threshold.png"
        exp_config["x_axis_values"] = exp_config["depth_weight_threshold_list"]
    else:
        exp_config["x_axis_label"] = "num_cameras"
        exp_config["plot_title"] = "Memory Usage vs Num Cameras"
        exp_config["default_plot_name"] = "memory_vs_num_cameras.png"
        exp_config["latency_plot_title"] = "Latency vs Num Cameras"
        exp_config["latency_plot_name"] = "latency_vs_num_cameras.png"
        exp_config["x_axis_values"] = exp_config["num_cameras_list"]
    
    methods = [m for m in DEFAULT_METHODS if m["fuse_projection"] and not m.get("use_bev_pool", False)] if args.kernel_only else DEFAULT_METHODS
    
    if args.kernel_only:
        print("Kernel-only mode: Only benchmarking FlashBEV methods (no mmdet3d required)")
    
    calib_params = None
    if args.load_calib:
        calib_params = load_calibration_params(args.load_calib, device=args.device)
    elif not args.kernel_only:
        print("Warning: --load-calib not provided. Using dummy calibration parameters.")
    
    depth_distribution_int = 1 if args.depth_distribution == "laplace" else 0
    
    print("\n" + "=" * 80)
    print("Benchmarking View Transform Methods")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Feature size: {args.feature_h} x {args.feature_w}")
    print(f"Input size: {args.input_h} x {args.input_w}")
    print(f"Grid config: {grid_config}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")
    print("=" * 80 + "\n")
    
    all_results, memory_data, exp_config = run_experiment(
        args, exp_config, grid_config, methods, calib_params, depth_distribution_int
    )
    
    print_summary(all_results, exp_config, exp_config["x_axis_values"])
    
    outputs_dir = setup_output_directory()
    plot_output, latency_plot_output = get_plot_paths(
        args.plot_output, exp_config["default_plot_name"],
        exp_config["latency_plot_name"], outputs_dir
    )
    
    if len(exp_config["x_axis_values"]) > 1:
        plot_memory_vs_x(
            memory_data, exp_config["x_axis_label"], exp_config["plot_title"],
            plot_output, exp_config["has_height_bins_exp"], exp_config["has_depth_threshold_exp"]
        )
        plot_latency_vs_x(
            memory_data, exp_config["x_axis_label"], exp_config["latency_plot_title"],
            latency_plot_output, exp_config["has_height_bins_exp"], exp_config["has_depth_threshold_exp"]
        )
    
    save_results(all_results, args, grid_config, exp_config)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
