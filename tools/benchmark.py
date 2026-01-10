#!/usr/bin/env python3
"""
Benchmark script for evaluating memory peak and latency of different view transform methods.

Usage:
    # Using default config (tools/config/config.yaml):
    python benchmark.py
    
    # Override config from command line:
    python benchmark.py num_height_bins=[8,10,16,20,40] kernel_only=true
    
    # Use a different config file:
    python benchmark.py --config-path=config --config-name=my_config
    
    # Edit config.yaml to customize settings
"""

import csv
import json
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.benchmark_utils import (
    benchmark_flashbevpool_kernel,
    benchmark_method,
)
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

from tabulate import tabulate


def _to_list(value):
    converted = OmegaConf.to_container(value, resolve=True) if value is not None else None
    return converted if isinstance(converted, list) else [converted] if converted is not None else None


def determine_experiment_type(cfg: DictConfig):
    has_height_bins_exp = cfg.num_height_bins is not None
    has_cameras_exp = cfg.num_cameras_list is not None
    has_depth_threshold_exp = cfg.depth_weight_threshold_list is not None
    
    if sum([has_height_bins_exp, has_cameras_exp, has_depth_threshold_exp]) > 1:
        raise ValueError("Cannot run multiple experiments simultaneously. "
                       "Please specify only one: num_height_bins OR num_cameras_list OR depth_weight_threshold_list")
    
    if not has_height_bins_exp and not has_cameras_exp and not has_depth_threshold_exp:
        num_height_bins_list = [10]
        has_height_bins_exp = True
    elif has_height_bins_exp:
        num_height_bins_list = _to_list(cfg.num_height_bins)
    else:
        num_height_bins_list = [10]
    
    depth_weight_threshold_list = _to_list(cfg.depth_weight_threshold_list) if has_depth_threshold_exp else [cfg.depth_weight_threshold]
    num_cameras_list = _to_list(cfg.num_cameras_list) if has_cameras_exp else [cfg.num_cameras]
    
    z_range = cfg.grid_config.z[1] - cfg.grid_config.z[0]
    z_resolutions = [z_range / float(num_bins) for num_bins in num_height_bins_list]
    
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
    cfg: DictConfig,
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
    is_kernel_only: bool,
):
    """Run a single benchmark for a method."""
    method_name = method_config["name"]
    is_flashbev = method_config["fuse_projection"] and not method_config.get("use_bev_pool", False)
    
    try:
        if is_flashbev and is_kernel_only:
            data = create_flashbevpool_data(
                batch_size=cfg.batch_size,
                num_cameras=num_cams,
                in_channels=cfg.in_channels,
                feature_h=cfg.feature_h,
                feature_w=cfg.feature_w,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=num_bins,
                roi_range=roi_range,
                device=cfg.device,
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
                num_warmup=cfg.num_warmup,
                num_iterations=cfg.num_iterations,
            )
        else:
            if input_list is None:
                input_list, _ = create_dummy_input(
                    batch_size=cfg.batch_size,
                    num_cameras=num_cams,
                    in_channels=cfg.in_channels,
                    feature_h=cfg.feature_h,
                    feature_w=cfg.feature_w,
                    device=cfg.device,
                    calib_params=calib_params,
                )
            
            transformer = create_view_transformer(
                grid_config=grid_config,
                input_size=input_size,
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                downsample=cfg.downsample,
                fuse_projection=method_config["fuse_projection"],
                use_bev_pool=method_config["use_bev_pool"],
                use_shared_memory=method_config.get("use_shared_memory", False),
                fuse_bilinear=method_config.get("fuse_bilinear"),
                device=cfg.device,
                depth_distribution=method_config.get("depth_distribution", cfg.depth_distribution),
                optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                use_warp_kernel=method_config.get("use_warp_kernel", False),
                use_vectorized_load=method_config.get("use_vectorized_load", False),
                depth_weight_threshold=depth_threshold,
            )
            
            stats = benchmark_method(
                transformer=transformer,
                input_list=input_list,
                num_warmup=cfg.num_warmup,
                num_iterations=cfg.num_iterations,
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
        if "CUDA error" in str(e):
            print("  Note: This may be a kernel-specific issue. Try running with different parameters.")
        traceback.print_exc()
        return None


def run_experiment(
    cfg: DictConfig,
    exp_config: Dict,
    grid_config: Dict,
    methods: List[Dict],
    calib_params,
    depth_distribution_int: int,
):
    """Run the benchmark experiment."""
    grid_x = int((float(grid_config["x"][1]) - float(grid_config["x"][0])) / float(grid_config["x"][2]))
    grid_y = int((float(grid_config["y"][1]) - float(grid_config["y"][0])) / float(grid_config["y"][2]))
    
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"Invalid grid dimensions: grid_x={grid_x}, grid_y={grid_y}. Check grid_config values.")
    
    roi_range = [
        float(grid_config["x"][0]), float(grid_config["x"][1]),
        float(grid_config["y"][0]), float(grid_config["y"][1]),
        float(grid_config["z"][0]), float(grid_config["z"][1]),
    ]
    input_size = (cfg.input_h, cfg.input_w)
    
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
        
        current_grid_config = {k: list(v) if isinstance(v, list) else v for k, v in grid_config.items()}
        current_grid_config["z"][2] = z_res
        
        print(f"\n{'='*80}")
        print(f"Testing {value_key}: {value}")
        print(f"{'='*80}\n")
        
        input_list = None
        
        if not cfg.kernel_only and any(not (m["fuse_projection"] and not m.get("use_bev_pool", False)) for m in methods):
            input_list, _ = create_dummy_input(
                batch_size=cfg.batch_size,
                num_cameras=num_cams,
                in_channels=cfg.in_channels,
                feature_h=cfg.feature_h,
                feature_w=cfg.feature_w,
                device=cfg.device,
                calib_params=calib_params,
            )
        
        for method_config in methods:
            method_name = method_config["name"]
            num_runs = cfg.get("num_runs", 1)
            
            if num_runs > 1:
                print(f"Benchmarking {method_name} ({value_key}={value}) - {num_runs} independent runs...")
                run_results = []
                
                for run_idx in range(num_runs):
                    print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
                    result = run_single_benchmark(
                        method_config, cfg, current_grid_config, grid_x, grid_y, roi_range,
                        input_size, calib_params, num_bins, z_res, num_cams,
                        depth_threshold, depth_distribution_int, input_list,
                        cfg.kernel_only,
                    )
                    
                    if result:
                        result["run_id"] = run_idx
                        run_results.append(result)
                        print(f"✓ ({result['latency_mean_ms']:.2f} ms)")
                
                if run_results:
                    import numpy as np
                    aggregated = {
                        "method": method_name,
                        "num_height_bins": num_bins,
                        "num_cameras": num_cams,
                        "z_resolution": z_res,
                        "depth_weight_threshold": depth_threshold,
                        **method_config,
                        "latency_mean_ms": float(np.mean([r["latency_mean_ms"] for r in run_results])),
                        "latency_std_ms": float(np.std([r["latency_mean_ms"] for r in run_results])),
                        "latency_min_ms": float(np.min([r["latency_min_ms"] for r in run_results])),
                        "latency_max_ms": float(np.max([r["latency_max_ms"] for r in run_results])),
                        "latency_p50_ms": float(np.mean([r["latency_p50_ms"] for r in run_results])),
                        "latency_p95_ms": float(np.mean([r["latency_p95_ms"] for r in run_results])),
                        "latency_p99_ms": float(np.mean([r["latency_p99_ms"] for r in run_results])),
                        "peak_memory_allocated_mb": float(np.mean([r["peak_memory_allocated_mb"] for r in run_results])),
                        "peak_memory_reserved_mb": float(np.mean([r["peak_memory_reserved_mb"] for r in run_results])),
                        "memory_std_mb": float(np.std([r["peak_memory_allocated_mb"] for r in run_results])),
                        "num_runs": num_runs,
                        "individual_runs": run_results,
                    }
                    
                    all_results.append(aggregated)
                    memory_data[method_name][exp_config["x_axis_label"]].append(value)
                    memory_data[method_name]["memory_mb"].append(aggregated['peak_memory_allocated_mb'])
                    memory_data[method_name]["latency_ms"].append(aggregated['latency_mean_ms'])
                    
                    if num_runs > 1:
                        memory_data[method_name].setdefault("latency_std", []).append(aggregated['latency_std_ms'])
                        memory_data[method_name].setdefault("memory_std", []).append(aggregated['memory_std_mb'])
                        memory_data[method_name].setdefault("individual_runs", []).append(run_results)
                    
                    print(f"  → Aggregated: {aggregated['latency_mean_ms']:.2f} ± {aggregated['latency_std_ms']:.2f} ms")
                    print(f"  → Memory: {aggregated['peak_memory_allocated_mb']:.2f} ± {aggregated['memory_std_mb']:.2f} MB")
            else:
                print(f"Benchmarking {method_name} ({value_key}={value})...")
                
                result = run_single_benchmark(
                    method_config, cfg, current_grid_config, grid_x, grid_y, roi_range,
                    input_size, calib_params, num_bins, z_res, num_cams,
                    depth_threshold, depth_distribution_int, input_list,
                    cfg.kernel_only,
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
            num_runs = result.get("num_runs", 1)
            if num_runs > 1:
                latency_str = f"{result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} (n={num_runs})"
                memory_str = f"{result['peak_memory_allocated_mb']:.2f} ± {result.get('memory_std_mb', 0):.2f}"
            else:
                latency_str = f"{result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f}"
                memory_str = f"{result['peak_memory_allocated_mb']:.2f}"
            
            table_data.append([
                result["method"],
                latency_str,
                f"{result['latency_p95_ms']:.2f}",
                f"{result['latency_p99_ms']:.2f}",
                memory_str,
                f"{result['peak_memory_reserved_mb']:.2f}",
            ])
        
        headers = ["Method", "Latency (ms)", "P95 (ms)", "P99 (ms)", 
                  "Peak Mem Alloc (MB)", "Peak Mem Resv (MB)"]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"Tested {len(x_axis_values)} values: {x_axis_values}")
        print(f"Total results: {len(all_results)}")
    print()
    

def save_results(all_results, cfg: DictConfig, grid_config: Dict, exp_config: Dict):
    """Save results to CSV/JSON files."""
    if cfg.output_csv:
        print(f"Saving results to {cfg.output_csv}...")
        with open(cfg.output_csv, "w", newline="") as f:
            if len(all_results) > 0:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print("  ✓ Saved CSV")
    
    output_json = cfg.output_json
    if not output_json and all_results:
        from pathlib import Path
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_json = str(outputs_dir / "benchmark_results.json")
    
    if output_json:
        print(f"Saving results to {output_json}...")
        with open(output_json, "w") as f:
            json.dump({
                "config": {
                    "batch_size": cfg.batch_size,
                    "num_cameras": cfg.num_cameras,
                    "feature_h": cfg.feature_h,
                    "feature_w": cfg.feature_w,
                    "in_channels": cfg.in_channels,
                    "out_channels": cfg.out_channels,
                    "input_h": cfg.input_h,
                    "input_w": cfg.input_w,
                    "downsample": cfg.downsample,
                    "num_warmup": cfg.num_warmup,
                    "num_iterations": cfg.num_iterations,
                    "device": cfg.device,
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
        print(f"  → Run 'python tools/paper_results.py --results-json {output_json}' to generate paper-ready tables")
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    grid_config = OmegaConf.to_container(cfg.grid_config, resolve=True)
    exp_config = determine_experiment_type(cfg)
    
    exp_type_configs = {
        "height_bins": {
            "x_axis_label": "num_height_bins",
            "plot_title": "Memory Usage vs Num Height Bins",
            "default_plot_name": "memory_vs_num_height_bins.png",
            "latency_plot_title": "Latency vs Num Height Bins",
            "latency_plot_name": "latency_vs_num_height_bins.png",
            "x_axis_values": exp_config["num_height_bins_list"],
        },
        "depth_threshold": {
            "x_axis_label": "depth_weight_threshold",
            "plot_title": "Memory Usage vs Depth Weight Threshold",
            "default_plot_name": "memory_vs_depth_weight_threshold.png",
            "latency_plot_title": "Latency vs Depth Weight Threshold",
            "latency_plot_name": "latency_vs_depth_weight_threshold.png",
            "x_axis_values": exp_config["depth_weight_threshold_list"],
        },
        "cameras": {
            "x_axis_label": "num_cameras",
            "plot_title": "Memory Usage vs Num Cameras",
            "default_plot_name": "memory_vs_num_cameras.png",
            "latency_plot_title": "Latency vs Num Cameras",
            "latency_plot_name": "latency_vs_num_cameras.png",
            "x_axis_values": exp_config["num_cameras_list"],
        },
    }
    
    exp_type = "height_bins" if exp_config["has_height_bins_exp"] else ("depth_threshold" if exp_config["has_depth_threshold_exp"] else "cameras")
    exp_config.update(exp_type_configs[exp_type])
    
    grid_x = int((float(grid_config["x"][1]) - float(grid_config["x"][0])) / float(grid_config["x"][2]))
    grid_y = int((float(grid_config["y"][1]) - float(grid_config["y"][0])) / float(grid_config["y"][2]))
    if grid_x == grid_y:
        grid_resolution_suffix = f" (X=Y={grid_x})"
    else:
        grid_resolution_suffix = f" (X={grid_x}, Y={grid_y})"
    exp_config["plot_title"] += grid_resolution_suffix
    exp_config["latency_plot_title"] += grid_resolution_suffix
    
    methods = [m for m in cfg.methods if m["fuse_projection"] and not m.get("use_bev_pool", False) and not m.get("use_warp_kernel", False)] if cfg.kernel_only else cfg.methods
    
    if cfg.kernel_only:
        print("Kernel-only mode: Only benchmarking FlashBEV methods (no mmdet3d required)")
    
    calib_params = None
    if cfg.load_calib:
        calib_params = load_calibration_params(cfg.load_calib, device=cfg.device)
    elif not cfg.kernel_only:
        print("Warning: load_calib not provided. Using dummy calibration parameters.")
    
    depth_distribution_int = 1 if cfg.depth_distribution == "laplace" else 0
    
    print("\n" + "=" * 80)
    print("Benchmarking View Transform Methods")
    print("=" * 80)
    print(f"Batch size: {cfg.batch_size}")
    print(f"Feature size: {cfg.feature_h} x {cfg.feature_w}")
    print(f"Input size: {cfg.input_h} x {cfg.input_w}")
    print(f"Grid config: {grid_config}")
    print(f"Warmup iterations: {cfg.num_warmup}")
    print(f"Benchmark iterations: {cfg.num_iterations}")
    num_runs = cfg.get("num_runs", 1)
    if num_runs > 1:
        print(f"Independent runs: {num_runs} (results will be aggregated)")
    print("=" * 80 + "\n")
    
    all_results, memory_data, exp_config = run_experiment(
        cfg, exp_config, grid_config, methods, calib_params, depth_distribution_int
    )
    
    print_summary(all_results, exp_config, exp_config["x_axis_values"])
    
    outputs_dir = setup_output_directory()
    plot_output, latency_plot_output = get_plot_paths(
        cfg.plot_output, exp_config["default_plot_name"],
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
    
    save_results(all_results, cfg, grid_config, exp_config)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
