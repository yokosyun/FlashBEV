# Copyright (c) Shunsuke Yokokawa. All rights reserved.

import json
import random
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

from utils.data_utils import create_dummy_input, load_calibration_params
from utils.transformer_utils import create_view_transformer


def compute_difference_metrics(
    baseline: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute various difference metrics between baseline and target tensors."""
    with torch.no_grad():
        diff = target - baseline
        
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        max_diff = torch.max(torch.abs(diff)).item()
        min_diff = torch.min(torch.abs(diff)).item()
        mean_diff = torch.mean(diff).item()
        std_diff = torch.std(diff).item()
        
        baseline_norm = torch.norm(baseline).item()
        target_norm = torch.norm(target).item()
        
        if baseline_norm > 1e-8:
            relative_error = torch.norm(diff).item() / baseline_norm
        else:
            relative_error = float('inf') if target_norm > 1e-8 else 0.0
        
        baseline_flat = baseline.flatten()
        target_flat = target.flatten()
        
        if baseline_norm > 1e-8 and target_norm > 1e-8:
            cosine_sim = torch.nn.functional.cosine_similarity(
                baseline_flat.unsqueeze(0),
                target_flat.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 1.0 if baseline_norm < 1e-8 and target_norm < 1e-8 else 0.0
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "max_abs_diff": max_diff,
            "min_abs_diff": min_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "relative_error": relative_error,
            "cosine_similarity": cosine_sim,
        }
        
        return metrics


def compare_two_methods(
    method1_output: torch.Tensor,
    method2_output: torch.Tensor,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
    tolerance_mse: float = 1e-6,
    tolerance_mae: float = 1e-6,
    tolerance_relative: float = 1e-5,
) -> Dict:
    """Compare two methods with focused metrics and pass/fail assessment.
    
    Args:
        method1_output: Output tensor from first method
        method2_output: Output tensor from second method
        method1_name: Name of first method
        method2_name: Name of second method
        tolerance_mse: Tolerance for MSE (default: 1e-6)
        tolerance_mae: Tolerance for MAE (default: 1e-6)
        tolerance_relative: Tolerance for relative error (default: 1e-5)
    
    Returns:
        Dictionary with metrics and pass/fail status
    """
    if method1_output.shape != method2_output.shape:
        return {
            "error": f"Shape mismatch: {method1_output.shape} vs {method2_output.shape}",
            "match": False,
        }
    
    metrics = compute_difference_metrics(method1_output, method2_output)
    
    mse_pass = metrics["mse"] < tolerance_mse
    mae_pass = metrics["mae"] < tolerance_mae
    relative_pass = metrics["relative_error"] < tolerance_relative
    cosine_pass = metrics["cosine_similarity"] > 0.9999
    
    overall_match = mse_pass and mae_pass and relative_pass and cosine_pass
    
    comparison = {
        "method1": method1_name,
        "method2": method2_name,
        "metrics": metrics,
        "tolerances": {
            "mse": tolerance_mse,
            "mae": tolerance_mae,
            "relative_error": tolerance_relative,
            "cosine_similarity": 0.9999,
        },
        "pass_fail": {
            "mse": mse_pass,
            "mae": mae_pass,
            "relative_error": relative_pass,
            "cosine_similarity": cosine_pass,
        },
        "overall_match": overall_match,
    }
    
    return comparison


def print_two_method_comparison(comparison: Dict):
    """Print a focused comparison table for two methods."""
    if "error" in comparison:
        print(f"\n❌ Error: {comparison['error']}")
        return
    
    print("\n" + "=" * 80)
    print("Two-Method Comparison")
    print("=" * 80)
    print(f"Method 1: {comparison['method1']}")
    print(f"Method 2: {comparison['method2']}\n")
    
    metrics = comparison["metrics"]
    pass_fail = comparison["pass_fail"]
    tolerances = comparison["tolerances"]
    
    table_data = [
        ["Metric", "Value", "Tolerance", "Status"],
        ["MSE", f"{metrics['mse']:.6e}", f"< {tolerances['mse']:.6e}", 
         "✓ PASS" if pass_fail["mse"] else "✗ FAIL"],
        ["MAE", f"{metrics['mae']:.6e}", f"< {tolerances['mae']:.6e}", 
         "✓ PASS" if pass_fail["mae"] else "✗ FAIL"],
        ["Relative Error", f"{metrics['relative_error']:.6e}", f"< {tolerances['relative_error']:.6e}", 
         "✓ PASS" if pass_fail["relative_error"] else "✗ FAIL"],
        ["Cosine Similarity", f"{metrics['cosine_similarity']:.6f}", f"> {tolerances['cosine_similarity']:.4f}", 
         "✓ PASS" if pass_fail["cosine_similarity"] else "✗ FAIL"],
        ["Max Abs Diff", f"{metrics['max_abs_diff']:.6e}", "-", "-"],
        ["Mean Diff", f"{metrics['mean_diff']:.6e}", "-", "-"],
        ["Std Diff", f"{metrics['std_diff']:.6e}", "-", "-"],
    ]
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    print(f"\n{'=' * 80}")
    if comparison["overall_match"]:
        print("✅ OVERALL: Methods MATCH within tolerances")
    else:
        print("❌ OVERALL: Methods DO NOT MATCH within tolerances")
    print("=" * 80)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_methods(
    methods: List[Dict],
    input_list: List[torch.Tensor],
    grid_config: Dict,
    sample_grid_z: Tuple[float, float, float],
    input_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    downsample: int,
    device: str = "cuda",
    depth_distribution: str = "laplace",
    depth_weight_threshold: float = 0.0,
    seed: int = 42,
) -> Dict:
    """Evaluate all methods and compare outputs against baseline."""
    if not methods:
        raise ValueError("No methods specified in config!")
    
    set_seed(seed)
    
    baseline_name = methods[0]["name"]
    baseline_output = None
    
    results = {}
    
    print("\n" + "=" * 80)
    print("Evaluating Output Differences")
    print("=" * 80)
    print(f"Baseline: {baseline_name}")
    print(f"Seed: {seed}\n")
    
    for method_config in methods:
        method_name = method_config["name"]
        print(f"Processing {method_name}...")
        
        try:
            set_seed(seed)
            transformer = create_view_transformer(
                grid_config=grid_config,
                sample_grid_z=sample_grid_z,
                input_size=input_size,
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
                fuse_projection=method_config.get("fuse_projection", False),
                use_bev_pool=method_config.get("use_bev_pool", False),
                use_shared_memory=method_config.get("use_shared_memory", False),
                depth_regression=method_config.get("depth_regression", True),
                use_bilinear=method_config.get("use_bilinear", True),
                fuse_bilinear=method_config.get("fuse_bilinear", False),
                device=device,
                depth_distribution=method_config.get("depth_distribution", depth_distribution),
                optimize_z_precompute=method_config.get("optimize_z_precompute", True),
                use_warp_kernel=method_config.get("use_warp_kernel", False),
                use_vectorized_load=method_config.get("use_vectorized_load", False),
                depth_weight_threshold=method_config.get("depth_weight_threshold", depth_weight_threshold),
            )
            
            with torch.no_grad():
                bev_feat, _ = transformer(input=input_list)
     
            if method_name == baseline_name:
                baseline_output = bev_feat
                results[method_name] = {
                    "output": bev_feat,
                    "transformer": transformer,
                    "metrics": None,
                }
                print(f"  ✓ Baseline output captured: {bev_feat.shape}")
            else:
                if baseline_output is None:
                    raise ValueError(f"Baseline ({baseline_name}) must be processed first!")
                
                if bev_feat.shape != baseline_output.shape:
                    print(f"  ⚠ Warning: Shape mismatch! Baseline: {baseline_output.shape}, {method_name}: {bev_feat.shape}")
                    results[method_name] = {
                        "output": bev_feat,
                        "transformer": transformer,
                        "metrics": {"error": "Shape mismatch"},
                    }
                else:
                    metrics = compute_difference_metrics(baseline_output, bev_feat)
                    results[method_name] = {
                        "output": bev_feat,
                        "transformer": transformer,
                        "metrics": metrics,
                    }
                    print(f"  ✓ Output captured: {bev_feat.shape}")
                    print(f"    MSE: {metrics['mse']:.6e}, MAE: {metrics['mae']:.6e}")
                    print(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        
        except Exception as e:
            print(f"  ✗ Error processing {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {
                "output": None,
                "transformer": None,
                "metrics": {"error": str(e)},
            }
    
    return results, baseline_name


def print_results_table(results: Dict, baseline_name: str):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("Output Difference Results")
    print("=" * 80)
    print(f"Baseline: {baseline_name}\n")
    
    table_data = []
    headers = ["Method", "MSE", "MAE", "Max Abs Diff", "Mean Diff", "Std Diff", 
               "Relative Error", "Cosine Sim"]
    
    for method_name, result in results.items():
        if method_name == baseline_name:
            table_data.append([method_name, "BASELINE", "-", "-", "-", "-", "-", "-"])
        elif result["metrics"] is None or "error" in result["metrics"]:
            error_msg = result["metrics"].get("error", "Unknown error") if result["metrics"] else "No metrics"
            table_data.append([method_name, f"ERROR: {error_msg}", "-", "-", "-", "-", "-", "-"])
        else:
            metrics = result["metrics"]
            table_data.append([
                method_name,
                f"{metrics['mse']:.6e}",
                f"{metrics['mae']:.6e}",
                f"{metrics['max_abs_diff']:.6e}",
                f"{metrics['mean_diff']:.6e}",
                f"{metrics['std_diff']:.6e}",
                f"{metrics['relative_error']:.6e}",
                f"{metrics['cosine_similarity']:.6f}",
            ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_results_json(results: Dict, baseline_name: str, output_path: str):
    """Save results to JSON file."""
    json_results = {
        "baseline": baseline_name,
        "methods": {}
    }
    
    for method_name, result in results.items():
        if method_name == baseline_name:
            json_results["methods"][method_name] = {
                "is_baseline": True,
                "output_shape": list(result["output"].shape) if result["output"] is not None else None,
            }
        else:
            json_results["methods"][method_name] = {
                "is_baseline": False,
                "output_shape": list(result["output"].shape) if result["output"] is not None else None,
                "metrics": result["metrics"],
            }
    
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function to evaluate output differences."""
    print("=" * 80)
    print("Output Difference Evaluation")
    print("=" * 80)
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    if not cfg.methods:
        raise ValueError("No methods specified in config.methods!")
    
    device = cfg.device
    batch_size = cfg.batch_size
    num_cameras = cfg.num_cameras
    in_channels = cfg.in_channels
    feature_h = cfg.feature_h
    feature_w = cfg.feature_w
    input_size = (cfg.input_h, cfg.input_w)
    downsample = cfg.downsample
    out_channels = cfg.out_channels
    
    grid_config = OmegaConf.to_container(cfg.grid_config, resolve=True)
    z_min = cfg.z_min
    z_max = cfg.z_max
    z_range = z_max - z_min
    num_height_bins = cfg.num_height_bins
    z_resolution = z_range / float(num_height_bins)
    sample_grid_z = (z_min, z_max, z_resolution)
    
    depth_distribution = cfg.depth_distribution
    depth_weight_threshold = cfg.depth_weight_threshold
    seed = cfg.get("seed", 42)
    
    set_seed(seed)
    
    calib_params = None
    if cfg.load_calib is not None:
        calib_params = load_calibration_params(cfg.load_calib, device=device)
    
    input_list, _ = create_dummy_input(
        batch_size=batch_size,
        num_cameras=num_cameras,
        in_channels=in_channels,
        feature_h=feature_h,
        feature_w=feature_w,
        device=device,
        calib_params=calib_params,
    )
    
    methods = OmegaConf.to_container(cfg.methods, resolve=True)
    
    results, baseline_name = evaluate_methods(
        methods=methods,
        input_list=input_list,
        grid_config=grid_config,
        sample_grid_z=sample_grid_z,
        input_size=input_size,
        in_channels=in_channels,
        out_channels=out_channels,
        downsample=downsample,
        device=device,
        depth_distribution=depth_distribution,
        depth_weight_threshold=depth_weight_threshold,
        seed=seed,
    )
    
    print_results_table(results, baseline_name)
    
    if len(methods) == 2:
        method1_name = methods[0]["name"]
        method2_name = methods[1]["name"]
        
        if method1_name in results and method2_name in results:
            method1_output = results[method1_name]["output"]
            method2_output = results[method2_name]["output"]
            
            if method1_output is not None and method2_output is not None:
                comparison = compare_two_methods(
                    method1_output,
                    method2_output,
                    method1_name,
                    method2_name,
                    tolerance_mse=cfg.get("tolerance_mse", 1e-6),
                    tolerance_mae=cfg.get("tolerance_mae", 1e-6),
                    tolerance_relative=cfg.get("tolerance_relative", 1e-5),
                )
                print_two_method_comparison(comparison)
    
    if cfg.output_json is not None:
        save_results_json(results, baseline_name, cfg.output_json)


if __name__ == "__main__":
    main()
