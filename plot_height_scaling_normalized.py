"""
Reproducible plotting script for ECCV paper.
Generates "Normalized Latency vs Num Height Bins" figure across multiple GPUs.

This script automatically:
1. Scans outputs/ for {GPU}.json files (e.g., A6000.json, A4000.json)
2. Extracts latencies for each GPU
3. Generates publication-ready normalized latency plots

Usage:
1. Run benchmarks on each GPU: python tools/benchmark.py output_json=outputs/A6000.json num_height_bins=[4,8,12,16]
2. Run: python plot_height_scaling_normalized.py
3. Output: height_scaling_normalized.pdf and height_scaling_normalized.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Height bins to analyze
Z = [4, 8, 12, 16]

# Method name mappings (in order of preference)
METHOD_MAP = {
    "dense": ["Dense PyTorch Sampling-VT", "Sampling-VT"],
    "flash": ["FlashBEV(ours)", "FlashBEV"],
}

def extract_gpus_from_json_files(outputs_dir: Path):
    """Auto-detect GPU names from {GPU}.json filenames."""
    gpus = []
    
    for json_file in outputs_dir.glob("*.json"):
        name = json_file.stem
        if not name.startswith("benchmark_results_") and name not in ["benchmark_results"]:
            gpus.append(name)
    
    pattern = re.compile(r"benchmark_results_(.+?)\.json")
    for json_file in outputs_dir.glob("benchmark_results_*.json"):
        match = pattern.match(json_file.name)
        if match:
            gpu_name = match.group(1)
            if gpu_name not in gpus:
                gpus.append(gpu_name)
    
    return sorted(gpus)

def extract_latencies_from_json(json_path: Path, gpu_name: str):
    """Extract latencies from a benchmark results JSON file."""
    if not json_path.exists():
        print(f"Warning: {json_path} not found, skipping {gpu_name}")
        return None
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None
    
    results = data.get("results", [])
    latencies = {"dense": [None] * len(Z), "flash": [None] * len(Z)}
    
    for result in results:
        method_name = result.get("method", "")
        num_bins = result.get("num_height_bins")
        latency = result.get("latency_mean_ms")
        
        if num_bins not in Z:
            continue
        
        z_idx = Z.index(num_bins)
        
        for method_key, method_names in METHOD_MAP.items():
            if any(name in method_name for name in method_names):
                if latencies[method_key][z_idx] is None:
                    latencies[method_key][z_idx] = latency
                break
    
    return latencies

def load_data_from_json_files(outputs_dir: Path = Path("outputs")):
    """Load latency data from all {GPU}.json files."""
    gpus = extract_gpus_from_json_files(outputs_dir)
    
    if not gpus:
        print("No {GPU}.json files found in outputs/")
        print("Please run benchmarks first:")
        print("  python tools/benchmark.py output_json=outputs/A6000.json num_height_bins=[4,8,12,16]")
        return None, {}
    
    print(f"Found GPUs: {', '.join(gpus)}")
    print("=" * 80)
    
    lat_ms = {}
    for gpu in gpus:
        json_path = outputs_dir / f"{gpu}.json"
        if not json_path.exists():
            json_path = outputs_dir / f"benchmark_results_{gpu}.json"
        latencies = extract_latencies_from_json(json_path, gpu)
        
        if latencies:
            lat_ms[gpu] = latencies
            print(f"\n{gpu}:")
            print(f"  Dense: {latencies['dense']}")
            print(f"  Flash: {latencies['flash']}")
        else:
            print(f"\n{gpu}: No data found")
    
    print("=" * 80)
    print()
    
    return gpus, lat_ms

def normalize_latencies(gpus, lat_ms):
    """Normalize both methods using Dense PyTorch smallest bin as baseline."""
    Z0 = Z[0]
    norm_lat = {}
    for gpu in gpus:
        norm_lat[gpu] = {}
        dense_base = lat_ms[gpu]["dense"][0]
        
        if dense_base is None or dense_base == 0:
            print(f"Warning: {gpu} Dense has invalid base value, cannot normalize")
            norm_lat[gpu]["dense"] = [None] * len(Z)
            norm_lat[gpu]["flash"] = [None] * len(Z)
            continue
        
        for method in ["dense", "flash"]:
            norm_lat[gpu][method] = []
            for val in lat_ms[gpu][method]:
                if val is None:
                    norm_lat[gpu][method].append(None)
                else:
                    norm_lat[gpu][method].append(val / dense_base)
    
    return norm_lat

def calculate_speedups(gpus, lat_ms):
    """Calculate speedups for each GPU at different Z values."""
    speedups = {}
    for gpu in gpus:
        speedups[gpu] = {}
        for i, z in enumerate(Z):
            dense_val = lat_ms[gpu]["dense"][i]
            flash_val = lat_ms[gpu]["flash"][i]
            if dense_val is not None and flash_val is not None and flash_val > 0:
                speedups[gpu][z] = dense_val / flash_val
            else:
                speedups[gpu][z] = None
    return speedups

def print_tables(gpus, lat_ms, norm_lat, speedups):
    """Print summary table with speedups."""
    print("=" * 100)
    print("Summary Table: Latency and Speedup vs Num Height Bins")
    print("=" * 100)
    
    for gpu in gpus:
        print(f"\n{gpu}:")
        header = f"{'Z':<6} {'Dense (ms)':<15} {'Flash (ms)':<15} {'Speedup':<10} {'Dense (norm)':<15} {'Flash (norm)':<15}"
        print(header)
        print("-" * 100)
        
        for i, z in enumerate(Z):
            dense_ms = lat_ms[gpu]["dense"][i]
            flash_ms = lat_ms[gpu]["flash"][i]
            speedup = speedups[gpu][z]
            dense_norm = norm_lat[gpu]["dense"][i]
            flash_norm = norm_lat[gpu]["flash"][i]
            
            dense_ms_str = f"{dense_ms:.3f}" if dense_ms is not None else "OOM"
            flash_ms_str = f"{flash_ms:.3f}" if flash_ms is not None else "OOM"
            speedup_str = f"×{speedup:.2f}" if speedup is not None else "—"
            dense_norm_str = f"{dense_norm:.3f}" if dense_norm is not None else "OOM"
            flash_norm_str = f"{flash_norm:.3f}" if flash_norm is not None else "OOM"
            
            print(f"{z:<6} {dense_ms_str:<15} {flash_ms_str:<15} {speedup_str:<10} {dense_norm_str:<15} {flash_norm_str:<15}")
    
    print("=" * 100)
    print()

def create_plots(gpus, lat_ms, norm_lat, speedups):
    """Create publication-ready normalized latency plots."""
    n_gpus = len(gpus)
    n_cols = n_gpus  # One column per GPU (arrange in a row)
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_gpus, 4))
    if n_gpus == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Publication-ready styling
    colors = {"dense": "#d62728", "flash": "#2ca02c"}
    markers = {"dense": "o", "flash": "s"}
    labels = {"dense": "Dense PyTorch", "flash": "FlashBEV"}
    linewidth = 2.5
    markersize = 9
    
    for idx, gpu in enumerate(gpus):
        ax = axes[idx]
        
        # Get valid data points
        dense_vals = norm_lat[gpu]["dense"]
        flash_vals = norm_lat[gpu]["flash"]
        dense_z = [z for i, z in enumerate(Z) if dense_vals[i] is not None]
        dense_norm = [v for v in dense_vals if v is not None]
        flash_z = [z for i, z in enumerate(Z) if flash_vals[i] is not None]
        flash_norm = [v for v in flash_vals if v is not None]
        
        # Plot Dense PyTorch
        if dense_z and dense_norm:
            ax.plot(
                dense_z, dense_norm,
                color=colors["dense"],
                marker=markers["dense"],
                label=labels["dense"],
                linewidth=linewidth,
                markersize=markersize,
                zorder=3,
                markerfacecolor=colors["dense"],
                markeredgecolor=colors["dense"],
                markeredgewidth=1.5
            )
        
        # Plot FlashBEV
        if flash_z and flash_norm:
            ax.plot(
                flash_z, flash_norm,
                color=colors["flash"],
                marker=markers["flash"],
                label=labels["flash"],
                linewidth=linewidth,
                markersize=markersize,
                zorder=3,
                markerfacecolor=colors["flash"],
                markeredgecolor=colors["flash"],
                markeredgewidth=1.5
            )
            
            # Add speedup annotations for all Z values where both methods are valid
            for z in Z:
                if z in speedups[gpu] and speedups[gpu][z] is not None:
                    z_idx = flash_z.index(z) if z in flash_z else None
                    if z_idx is not None:
                        z_y = flash_norm[z_idx]
                        speedup_val = speedups[gpu][z]
                        ax.annotate(
                            f"×{speedup_val:.1f}",
                            xy=(z, z_y),
                            xytext=(8, -12),
                            textcoords="offset points",
                            fontsize=10,
                            color=colors["flash"],
                            fontweight="bold",
                            bbox=dict(boxstyle="round", facecolor="white", edgecolor=colors["flash"], linewidth=1.0),
                            ha="left"
                        )
        
        # Subplot styling
        ax.set_xlabel("Num Height Bins (Z)", fontsize=13)
        ax.set_ylabel("Latency (normalized by Dense@Z=4)", fontsize=13)
        ax.set_title(gpu, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.set_xticks(Z)
        ax.tick_params(axis="both", labelsize=11)
        
        # Set y-axis with linear scale
        all_vals = []
        if dense_norm:
            all_vals.extend([v for v in dense_norm if v is not None])
        if flash_norm:
            all_vals.extend([v for v in flash_norm if v is not None])
        
        if all_vals:
            y_min = 0
            y_max = max(all_vals) * 1.15
            y_max = max(y_max, 1.2)
        else:
            y_min = 0
            y_max = 2.0
        
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.6, zorder=1)
    
    # Remove extra subplots
    for idx in range(n_gpus, len(axes)):
        fig.delaxes(axes[idx])
    
    # Global legend (only once)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels_legend, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        ncol=2,
        fontsize=12,
    )
    
    # Main title
    fig.suptitle(
        "Height-Resolution Scaling Across GPUs",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save as both PDF and PNG
    output_pdf = "height_scaling_normalized.pdf"
    output_png = "height_scaling_normalized.png"
    
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")

def main():
    """Main function."""
    # Load data from JSON files
    gpus, lat_ms = load_data_from_json_files()
    
    if not gpus or not lat_ms:
        return
    
    # Normalize latencies
    norm_lat = normalize_latencies(gpus, lat_ms)
    
    # Calculate speedups
    speedups = calculate_speedups(gpus, lat_ms)
    
    # Print summary tables
    print_tables(gpus, lat_ms, norm_lat, speedups)
    
    # Create and save plots
    create_plots(gpus, lat_ms, norm_lat, speedups)

if __name__ == "__main__":
    main()
