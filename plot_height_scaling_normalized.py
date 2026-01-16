"""
Reproducible plotting script for ECCV paper.
Generates "Normalized Latency vs Num Height Bins" figure across multiple GPUs.

This script automatically:
1. Scans outputs/ for {GPU}.json files (e.g., A6000.json, A4000.json)
2. Extracts latencies for each GPU
3. Generates normalized latency plots

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
    
    # Look for JSON files that are not benchmark_results_*.json (for backward compat)
    # Prefer simple {GPU}.json format
    for json_file in outputs_dir.glob("*.json"):
        name = json_file.stem  # filename without .json extension
        # Skip benchmark_results files and other special files
        if not name.startswith("benchmark_results_") and name not in ["benchmark_results"]:
            gpus.append(name)
    
    # Also check for old format for backward compatibility
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
    
    # Group results by num_height_bins
    for result in results:
        method_name = result.get("method", "")
        num_bins = result.get("num_height_bins")
        latency = result.get("latency_mean_ms")
        
        if num_bins not in Z:
            continue
        
        z_idx = Z.index(num_bins)
        
        # Match method
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
        # Try simple format first, then fall back to old format
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
    # norm[z] = lat[z] / dense[Z0] where Z0 is the first element
    # Both dense and flash use dense's Z=4 as the baseline
    Z0 = Z[0]
    norm_lat = {}
    for gpu in gpus:
        norm_lat[gpu] = {}
        # Use dense's Z=4 as baseline for both methods
        dense_base = lat_ms[gpu]["dense"][0]
        
        if dense_base is None or dense_base == 0:
            print(f"Warning: {gpu} Dense has invalid base value, cannot normalize")
            norm_lat[gpu]["dense"] = [None] * len(Z)
            norm_lat[gpu]["flash"] = [None] * len(Z)
            continue
        
        # Normalize both methods by dense baseline
        for method in ["dense", "flash"]:
            norm_lat[gpu][method] = []
            for val in lat_ms[gpu][method]:
                if val is None:
                    norm_lat[gpu][method].append(None)
                else:
                    norm_lat[gpu][method].append(val / dense_base)
    
    return norm_lat

def print_tables(gpus, norm_lat):
    """Print normalized latency tables for each GPU."""
    print("=" * 80)
    print("Normalized Latency vs Num Height Bins")
    print("=" * 80)
    for gpu in gpus:
        print(f"\n{gpu}:")
        print(f"{'Z':<8} {'Dense (norm)':<15} {'FlashBEV (norm)':<15}")
        print("-" * 40)
        for i, z in enumerate(Z):
            dense_val = norm_lat[gpu]["dense"][i]
            flash_val = norm_lat[gpu]["flash"][i]
            dense_str = f"{dense_val:.3f}" if dense_val is not None else "OOM"
            flash_str = f"{flash_val:.3f}" if flash_val is not None else "OOM"
            print(f"{z:<8} {dense_str:<15} {flash_str:<15}")
    print("=" * 80)
    print()

def create_plots(gpus, norm_lat):
    """Create normalized latency plots for all GPUs."""
    n_gpus = len(gpus)
    n_cols = 2
    n_rows = (n_gpus + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    if n_gpus == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Plotting parameters
    colors = {"dense": "#d62728", "flash": "#2ca02c"}
    markers = {"dense": "o", "flash": "s"}
    labels = {"dense": "Dense PyTorch", "flash": "FlashBEV"}
    linewidth = 2.5
    markersize = 8
    
    # Plot each GPU
    for idx, gpu in enumerate(gpus):
        ax = axes[idx]
        
        # Plot Dense
        dense_vals = norm_lat[gpu]["dense"]
        dense_z = [z for i, z in enumerate(Z) if dense_vals[i] is not None]
        dense_norm = [v for v in dense_vals if v is not None]
        
        if dense_z and dense_norm:
            ax.plot(
                dense_z, dense_norm,
                color=colors["dense"],
                marker=markers["dense"],
                label=labels["dense"],
                linewidth=linewidth,
                markersize=markersize,
                zorder=3
            )
            # Annotate OOM if last point is not the last Z
            if len(dense_z) < len(Z):
                last_z = dense_z[-1]
                last_val = dense_norm[-1]
                ax.annotate(
                    "OOM", xy=(last_z, last_val),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=colors["dense"],
                    fontweight="bold"
                )
        
        # Plot FlashBEV
        flash_vals = norm_lat[gpu]["flash"]
        flash_z = [z for i, z in enumerate(Z) if flash_vals[i] is not None]
        flash_norm = [v for v in flash_vals if v is not None]
        
        if flash_z and flash_norm:
            ax.plot(
                flash_z, flash_norm,
                color=colors["flash"],
                marker=markers["flash"],
                label=labels["flash"],
                linewidth=linewidth,
                markersize=markersize,
                zorder=3
            )
            # Annotate OOM if last point is not the last Z
            if len(flash_z) < len(Z):
                last_z = flash_z[-1]
                last_val = flash_norm[-1]
                ax.annotate(
                    "OOM", xy=(last_z, last_val),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=colors["flash"],
                    fontweight="bold"
                )
        
        # Subplot styling
        ax.set_xlabel("Num Height Bins", fontsize=12)
        ax.set_ylabel("Normalized Latency (× over Dense Z=4)", fontsize=12)
        ax.set_title(gpu, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(Z)
        
        # Set y-axis to include both methods clearly on linear scale
        # This makes it easier to see speedup ratios directly
        all_vals = []
        if dense_norm:
            all_vals.extend([v for v in dense_norm if v is not None])
        if flash_norm:
            all_vals.extend([v for v in flash_norm if v is not None])
        
        if all_vals:
            y_min = 0  # Start from 0 to clearly show FlashBEV values
            y_max = max(all_vals) * 1.15
            # Ensure y_max is at least 1.2 to see the baseline
            y_max = max(y_max, 1.2)
        else:
            y_min = 0
            y_max = 2.0
        
        ax.set_ylim(y_min, y_max)
        ax.set_yscale("linear")  # Use linear scale for easier speedup reading
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)
    
    # Remove extra subplots if any
    for idx in range(n_gpus, len(axes)):
        fig.delaxes(axes[idx])
    
    # Global legend (only once)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend,
        loc="upper center",
        ncol=2,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.98)
    )
    
    # Main title
    fig.suptitle(
        "Normalized Latency vs Height Bins (per GPU)",
        fontsize=15,
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
    
    # Print tables
    print_tables(gpus, norm_lat)
    
    # Create and save plots
    create_plots(gpus, norm_lat)

if __name__ == "__main__":
    main()

