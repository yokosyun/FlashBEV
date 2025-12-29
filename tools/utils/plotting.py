import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


def setup_output_directory() -> Path:
    """Create outputs directory if it doesn't exist."""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


def get_plot_paths(
    plot_output: str,
    default_plot_name: str,
    latency_plot_name: str,
    outputs_dir: Path,
) -> Tuple[str, str]:
    """Determine plot output paths."""
    if plot_output:
        if os.path.dirname(plot_output):
            plot_path = plot_output
        else:
            plot_path = str(outputs_dir / plot_output)
        latency_plot_path = plot_path.replace("memory", "latency")
    else:
        plot_path = str(outputs_dir / default_plot_name)
        latency_plot_path = str(outputs_dir / latency_plot_name)
    
    return plot_path, latency_plot_path


def plot_memory_vs_x(
    memory_data: Dict,
    x_axis_label: str,
    plot_title: str,
    plot_output: str,
    has_height_bins_exp: bool,
    has_depth_threshold_exp: bool,
):
    """Plot memory usage vs x-axis variable with error bars for multiple runs."""
    print(f"Generating memory plot: {plot_output}...")
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    has_multiple_runs = any("individual_runs" in data for data in memory_data.values())
    
    for idx, (method_name, data) in enumerate(memory_data.items()):
        if len(data[x_axis_label]) > 0:
            x_values = data[x_axis_label]
            memory = data["memory_mb"]
            sorted_pairs = sorted(zip(x_values, memory))
            x_sorted, memory_sorted = zip(*sorted_pairs)
            
            if has_multiple_runs and "memory_std" in data and len(data["memory_std"]) > 0:
                memory_std = data["memory_std"]
                sorted_std_pairs = sorted(zip(x_values, memory_std))
                _, memory_std_sorted = zip(*sorted_std_pairs)
                
                plt.errorbar(
                    x_sorted, memory_sorted, yerr=memory_std_sorted,
                    marker=markers[idx % len(markers)],
                    label=method_name,
                    linewidth=2,
                    markersize=8,
                    color=colors[idx % len(colors)],
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5
                )
                
                if "individual_runs" in data and len(data["individual_runs"]) == len(x_values):
                    for x_val_idx, x_val in enumerate(x_sorted):
                        if x_val_idx < len(data["individual_runs"]) and data["individual_runs"][x_val_idx]:
                            run_memory = [r["peak_memory_allocated_mb"] for r in data["individual_runs"][x_val_idx]]
                            plt.scatter(
                                [x_val] * len(run_memory), run_memory,
                                alpha=0.3,
                                s=20,
                                color=colors[idx % len(colors)],
                                edgecolors='none',
                                zorder=0
                            )
            else:
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


def plot_latency_vs_x(
    memory_data: Dict,
    x_axis_label: str,
    latency_plot_title: str,
    latency_plot_output: str,
    has_height_bins_exp: bool,
    has_depth_threshold_exp: bool,
):
    """Plot latency vs x-axis variable with error bars for multiple runs."""
    print(f"Generating latency plot: {latency_plot_output}...")
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    has_multiple_runs = any("individual_runs" in data for data in memory_data.values())
    
    for idx, (method_name, data) in enumerate(memory_data.items()):
        if len(data[x_axis_label]) > 0:
            x_values = data[x_axis_label]
            latency = data["latency_ms"]
            sorted_pairs = sorted(zip(x_values, latency))
            x_sorted, latency_sorted = zip(*sorted_pairs)
            
            if has_multiple_runs and "latency_std" in data and len(data["latency_std"]) > 0:
                latency_std = data["latency_std"]
                sorted_std_pairs = sorted(zip(x_values, latency_std))
                _, latency_std_sorted = zip(*sorted_std_pairs)
                
                plt.errorbar(
                    x_sorted, latency_sorted, yerr=latency_std_sorted,
                    marker=markers[idx % len(markers)],
                    label=method_name,
                    linewidth=2,
                    markersize=8,
                    color=colors[idx % len(colors)],
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5
                )
                
                if "individual_runs" in data and len(data["individual_runs"]) == len(x_values):
                    for x_val_idx, x_val in enumerate(x_sorted):
                        if x_val_idx < len(data["individual_runs"]) and data["individual_runs"][x_val_idx]:
                            run_latency = [r["latency_mean_ms"] for r in data["individual_runs"][x_val_idx]]
                            plt.scatter(
                                [x_val] * len(run_latency), run_latency,
                                alpha=0.3,
                                s=20,
                                color=colors[idx % len(colors)],
                                edgecolors='none',
                                zorder=0
                            )
            else:
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

