from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


def setup_output_directory() -> Path:
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


def get_plot_paths(
    plot_output: str,
    default_plot_name: str,
    latency_plot_name: str,
    outputs_dir: Path,
) -> Tuple[str, str]:
    if plot_output:
        plot_path = plot_output if "/" in plot_output or "\\" in plot_output else str(outputs_dir / plot_output)
        latency_plot_path = plot_path.replace("memory", "latency")
    else:
        plot_path = str(outputs_dir / default_plot_name)
        latency_plot_path = str(outputs_dir / latency_plot_name)
    
    return plot_path, latency_plot_path


def _plot_vs_x(
    memory_data: Dict,
    x_axis_label: str,
    plot_title: str,
    plot_output: str,
    y_data_key: str,
    ylabel: str,
    has_height_bins_exp: bool,
    has_depth_threshold_exp: bool,
):
    print(f"Generating plot: {plot_output}...")
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (method_name, data) in enumerate(memory_data.items()):
        if len(data[x_axis_label]) > 0:
            x_values = data[x_axis_label]
            y_values = data[y_data_key]
            sorted_pairs = sorted(zip(x_values, y_values))
            x_sorted, y_sorted = zip(*sorted_pairs)
            
            plt.plot(
                x_sorted, y_sorted,
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
    plt.ylabel(ylabel, fontsize=12)
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {plot_output}")
    print()


def plot_memory_vs_x(
    memory_data: Dict,
    x_axis_label: str,
    plot_title: str,
    plot_output: str,
    has_height_bins_exp: bool,
    has_depth_threshold_exp: bool,
):
    _plot_vs_x(
        memory_data, x_axis_label, plot_title, plot_output,
        "memory_mb", "Peak Memory Allocated (MB)",
        has_height_bins_exp, has_depth_threshold_exp
    )


def plot_latency_vs_x(
    memory_data: Dict,
    x_axis_label: str,
    latency_plot_title: str,
    latency_plot_output: str,
    has_height_bins_exp: bool,
    has_depth_threshold_exp: bool,
):
    _plot_vs_x(
        memory_data, x_axis_label, latency_plot_title, latency_plot_output,
        "latency_ms", "Latency (ms)",
        has_height_bins_exp, has_depth_threshold_exp
    )

