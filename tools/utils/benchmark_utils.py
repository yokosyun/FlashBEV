# Copyright (c) Shunsuke Yokokawa. All rights reserved.

import time
from typing import Dict, List

import numpy as np
import torch

from flashbevpool import SamplingVT, flash_bevpool


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
) -> Dict[str, float]:
    """Benchmark FlashBEVPool kernel directly (no mmdet3d dependency)."""
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
) -> Dict[str, float]:
    """Benchmark a view transform method."""
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

