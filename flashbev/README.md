# FlashBEVPool

Fast BEV Pooling with Fused CUDA Kernels for 3D Object Detection.

## Overview

FlashBEVPool is a high-performance implementation of Bird's Eye View (BEV) pooling for 3D object detection, featuring:

- **Fused CUDA kernels** that combine projection, depth weighting, and interpolation
- **Support for Gaussian and Laplace depth distributions**
- **Optimized memory access patterns** with shared memory and vectorized loads
- **Warp-level optimizations** for improved GPU utilization

## Installation

```bash
cd flashbev
pip install -e .
```

## Usage

```python
import torch
from flashbevpool import flash_bevpool

# Example usage
depth_params = torch.randn(B, N, H, W, 2)  # [mean, sigma]
image_feats = torch.randn(B, N, H, W, C)
projection_matrices = torch.randn(B, N, 4, 4)
feature_size = torch.tensor([[H, W]] * B * N, dtype=torch.int32)
image_size = torch.tensor([H, W], dtype=torch.int32)
roi_range = torch.tensor([min_x, max_x, min_y, max_y, min_z, max_z], dtype=torch.float32)
grid_size = torch.tensor([grid_x, grid_y, grid_z], dtype=torch.int32)

bev_feat = flash_bevpool(
    image_feats=image_feats,
    depth_params=depth_params,
    projection_matrices=projection_matrices,
    feature_size=feature_size,
    image_size=image_size,
    roi_range=roi_range,
    grid_size=grid_size,
    depth_distribution=0,  # 0=gaussian, 1=laplace
    use_shared_memory=True,
    optimize_z_precompute=True,
    use_warp_kernel=False,
    use_vectorized_load=False,
    epsilon=1e-6,
    depth_weight_threshold=1e-6,
)
```

## API Reference

### `flash_bevpool`

Main function for BEV pooling.

**Arguments:**
- `image_feats` (Tensor): Image features [B,N,H,W,C]
- `depth_params` (Tensor): Depth parameters [B,N,H,W,2] - [mean, sigma] or [mean, b]
- `projection_matrices` (Tensor): Camera projection matrices [B,N,4,4]
- `feature_size` (Tensor): Feature dimensions [B,N,2]
- `image_size` (Tensor): Image dimensions [2] [height, width]
- `roi_range` (Tensor): ROI bounds [6] [min_x,max_x,min_y,max_y,min_z,max_z]
- `grid_size` (Tensor): Grid dimensions [3] [grid_x,grid_y,grid_z]
- `depth_distribution` (int): 0=gaussian, 1=laplace (default: 0)
- `use_shared_memory` (bool): Use shared memory for projection matrices (default: True)
- `optimize_z_precompute` (bool): Optimize z-independent projection (default: True)
- `use_warp_kernel` (bool): Use warp-optimized kernel (default: False)
- `use_vectorized_load` (bool): Use vectorized loads (requires use_warp_kernel=True) (default: False)
- `epsilon` (float): Numerical stability constant (default: 1e-6)
- `depth_weight_threshold` (float): Threshold for depth weight filtering (default: 1e-6)

**Returns:**
- `bev_feat` (Tensor): BEV features [B,X,Y,C]

## Benchmarking

A comprehensive benchmark script is included that supports both kernel-only and full system benchmarking:

### Kernel-Only Mode (No mmdet3d dependency)

```bash
# Benchmark FlashBEVPool kernel directly:
python benchmark_view_transform.py --kernel-only --num-height-bins 8,10,16,20,40
python benchmark_view_transform.py --kernel-only --load-calib calib.json --num-cameras-list 1,2,3,4,5,6
```

### Full System Mode (Requires mmdet3d)

```bash
# Benchmark all SamplingVT methods:
python benchmark_view_transform.py --load-calib calib.json
python benchmark_view_transform.py --num-height-bins 8,10,16,20,40
```

**Note:** 
- `--kernel-only` mode benchmarks FlashBEV kernel directly, no mmdet3d dependency
- Full system mode requires `flashbevpool.SamplingVT` (or `mmdet3d.models.necks.view_transformer_back.SamplingVT`)

For more options:
```bash
python benchmark_view_transform.py --help
```

## License

Apache License 2.0
