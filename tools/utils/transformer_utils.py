from typing import Dict, Tuple

from flashbevpool import SamplingVT


def create_view_transformer(
    grid_config: Dict,
    sample_grid_z: Tuple[float, float, float],
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
) -> SamplingVT:
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

