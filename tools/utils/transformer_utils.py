from typing import Dict, Tuple

from flashbevpool import SamplingVT


def create_view_transformer(
    grid_config: Dict,
    input_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    downsample: int,
    fuse_projection: bool,
    use_bev_pool: bool,
    use_shared_memory: bool,
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
        use_bev_pool=use_bev_pool,
        use_shared_memory=use_shared_memory,
        fuse_bilinear=fuse_bilinear,
        depth_distribution=depth_distribution,
        optimize_z_precompute=optimize_z_precompute,
        use_warp_kernel=use_warp_kernel,
        use_vectorized_load=use_vectorized_load,
        depth_weight_threshold=depth_weight_threshold,
    )
    
    transformer = transformer.to(device)
    transformer.eval()
    
    return transformer

