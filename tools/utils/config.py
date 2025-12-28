"""Configuration constants for benchmarking."""

DEFAULT_METHODS = [
    {
        "name": "FlashBEV(baseline)",
        "fuse_projection": True,
        "use_bev_pool": False,
        "use_shared_memory": False,
        "depth_regression": True,
        "use_bilinear": True,
        "optimize_z_precompute": False,
    },
    {
        "name": "FlashBEV(warp_kernel)",
        "fuse_projection": True,
        "use_bev_pool": False,
        "use_shared_memory": False,
        "depth_regression": True,
        "use_bilinear": True,
        "optimize_z_precompute": False,
        "use_warp_kernel": True,
        "use_vectorized_load": False,
    },
    {
        "name": "Sampling-VT(FullPytorch)",
        "fuse_projection": False,
        "use_bev_pool": False,
        "depth_regression": True,
        "use_bilinear": True,
    },
    {
        "name": "Sampling-VT(PillarPool)",
        "fuse_projection": False,
        "use_bev_pool": True,
        "depth_regression": True,
        "use_bilinear": True,
        "fuse_bilinear": True,
    },
]

