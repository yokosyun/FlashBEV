"""Configuration constants for benchmarking."""

DEFAULT_GRID_CONFIG = {
    "x": [-51.2, 51.2, 0.8],
    "y": [-51.2, 51.2, 0.4],
    "z": [-5.0, 3.0, 8.0],
    "depth": [1.0, 60.0, 1.0],
}

Z_MIN, Z_MAX = -5.0, 3.0
Z_RANGE = Z_MAX - Z_MIN

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

