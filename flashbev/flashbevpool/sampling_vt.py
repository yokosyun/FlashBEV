# Copyright (c) Shunsuke Yokokawa. All rights reserved.

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

#TODO(yoshuang): remove this once we have a proper base module
try:
    from mmcv.runner import BaseModule
except ImportError:
    BaseModule = nn.Module

from flashbevpool import flash_bevpool

from .sampling_vt_ops import (
    sampling_vt_pillarpool_fused,
)


def create_projection_matrix(sensor2ego, ego2global, camera2imgs, post_rots, post_trans, bda):
    B, N, _, _ = camera2imgs.shape

    extrinsic_matrices = sensor2ego
    extrinsic_matrices = torch.inverse(extrinsic_matrices)
    intrinsic_matrices = torch.eye(4, device=camera2imgs.device).repeat(B, N, 1, 1)
    intrinsic_matrices[..., :3, :3] = torch.matmul(post_rots, camera2imgs)
    intrinsic_matrices[..., :3, 2] += post_trans
    
    projection_matrices = torch.matmul(intrinsic_matrices, extrinsic_matrices)
    projection_matrices = projection_matrices[:, :, :3, :]

    return projection_matrices


def sampling_vt_pytorch(
    image_uvd: torch.Tensor,
    features_pv: torch.Tensor,
    depths: torch.Tensor,
    B: int,
    N: int,
    image_size: Tuple[int, int],
    depth_weight_threshold: float,
    depth_distribution: str,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    image_uvd = image_uvd.flatten(0, 1)
    
    image_height, image_width = image_size
    feat_h, feat_w = features_pv.shape[-2:]

    image_u = image_uvd[..., 0]
    image_v = image_uvd[..., 1]
    image_d = image_uvd[..., 2]
    
    feat_u = image_u / image_width * feat_w
    feat_v = image_v / image_height * feat_h
    
    fov_masks = (
        (feat_u >= 0) & (feat_u < feat_w) &
        (feat_v >= 0) & (feat_v < feat_h) &
        (image_d > 0.0)
    )
    
    feat_u_normalized = feat_u / (feat_w - 1.0) * 2.0 - 1.0
    feat_v_normalized = feat_v / (feat_h - 1.0) * 2.0 - 1.0
    
    coords_uv = torch.stack([feat_u_normalized, feat_v_normalized], dim=-1)

    bev_grid_shape = coords_uv.shape[1:3]

    # Note: there was no clear difference in performance between the two approaches.
    if True:
        features_3d = torch.nn.functional.grid_sample(
            input=features_pv,
            grid=coords_uv.flatten(1,2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).unflatten(2, bev_grid_shape)

        depths_3d = torch.nn.functional.grid_sample(
            input=depths,
            grid=coords_uv.flatten(1,2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).unflatten(2, bev_grid_shape)
    else:
        samples_2d = torch.cat([features_pv, depths], dim=1)
        samples_3d = torch.nn.functional.grid_sample(
            input=samples_2d,
            grid=coords_uv.flatten(1,2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).unflatten(2, bev_grid_shape)
        features_3d = samples_3d[:, :features_pv.shape[1]]
        depths_3d = samples_3d[:, features_pv.shape[1]:]

    depth_mu = depths_3d[:, 0]
    depth_sigma = depths_3d[:, 1]

    z_score = (image_d - depth_mu) / (depth_sigma + epsilon)
    
    if depth_distribution.lower() == "laplace":
        depth_prob_3d = 0.5 * torch.exp(-torch.abs(z_score)) / (depth_sigma + epsilon)
    else:
        depth_prob_3d = torch.exp(-0.5 * z_score * z_score) / (depth_sigma + epsilon)

    valid_mask = depth_prob_3d >= depth_weight_threshold
    valid_mask = valid_mask & fov_masks

    features_3d = features_3d * depth_prob_3d.unsqueeze(1)
    features_3d = features_3d * valid_mask.unsqueeze(1)
    bev_feat = features_3d.unflatten(0, (B, N)).sum(dim=[1, -1])

    valid_counts = valid_mask.unflatten(0, (B, N)).sum(dim=[1, -1])
    bev_feat = bev_feat / torch.clamp(valid_counts, min=1.0).unsqueeze(1)

    bev_feat = bev_feat.movedim(-1, -2)
    
    return bev_feat


class SamplingVT(BaseModule):
    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        fuse_projection=True,
        use_bev_pool=True,
        use_shared_memory=False,
        fuse_bilinear=False,
        depth_distribution="laplace",
        optimize_z_precompute=True,
        use_warp_kernel=False,
        use_vectorized_load=False,
        depth_weight_threshold=0.0,
    ):
        super(SamplingVT, self).__init__()

        self.use_bev_pool = use_bev_pool
        self.index_dtype = torch.int64
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.image_size = input_size
        self.depth_range = grid_config['depth'][:2]

        self.use_shared_memory = use_shared_memory
        self.fuse_projection = fuse_projection
        self.fuse_bilinear = fuse_bilinear
        self.optimize_z_precompute = optimize_z_precompute
        self.use_warp_kernel = use_warp_kernel
        self.use_vectorized_load = use_vectorized_load
        self.depth_weight_threshold = depth_weight_threshold
        self.depth_distribution = depth_distribution.lower()
        if self.depth_distribution not in ["laplace", "gaussian"]:
            raise ValueError(f"depth_distribution must be 'laplace' or 'gaussian', got '{depth_distribution}'")

        self.context_network = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, padding=0)

        self.register_buffer(
            name="depth_bins",
            tensor=torch.arange(*grid_config['depth'], dtype=torch.float),
        )
        self.num_depth_bins = len(self.depth_bins)

        self.depth_network = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
            nn.ReLU()
        )

        self.register_buffer(
            name="roi_ranges",
            tensor=torch.as_tensor(
                [grid_config["x"][:2], grid_config["y"][:2], grid_config["z"][:2]]),
        )
        self.register_buffer(
            name="output_grid_size",
            tensor=torch.tensor([(grid_config["x"][1] - grid_config["x"][0]) / grid_config["x"][2],
                                (grid_config["y"][1] - grid_config["y"][0]) / grid_config["y"][2],
                                1], dtype=torch.int32),
        )

        self.register_buffer(
            name="sample_grid_size",
            tensor=torch.tensor([(cfg[1] - cfg[0]) / cfg[2]
                                    for cfg in [grid_config["x"], grid_config["y"], grid_config["z"]]], dtype=torch.int32),
        )
        if self.fuse_projection == False:
            self.register_buffer(
                name="voxel_size",
                tensor=torch.as_tensor([grid_config["x"][2], grid_config["y"][2], grid_config["z"][2]]),
            )
            coords_3d = self._calculate_coords_3d(device="cpu")
            self.register_buffer(
                name="coords_3d",
                tensor=coords_3d,
            )

            round_fn = {
                "floor": torch.floor,
                "ceil": torch.ceil,
                "round": torch.round,
            }
            rounding_type = "floor"
            self.round_fn = round_fn[rounding_type]

    def compute_depth_weight(self, z_score, depth_sigma, epsilon=1e-4):
        """Compute depth weight/probability using the configured distribution."""
        if self.depth_distribution == "laplace":
            return 0.5 * torch.exp(-torch.abs(z_score)) / (depth_sigma + epsilon)
        else:
            return torch.exp(-0.5 * z_score * z_score) / (depth_sigma + epsilon)

    def forward_depth(self, x):
        depth_params = self.depth_network(x)
        depth_params[:, 1] += 0.1
        return depth_params

    def forward(self, input, images=None, img_meta=None, lidar_cloud=None, depth_from_lidar=None):
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        features_pv = self.context_network(x)
        depths = self.forward_depth(x)
        bev_features = self.sampling_vt(input, features_pv, depths)

        return bev_features, depths

    def sampling_vt(self, input, features_pv, depths):
        B, N, _, _ = input[3].shape

        projection_matrices = create_projection_matrix(*input[1:7])

        if self.fuse_projection:
            feature_size = torch.tensor([features_pv.shape[-2], features_pv.shape[-1]],
                                        device=features_pv.device, dtype=torch.int32)
            feature_size = feature_size.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            
            image_size = torch.tensor([self.image_size[0], self.image_size[1]],
                                        device=features_pv.device, dtype=torch.int32)

            depth_distribution_int = 1 if self.depth_distribution == "laplace" else 0

            bev_feat = flash_bevpool(
                image_feats=features_pv.movedim(-3, -1).unflatten(0, (B, N)),
                depth_params=depths.movedim(-3, -1).unflatten(0, (B, N)),
                projection_matrices=projection_matrices,
                depth_distribution=depth_distribution_int,
                use_shared_memory=self.use_shared_memory,
                optimize_z_precompute=self.optimize_z_precompute,
                use_warp_kernel=self.use_warp_kernel,
                use_vectorized_load=self.use_vectorized_load,
                feature_size=feature_size,
                image_size=image_size,
                roi_range=self.roi_ranges.flatten(),
                grid_size=self.sample_grid_size,
                epsilon=1e-6,
                depth_weight_threshold=self.depth_weight_threshold,
            )
            
            bev_feat = bev_feat.movedim(-1, -3)
        else:
            coords_3d = self.coords_3d
            coords_3d = coords_3d.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1, -1)

            image_uvd = project_coords(
                coords_xyz=coords_3d,
                projection_matrices=projection_matrices,
            )

            if self.use_bev_pool:
                bev_feat = self._sampling_vt_pillarpool_fused(coords_3d, image_uvd, features_pv.unflatten(0, (B, N)), depths.movedim(-3, -1).unflatten(0, (B, N)))
                bev_feat = bev_feat.movedim(-1, -2)
            else:
                bev_feat = sampling_vt_pytorch(
                    image_uvd=image_uvd,
                    features_pv=features_pv,
                    depths=depths,
                    B=B,
                    N=N,
                    image_size=self.image_size,
                    depth_weight_threshold=self.depth_weight_threshold,
                    depth_distribution=self.depth_distribution,
                    epsilon=1e-6,
                )

        return bev_feat

    def _get_bev_pool_indices(self, voxel_indices):
        voxel_indices, sorted_indices = torch.sort(voxel_indices)

        consecutive_masks = torch.ones_like(voxel_indices, dtype=torch.bool)
        consecutive_masks[1:] = voxel_indices[1:] != voxel_indices[:-1]
        [interval_starts] = torch.where(consecutive_masks)
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1:] = voxel_indices.shape[0] - interval_starts[-1:]

        return (
            voxel_indices,
            interval_starts,
            interval_lengths,
            sorted_indices,
        )

    def _get_valid_indices(self, coords_3d, fov_masks, feature_u, feature_v, image_d):
        B, V, X, Y, Z, _ = coords_3d.shape
        voxel_indices = torch.arange(
            B * X * Y * 1, device=coords_3d.device, dtype=self.index_dtype
        )
        voxel_indices = voxel_indices.view(B, 1, X, Y, 1).expand(B, V, X, Y, Z)

        voxel_indices = voxel_indices[fov_masks]
        feature_u = feature_u[fov_masks]
        feature_v = feature_v[fov_masks]
        image_d = image_d[fov_masks]

        num_valid_voxels_per_camera = fov_masks.sum(dim=(-3, -2, -1))
        camera_indices = torch.arange(B * V, device=num_valid_voxels_per_camera.device, dtype=self.index_dtype)
        camera_indices = camera_indices.repeat_interleave(num_valid_voxels_per_camera.flatten())

        return (
            voxel_indices,
            camera_indices,
            feature_u,
            feature_v,
            image_d,
        )

    def _get_feature_masks(
        self,
        image_uvd: torch.Tensor,
        feature_size: Tuple[int, int],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        image_u, image_v, image_d = torch.unbind(image_uvd, dim=-1)
        image_height, image_width = self.image_size
        feature_height, feature_width = feature_size
        feature_u = image_u / image_width * feature_width
        feature_v = image_v / image_height * feature_height
        feature_d = image_d

        feature_masks_u = (0 <= feature_u) & (feature_u < feature_width)
        feature_masks_v = (0 <= feature_v) & (feature_v < feature_height)
        feature_masks_d = 0.0 < feature_d
        feature_masks = feature_masks_u & feature_masks_v & feature_masks_d

        return (
            feature_u,
            feature_v,
            feature_d,
            feature_masks,
        )

    def _calculate_coords_3d(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        coords_3d = [
            torch.linspace(*roi_range, sample_grid_size + 1, device=device)[:-1] + voxel_size / 2.0
            for roi_range, voxel_size, sample_grid_size in (
                list(zip(self.roi_ranges, self.voxel_size, self.sample_grid_size))
            )
        ]

        coords_3d = torch.meshgrid(*coords_3d, indexing="ij")
        coords_3d = torch.stack(coords_3d, dim=-1)

        return coords_3d


    def _sampling_vt_pillarpool_fused(
        self,
        coords_3d: torch.Tensor,
        image_uvd: torch.Tensor,
        features_pv: torch.Tensor,
        depths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _, _, _ = features_pv.shape
        feature_size = features_pv.shape[-2:]
        feature_height, feature_width = feature_size
        (
            feature_u,
            feature_v,
            image_d,
            fov_masks,
        ) = self._get_feature_masks(
            image_uvd=image_uvd,
            feature_size=feature_size,
        )

        (
            voxel_indices,
            camera_indices,
            feature_u,
            feature_v,
            image_d,
        ) = self._get_valid_indices(
            coords_3d,
            fov_masks,
            feature_u,
            feature_v,
            image_d,
        )

        (
            voxel_indices,
            interval_starts,
            interval_lengths,
            sorted_indices,
        ) = self._get_bev_pool_indices(voxel_indices)

        camera_indices_sorted = camera_indices[sorted_indices]
        feature_v_sorted = feature_v[sorted_indices]
        feature_u_sorted = feature_u[sorted_indices]

        depth_distribution_int = 1 if self.depth_distribution == "laplace" else 0
        bev_feat = sampling_vt_pillarpool_fused(
            depth=depths,
            feat=features_pv.movedim(-3, -1),
            u_coords=feature_u_sorted,
            v_coords=feature_v_sorted,
            z_coords=image_d[sorted_indices],
            batch_camera_indices=camera_indices_sorted,
            ranks_bev=voxel_indices,
            bev_feat_shape=(B, self.output_grid_size[0], self.output_grid_size[1], self.output_grid_size[2], features_pv.size(-3)),
            interval_starts=interval_starts,
            interval_lengths=interval_lengths,
            batch_size=features_pv.shape[0],
            num_cameras=N,
            feat_h=feature_height,
            feat_w=feature_width,
            epsilon=1e-6,
            depth_distribution=depth_distribution_int,
        )
        bev_feat = bev_feat.squeeze(-1)

        return bev_feat

def project_coords(
    coords_xyz: torch.Tensor,
    projection_matrices: torch.Tensor = None,
) -> torch.Tensor:
    coords_xyzw = nn.functional.pad(coords_xyz, (0, 1), mode="constant", value=1.0)
    
    camera_uvd = torch.einsum("bnij,bn...j->bn...i", projection_matrices, coords_xyzw)
    
    epsilon = 1e-6
    clamped_d = torch.where(
        camera_uvd[..., 2:3] >= 0.0,
        torch.clamp(camera_uvd[..., 2:3], min=epsilon),
        torch.clamp(camera_uvd[..., 2:3], max=-epsilon)
    )
    camera_uvd[..., 0:1] = camera_uvd[..., 0:1] / clamped_d
    camera_uvd[..., 1:2] = camera_uvd[..., 1:2] / clamped_d
    
    return camera_uvd
