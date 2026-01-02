import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

try:
    from mmcv.runner import BaseModule
    HAS_MMCV = True
except ImportError:
    BaseModule = nn.Module
    HAS_MMCV = False

from flashbevpool import flash_bevpool

try:
    from .bevpoolv2 import (
        bev_pool_v2_classification_nearest,
        bev_pool_v2_regression_nearest,
    )
    from .sampling_vt_ops import (
        sampling_vt_pillarpool,
        sampling_vt_pillarpool_fused,
    )
    HAS_MMDET3D_OPS = True
except ImportError:
    HAS_MMDET3D_OPS = False
    print("Warning: flashbevpool ops not available. Some SamplingVT methods will be disabled.")


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int32)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def depth_to_colormap(depth_tensor, min_depth=0.0, max_depth=3.0):
    """
    Convert depth tensor to RGB colormap using continuous jet colormap.
    
    Args:
        depth_tensor: torch.Tensor with depth values
        min_depth: minimum depth value for normalization
        max_depth: maximum depth value for normalization
    
    Returns:
        torch.Tensor with shape [..., 3] with RGB values in [0, 1]
    """
    depth_normalized = (depth_tensor - min_depth) / (max_depth - min_depth)
    depth_normalized = torch.clamp(depth_normalized, 0.0, 1.0)
    
    r = torch.clamp(4.0 * depth_normalized - 1.5, 0.0, 1.0) - torch.clamp(4.0 * depth_normalized - 3.5, 0.0, 1.0)
    g = torch.clamp(4.0 * depth_normalized - 0.5, 0.0, 1.0) - torch.clamp(4.0 * depth_normalized - 2.5, 0.0, 1.0)
    b = torch.clamp(4.0 * depth_normalized + 0.5, 0.0, 1.0) - torch.clamp(4.0 * depth_normalized - 1.5, 0.0, 1.0)
    
    result = torch.stack([r, g, b], dim=-1)
    return result


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


class SamplingVT(BaseModule):
    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
        fuse_projection=True,
        depth_regression=True,
        use_bev_pool=True,
        use_shared_memory=False,
        use_bilinear=True,
        fuse_bilinear=False,
        sample_grid_z=[-5.0, 3.0, 0.2],
        depth_distribution="laplace",
        optimize_z_precompute=True,
        use_warp_kernel=False,
        use_vectorized_load=False,
        depth_weight_threshold=0.0,
    ):
        super(SamplingVT, self).__init__()

        self.depth_regression = depth_regression
        self.use_bev_pool = use_bev_pool
        self.VISUALIZE_PROJECTION = False
        self.VISUALIZE_LIDAR = False
        self.index_dtype = torch.int64
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.image_size = input_size
        self.depth_range = grid_config['depth'][:2]

        self.use_bilinear = use_bilinear
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

        if self.depth_regression:
            self.depth_network = nn.Sequential(
                nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
                nn.ReLU()
            )
        else:
            self.depth_network = nn.Sequential(
                nn.Conv2d(in_channels, self.num_depth_bins, kernel_size=1, padding=0),
                nn.Softmax(dim=1)
            )

        self.register_buffer(
            name="roi_ranges",
            tensor=torch.as_tensor(
                [grid_config["x"][:2], grid_config["y"][:2], grid_config["z"][:2]]),
        )
        self.register_buffer(
            name="output_grid_size",
            tensor=torch.tensor([(cfg[1] - cfg[0]) / cfg[2]
                                    for cfg in [grid_config["x"], grid_config["y"], grid_config["z"]]], dtype=torch.int32),
        )

        self.register_buffer(
            name="sample_grid_size",
            tensor=torch.tensor([(cfg[1] - cfg[0]) / cfg[2]
                                    for cfg in [grid_config["x"], grid_config["y"], sample_grid_z]], dtype=torch.int32),
        )
        if self.fuse_projection == False:
            self.register_buffer(
                name="voxel_size",
                tensor=torch.as_tensor([grid_config["x"][2], grid_config["y"][2], sample_grid_z[2]]),
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
        if self.depth_regression:
            depth_params = self.depth_network(x)
            depth_mu, depth_sigma = depth_params.chunk(chunks=2, dim=1)
            depth_sigma = depth_sigma + 0.1
            depth_mu = depth_mu.squeeze(1)
            depth_sigma = depth_sigma.squeeze(1)
            depth = torch.stack([depth_mu, depth_sigma], dim=-1)
        else:
            depth = self.depth_network(x)
        return depth

    def forward(self, input, images=None, img_meta=None, lidar_cloud=None, depth_from_lidar=None):
        if self.VISUALIZE_LIDAR:
            canva_size=1000
            show_range=50
            import cv2
            canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
            lidar_points_vis = lidar_cloud[:, :3].cpu().numpy()
            lidar_points_vis[:, 1] = -lidar_points_vis[:, 1]
            lidar_points_vis[:, :2] = (lidar_points_vis[:, :2] + show_range) / show_range / 2.0 * canva_size
            for p in lidar_points_vis:
                if check_point_in_img(p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                    color = depth2color(p[2])
                    cv2.circle(
                        canvas, (int(p[0]), int(p[1])),
                        radius=0,
                        color=color,
                        thickness=1)
            cv2.imwrite("canvas.jpg", canvas)

        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        features_pv = self.context_network(x)
        depths = self.forward_depth(x)
        bev_features = self.sampling_vt(input, features_pv, depths)

        return bev_features, depths

    def sampling_vt(self, input, features_pv, depths):
        B, N, _, _ = input[3].shape
        OUT_CHANNELS = features_pv.size(-3)

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
                depth_params=depths.unflatten(0, (B, N)),
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
            if not HAS_MMDET3D_OPS:
                raise ImportError("Non-fused projection requires mmdet3d ops. Install mmdet3d or use fuse_projection=True.")
            
            if self.VISUALIZE_LIDAR:
                coords_3d = lidar_cloud.view(1, 1, *lidar_cloud.shape)[..., :3]
                coords_3d = coords_3d.expand(B, N, *coords_3d.shape)
            else:
                coords_3d = self.coords_3d
                coords_3d = coords_3d.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1, -1)

            image_uvd = project_coords(
                coords_xyz=coords_3d,
                projection_matrices=projection_matrices,
            )

            if self.use_bev_pool:
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

                if self.VISUALIZE_PROJECTION:
                    import torchvision
                    image_height, image_width = self.image_size
                    images = images.flatten(0,1)
                    images = (images - images.min()) / (images.max() - images.min())
                    depth_colored = depth_to_colormap(image_d, min_depth=5, max_depth=35)

                    USE_FEATURE = False
                    if USE_FEATURE:
                        image_u_vis = feature_u * image_width / feature_width
                        image_v_vis = feature_v * image_height / feature_height
                    else:
                        image_u_vis = image_uvd[...,0][fov_masks]
                        image_v_vis = image_uvd[...,1][fov_masks]
                    images[camera_indices, :, image_v_vis.long(), image_u_vis.long()] = depth_colored
                    torchvision.utils.save_image(images, "images.png", nrow=3)

                (
                    voxel_indices,
                    interval_starts,
                    interval_lengths,
                    sorted_indices,
                ) = self._get_bev_pool_indices(voxel_indices)

                camera_indices_sorted = camera_indices[sorted_indices]
                feature_v_sorted = feature_v[sorted_indices]
                feature_u_sorted = feature_u[sorted_indices]
                if self.depth_regression:
                    depth_mu = depths[..., 0]
                    depth_sigma = depths[..., 1]
                    
                    if self.use_bilinear:
                        if self.fuse_bilinear:
                            depth_distribution_int = 1 if self.depth_distribution == "laplace" else 0
                            bev_feat = sampling_vt_pillarpool_fused(
                                depth=depths.unflatten(0, (B, N)),
                                feat=features_pv.movedim(-3, -1).unflatten(0, (B, N)),
                                u_coords=feature_u_sorted,
                                v_coords=feature_v_sorted,
                                z_coords=image_d[sorted_indices],
                                batch_camera_indices=camera_indices_sorted,
                                ranks_bev=voxel_indices,
                                bev_feat_shape=(B, self.output_grid_size[0], self.output_grid_size[1], self.output_grid_size[2], OUT_CHANNELS),
                                interval_starts=interval_starts,
                                interval_lengths=interval_lengths,
                                batch_size=B,
                                num_cameras=N,
                                feat_h=feature_height,
                                feat_w=feature_width,
                                epsilon=1e-6,
                                depth_distribution=depth_distribution_int,
                            )
                            bev_feat = bev_feat.squeeze(-1)
                        else:
                            u0 = torch.floor(feature_u_sorted)
                            v0 = torch.floor(feature_v_sorted)
                            u1 = torch.clamp(u0 + 1, max=feature_width - 1)
                            v1 = torch.clamp(v0 + 1, max=feature_height - 1)
                            wu = feature_u_sorted - u0
                            wv = feature_v_sorted - v0

                            u0 = u0.long()
                            v0 = v0.long()
                            u1 = u1.long()
                            v1 = v1.long()

                            depth_mu = (1-wu) * (1-wv) * depth_mu[camera_indices_sorted, v0, u0] + (wu) * (1-wv) * depth_mu[camera_indices_sorted, v0, u1] + (1-wu) * (wv) * depth_mu[camera_indices_sorted, v1, u0] + (wu) * (wv) * depth_mu[camera_indices_sorted, v1, u1]
                            depth_sigma = (1-wu) * (1-wv) * depth_sigma[camera_indices_sorted, v0, u0] + (wu) * (1-wv) * depth_sigma[camera_indices_sorted, v0, u1] + (1-wu) * (wv) * depth_sigma[camera_indices_sorted, v1, u0] + (wu) * (wv) * depth_sigma[camera_indices_sorted, v1, u1]
                            
                            epsilon = 1e-4
                            z_score = (depth_mu - image_d[sorted_indices]) / (depth_sigma + epsilon);
                            depth_weight = self.compute_depth_weight(z_score, depth_sigma, epsilon)
                            features_pv = features_pv.movedim(-3, -1)
                            w00 = (1-wu) * (1-wv)
                            w01 = (wu) * (1-wv)
                            w10 = (1-wu) * (wv)
                            w11 = (wu) * (wv)
                            w00 = w00.unsqueeze(-1)
                            w01 = w01.unsqueeze(-1)
                            w10 = w10.unsqueeze(-1)
                            w11 = w11.unsqueeze(-1)

                            features_pv = w00 * features_pv[camera_indices_sorted, v0, u0] + w01 * features_pv[camera_indices_sorted, v0, u1] + w10 * features_pv[camera_indices_sorted, v1, u0] + w11 * features_pv[camera_indices_sorted, v1, u1]
                            
                            weighted_feature = features_pv * depth_weight.unsqueeze(-1)
                            
                            output_shape = (B, 1, self.output_grid_size[1], self.output_grid_size[0], OUT_CHANNELS)
                            n_points = weighted_feature.shape[0]
                            ranks_feat_sequential = torch.arange(
                                n_points,
                                device=weighted_feature.device,
                                dtype=self.index_dtype
                            )
                            weighted_feature_reshaped = weighted_feature.view(1, 1, 1, n_points, OUT_CHANNELS)
                            features_3d = sampling_vt_pillarpool(
                                feat=weighted_feature_reshaped,
                                ranks_feat=ranks_feat_sequential,
                                ranks_bev=voxel_indices,
                                bev_feat_shape=output_shape,
                                interval_starts=interval_starts,
                                interval_lengths=interval_lengths,
                            )
                            bev_feat = features_3d.squeeze(-3)

                    else:
                        feature_u_sorted = self.round_fn(feature_u_sorted)
                        feature_v_sorted = self.round_fn(feature_v_sorted)
                        feature_u_sorted = feature_u_sorted.to(self.index_dtype)
                        feature_v_sorted = feature_v_sorted.to(self.index_dtype)
                        
                        feature_indices = camera_indices_sorted * feature_width * feature_height + feature_v_sorted * feature_width + feature_u_sorted
                        
                        image_d_sorted = image_d[sorted_indices]

                        depth_mu = depths[..., 0]
                        depth_sigma = depths[..., 1]

                        depth_mu = depth_mu[camera_indices_sorted, feature_v_sorted, feature_u_sorted]
                        depth_sigma = depth_sigma[camera_indices_sorted, feature_v_sorted, feature_u_sorted]
                        z_score = (depth_mu - image_d_sorted) / (depth_sigma + 1e-6)
                        depth_prob = self.compute_depth_weight(z_score, depth_sigma, epsilon=1e-6)
                        output_shape = (B, 1, self.output_grid_size[1], self.output_grid_size[0], OUT_CHANNELS)
                        features_3d = bev_pool_v2_regression_nearest(
                            feat=features_pv.movedim(-3, -1).unflatten(0, (B, N)),
                            depth=depth_prob,
                            ranks_bev=voxel_indices,
                            ranks_feat=feature_indices,
                            interval_starts=interval_starts,
                            interval_lengths=interval_lengths,
                            bev_feat_shape=output_shape,
                        )
                        bev_feat = features_3d.squeeze(-3)
                else:
                    if self.use_bilinear:
                        assert False, "not supported"
                    else:
                        feature_u_sorted = self.round_fn(feature_u_sorted)
                        feature_v_sorted = self.round_fn(feature_v_sorted)
                        feature_u_sorted = feature_u_sorted.to(self.index_dtype)
                        feature_v_sorted = feature_v_sorted.to(self.index_dtype)
        
                        feature_indices = camera_indices_sorted * feature_width * feature_height + feature_v_sorted * feature_width + feature_u_sorted

                        d_min, d_max = self.depth_range
                        frustum_d = (image_d - d_min) / (d_max - d_min) * (self.num_depth_bins - 1)
                        frustum_d = torch.clamp(frustum_d, 0, self.num_depth_bins - 1)
                        frustum_d = self.round_fn(frustum_d)
                        frustum_d = frustum_d.to(self.index_dtype)

                        depth_indices = (
                            frustum_d * feature_width * feature_height + feature_v * feature_width + feature_u
                        )
                        depth_indices += camera_indices * feature_width * feature_height * self.num_depth_bins
                        depth_indices = depth_indices[sorted_indices]
                        output_shape = (B, *self.output_grid_size.flip(-1), OUT_CHANNELS)

                        features_3d = bev_pool_v2_classification_nearest(
                            feat=features_pv.movedim(-3, -1).unflatten(0, (B, N)),
                            depth=depths.unflatten(0, (B, N)),
                            ranks_bev=voxel_indices,
                            ranks_depth=depth_indices,
                            ranks_feat=feature_indices,
                            interval_starts=interval_starts,
                            interval_lengths=interval_lengths,
                            bev_feat_shape=output_shape,
                        )
                        bev_feat = torch.cat(features_3d.unbind(dim=2), 1)
            else:
                if self.depth_regression:
                    if self.use_bilinear:
                        image_uvd = image_uvd.flatten(0, 1)
                        
                        image_height, image_width = self.image_size
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

                        feat_u_clamped = torch.clamp(feat_u, 0.0, feat_w - 1.0)
                        feat_v_clamped = torch.clamp(feat_v, 0.0, feat_h - 1.0)

                        features_3d = torch.nn.functional.grid_sample(
                            input=features_pv,
                            grid=coords_uv.flatten(1,2),
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=True,
                        )
                        features_3d = features_3d.unflatten(2, coords_uv.shape[1:3])

                        depths_3d = torch.nn.functional.grid_sample(
                            input=depths.movedim(-1,1),
                            grid=coords_uv.flatten(1,2),
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=True,
                        )
                        depths_3d = depths_3d.unflatten(2, coords_uv.shape[1:3])

                        depth_mu = depths_3d[:, 0]
                        depth_sigma = depths_3d[:, 1]

                        epsilon = 1e-6
                        z_score = (image_d - depth_mu) / (depth_sigma + epsilon)
                        depth_prob_3d = self.compute_depth_weight(z_score, depth_sigma, epsilon)

                        valid_mask = depth_prob_3d >= self.depth_weight_threshold
                        valid_mask = valid_mask & fov_masks

                        features_3d = features_3d * depth_prob_3d.unsqueeze(1)
                        features_3d = features_3d * valid_mask.unsqueeze(1)
                        bev_feat = features_3d.unflatten(0, (B, N)).sum(dim=[1, -1])

                        valid_counts = valid_mask.unflatten(0, (B, N)).sum(dim=[1, -1])
                        bev_feat = bev_feat  / torch.clamp(valid_counts, min=1.0).unsqueeze(1)

                    else:
                        assert False, "not supported"
                else:
                    assert False, "not supported"

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

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

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
