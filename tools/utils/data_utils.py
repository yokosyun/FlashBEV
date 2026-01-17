# Copyright (c) Shunsuke Yokokawa. All rights reserved.

import json
from typing import Dict, List, Tuple, Optional

import torch


def load_calibration_params(input_path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load calibration parameters from JSON file."""
    with open(input_path, "r") as f:
        calib_params = json.load(f)
    
    result = {
        "sensor2ego": torch.tensor(calib_params["sensor2ego"], device=device, dtype=torch.float32),
        "ego2global": torch.tensor(calib_params["ego2global"], device=device, dtype=torch.float32),
        "camera2imgs": torch.tensor(calib_params["camera2imgs"], device=device, dtype=torch.float32),
        "post_rots": torch.tensor(calib_params["post_rots"], device=device, dtype=torch.float32),
        "post_trans": torch.tensor(calib_params["post_trans"], device=device, dtype=torch.float32),
        "bda": torch.tensor(calib_params["bda"], device=device, dtype=torch.float32),
    }
    
    print(f"Calibration parameters loaded from {input_path}")
    return result


def create_projection_matrix(
    sensor2ego: torch.Tensor,
    ego2global: torch.Tensor,
    camera2imgs: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    bda: torch.Tensor,
) -> torch.Tensor:
    """Create 4x4 projection matrices from camera parameters."""
    B, N, _, _ = camera2imgs.shape
    
    extrinsic_matrices = sensor2ego
    extrinsic_matrices = torch.inverse(extrinsic_matrices)
    
    intrinsic_matrices = torch.eye(4, device=camera2imgs.device).repeat(B, N, 1, 1)
    intrinsic_matrices[..., :3, :3] = torch.matmul(post_rots, camera2imgs)
    intrinsic_matrices[..., :3, 2] += post_trans
    
    projection_matrices = torch.matmul(intrinsic_matrices, extrinsic_matrices)
    
    return projection_matrices


def create_dummy_input(
    batch_size: int,
    num_cameras: int,
    in_channels: int,
    feature_h: int,
    feature_w: int,
    device: str = "cuda",
    calib_params: Dict[str, torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], Dict]:
    """Create dummy input for view transformer."""
    img_feat = torch.randn(
        batch_size, num_cameras, in_channels, feature_h, feature_w,
        device=device, dtype=torch.float32
    )
    
    if calib_params is not None:
        sensor2ego = calib_params["sensor2ego"]
        ego2global = calib_params["ego2global"]
        camera2imgs = calib_params["camera2imgs"]
        post_rots = calib_params["post_rots"]
        post_trans = calib_params["post_trans"]
        bda = calib_params["bda"]
        
        if sensor2ego.shape[0] != batch_size:
            if sensor2ego.shape[0] == 1:
                sensor2ego = sensor2ego.expand(batch_size, -1, -1, -1)
                ego2global = ego2global.expand(batch_size, -1, -1, -1)
                camera2imgs = camera2imgs.expand(batch_size, -1, -1, -1)
                post_rots = post_rots.expand(batch_size, -1, -1, -1)
                post_trans = post_trans.expand(batch_size, -1, -1)
                bda = bda.expand(batch_size, -1, -1)
            else:
                sensor2ego = sensor2ego[:batch_size]
                ego2global = ego2global[:batch_size]
                camera2imgs = camera2imgs[:batch_size]
                post_rots = post_rots[:batch_size]
                post_trans = post_trans[:batch_size]
                bda = bda[:batch_size]
        
        if sensor2ego.shape[1] != num_cameras:
            sensor2ego = sensor2ego[:, :num_cameras]
            ego2global = ego2global[:, :num_cameras]
            camera2imgs = camera2imgs[:, :num_cameras]
            post_rots = post_rots[:, :num_cameras]
            post_trans = post_trans[:, :num_cameras]
    else:
        sensor2ego = torch.eye(4, device=device).view(1, 1, 4, 4).expand(
            batch_size, num_cameras, 4, 4
        )
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).expand(
            batch_size, num_cameras, 4, 4
        )
        
        camera2imgs = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size, num_cameras, 3, 3
        )
        camera2imgs[..., 0, 0] = 700.0
        camera2imgs[..., 1, 1] = 700.0
        camera2imgs[..., 0, 2] = feature_w / 2
        camera2imgs[..., 1, 2] = feature_h / 2
        
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).expand(
            batch_size, num_cameras, 3, 3
        )
        post_trans = torch.zeros(batch_size, num_cameras, 3, device=device)
        bda = torch.eye(4, device=device).view(1, 4, 4).expand(batch_size, 4, 4)
    
    input_list = [
        img_feat,
        sensor2ego,
        ego2global,
        camera2imgs,
        post_rots,
        post_trans,
        bda,
    ]
    
    return input_list, {}


def create_flashbevpool_data(
    batch_size: int,
    num_cameras: int,
    in_channels: int,
    feature_h: int,
    feature_w: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    roi_range: List[float],
    device: str = "cuda",
    calib_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Create dummy data for FlashBEVPool kernel-only benchmarking."""
    image_feats = torch.randn(
        batch_size, num_cameras, feature_h, feature_w, in_channels,
        device=device, dtype=torch.float32
    )
    
    depth_params = torch.randn(
        batch_size, num_cameras, feature_h, feature_w, 2,
        device=device, dtype=torch.float32
    )
    depth_params[..., 0] = depth_params[..., 0].abs() * 10.0 + 5.0
    depth_params[..., 1] = depth_params[..., 1].abs() * 2.0 + 0.5
    
    if calib_params is not None:
        sensor2ego = calib_params["sensor2ego"]
        ego2global = calib_params["ego2global"]
        camera2imgs = calib_params["camera2imgs"]
        post_rots = calib_params["post_rots"]
        post_trans = calib_params["post_trans"]
        bda = calib_params["bda"]
        
        if sensor2ego.shape[0] != batch_size:
            if sensor2ego.shape[0] == 1:
                sensor2ego = sensor2ego.expand(batch_size, -1, -1, -1)
                ego2global = ego2global.expand(batch_size, -1, -1, -1)
                camera2imgs = camera2imgs.expand(batch_size, -1, -1, -1)
                post_rots = post_rots.expand(batch_size, -1, -1, -1)
                post_trans = post_trans.expand(batch_size, -1, -1)
                bda = bda.expand(batch_size, -1, -1)
            else:
                sensor2ego = sensor2ego[:batch_size]
                ego2global = ego2global[:batch_size]
                camera2imgs = camera2imgs[:batch_size]
                post_rots = post_rots[:batch_size]
                post_trans = post_trans[:batch_size]
                bda = bda[:batch_size]
        
        if sensor2ego.shape[1] != num_cameras:
            sensor2ego = sensor2ego[:, :num_cameras]
            ego2global = ego2global[:, :num_cameras]
            camera2imgs = camera2imgs[:, :num_cameras]
            post_rots = post_rots[:, :num_cameras]
            post_trans = post_trans[:, :num_cameras]
    else:
        sensor2ego = torch.eye(4, device=device).view(1, 1, 4, 4).expand(batch_size, num_cameras, 4, 4)
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).expand(batch_size, num_cameras, 4, 4)
        camera2imgs = torch.eye(3, device=device).view(1, 1, 3, 3).expand(batch_size, num_cameras, 3, 3)
        camera2imgs[..., 0, 0] = 700.0
        camera2imgs[..., 1, 1] = 700.0
        camera2imgs[..., 0, 2] = feature_w / 2
        camera2imgs[..., 1, 2] = feature_h / 2
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).expand(batch_size, num_cameras, 3, 3)
        post_trans = torch.zeros(batch_size, num_cameras, 3, device=device)
        bda = torch.eye(4, device=device).view(1, 4, 4).expand(batch_size, 4, 4)
    
    projection_matrices = create_projection_matrix(
        sensor2ego, ego2global, camera2imgs, post_rots, post_trans, bda
    )
    
    feature_size = torch.tensor(
        [[feature_h, feature_w]] * batch_size * num_cameras,
        device=device, dtype=torch.int32
    ).view(batch_size, num_cameras, 2)
    
    image_size = torch.tensor([feature_h * 16, feature_w * 16], device=device, dtype=torch.int32)
    roi_range_tensor = torch.tensor(roi_range, device=device, dtype=torch.float32)
    grid_size = torch.tensor([grid_x, grid_y, grid_z], device=device, dtype=torch.int32)
    
    return {
        "image_feats": image_feats,
        "depth_params": depth_params,
        "projection_matrices": projection_matrices,
        "feature_size": feature_size,
        "image_size": image_size,
        "roi_range": roi_range_tensor,
        "grid_size": grid_size,
    }

