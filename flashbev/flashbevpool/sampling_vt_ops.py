# Copyright (c) Shunsuke Yokokawa. All rights reserved.

from . import sampling_vt_ext

__all__ = ['sampling_vt_pillarpool_fused']


def sampling_vt_pillarpool_fused(depth, feat, u_coords, v_coords, z_coords,
                batch_camera_indices, ranks_bev, bev_feat_shape,
                interval_starts, interval_lengths, batch_size, num_cameras,
                feat_h, feat_w, epsilon=1e-6, depth_distribution=0):
    
    depth = depth.contiguous().float()
    feat = feat.contiguous().float()
    u_coords = u_coords.contiguous().float()
    v_coords = v_coords.contiguous().float()
    z_coords = z_coords.contiguous().float()
    batch_camera_indices = batch_camera_indices.contiguous().int()
    ranks_bev = ranks_bev.contiguous().int()
    interval_lengths = interval_lengths.contiguous().int()
    interval_starts = interval_starts.contiguous().int()
    
    out = feat.new_zeros(bev_feat_shape)
    
    sampling_vt_ext.sampling_vt_pillarpool_fused_forward(
        depth, feat, u_coords, v_coords, z_coords, batch_camera_indices,
        ranks_bev, interval_lengths, interval_starts, out,
        batch_size, num_cameras, feat_h, feat_w, epsilon, depth_distribution
    )
    
    x = out.movedim(-1, 1).contiguous()
    return x
