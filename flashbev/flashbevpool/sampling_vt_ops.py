# Copyright (c) Phigent Robotics. All rights reserved.

import torch

try:
    from . import sampling_vt_ext
    HAS_SAMPLING_VT_EXT = True
except ImportError:
    HAS_SAMPLING_VT_EXT = False
    print("Warning: sampling_vt_ext not available. Please rebuild the extension.")


__all__ = ['sampling_vt_pillarpool', 'sampling_vt_pillarpool_fused']


class SamplingVTPillarPool(torch.autograd.Function):
    r"""Sampling-VT pillar pooling implementation.
    """
    @staticmethod
    def forward(ctx, feat, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        if not HAS_SAMPLING_VT_EXT:
            raise ImportError("sampling_vt_ext is required. Please rebuild the extension.")
        
        ranks_bev = ranks_bev.int()
        feat = feat.contiguous().float()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        sampling_vt_ext.sampling_vt_pillarpool_forward(
            feat,
            out,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, feat, ranks_feat, interval_starts, interval_lengths)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, feat, ranks_feat, interval_starts, interval_lengths = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_bev = \
            ranks_feat[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        feat = feat.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        
        return feat_grad, None, None, None, None, None, None

def sampling_vt_pillarpool(feat, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = SamplingVTPillarPool.apply(feat, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

def sampling_vt_pillarpool_fused(depth, feat, u_coords, v_coords, z_coords,
                batch_camera_indices, ranks_bev, bev_feat_shape,
                interval_starts, interval_lengths, batch_size, num_cameras,
                feat_h, feat_w, epsilon=1e-6, depth_distribution=0):
    if not HAS_SAMPLING_VT_EXT:
        raise ImportError("sampling_vt_ext is required. Please rebuild the extension.")
    
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
