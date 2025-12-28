# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

try:
    from . import bevpoolv2_ext
    HAS_BEVPOOLV2_EXT = True
except ImportError:
    HAS_BEVPOOLV2_EXT = False
    print("Warning: bevpoolv2_ext not available. Please rebuild the extension.")


__all__ = ['bev_pool_v2_classification_nearest', 'bev_pool_v2_regression_nearest']


class BEVPoolV2_ClassificationNearest(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        if not HAS_BEVPOOLV2_EXT:
            raise ImportError("bevpoolv2_ext is required. Please rebuild the extension.")
        
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        bevpoolv2_ext.bev_pool_v2_classification_nearest_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bevpoolv2_ext.bev_pool_v2_classification_nearest_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, \
            None, None, None

class BEVPoolV2_RegressionNearest(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        if not HAS_BEVPOOLV2_EXT:
            raise ImportError("bevpoolv2_ext is required. Please rebuild the extension.")
        
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        bevpoolv2_ext.bev_pool_v2_regression_nearest_forward(
            depth,
            feat,
            out,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat = ctx.saved_tensors

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

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bevpoolv2_ext.bev_pool_v2_regression_nearest_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, \
            None, None, None

def bev_pool_v2_classification_nearest(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = BEVPoolV2_ClassificationNearest.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

def bev_pool_v2_regression_nearest(depth, feat, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = BEVPoolV2_RegressionNearest.apply(depth, feat, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
