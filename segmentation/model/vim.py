# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/backbones/bisenetv1.py
import torch
import torch.nn as nn
from vision_base.networks.models.backbone.mamba.vim import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual as Vim
from vision_base.utils.builder import build
import torch.nn.functional as F
import warnings


class VimSeg(nn.Module):
    """VimSeg backbone.

    This backbone is the implementation of `Vision Mamba: Efficient Visual Representation
      Learning with Bidirectional State Space Model 
      <https://arxiv.org/abs/2401.09417>`.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input
            image. Default: 3.
        spatial_channels (Tuple[int]): Size of channel numbers of
            various layers in Spatial Path.
            Default: (64, 64, 64, 128).
        context_channels (Tuple[int]): Size of channel numbers of
            various modules in Context Path.
            Default: (128, 256, 512).
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        out_channels(int): The number of channels of output.
            It must be the same with `in_channels` of decode_head.
            Default: 256.
    """

    def __init__(self,
                 backbone_cfg,
                 out_channels=256, output_num_classes=45,
                 *args, **kwargs):

        super().__init__()
        self.backbone = build(**backbone_cfg)
        self.outConv = nn.Sequential(
            nn.Conv2d(out_channels, output_num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        b, _, H, W = x.shape
        x = self.backbone(x, return_features=True)
        x = x.reshape(b, H//16, W//16, -1).permute(0, 3, 1, 2).contiguous()
        feat_1 = self.outConv(x)
        return dict(scale_1=feat_1)

