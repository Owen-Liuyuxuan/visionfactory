# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/backbones/bisenetv1.py
import torch
import torch.nn as nn
from vision_base.networks.blocks.blocks import ConvBnReLU
from vision_base.utils.builder import build
import torch.nn.functional as F
import warnings


# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/utils/wrappers.py
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class SpatialPath(nn.Module):
    """Spatial Path to preserve the spatial size of the original input image
    and encode affluent spatial information.

    Args:
        in_channels(int): The number of channels of input
            image. Default: 3.
        num_channels (Tuple[int]): The number of channels of
            each layers in Spatial Path.
            Default: (64, 64, 64, 128).
    Returns:
        x (torch.Tensor): Feature map for Feature Fusion Module.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(64, 64, 64, 128),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__()
        assert len(num_channels) == 4, 'Length of input channels \
                                        of Spatial Path must be 4!'

        self.layers = []
        for i in range(len(num_channels)):
            layer_name = f'layer{i + 1}'
            self.layers.append(layer_name)
            if i == 0:
                self.add_module(
                    layer_name,
                    ConvBnReLU(in_channels, num_channels[i], 7, stride=2)
                )
            elif i == len(num_channels) - 1:
                self.add_module(
                    layer_name,
                    ConvBnReLU(num_channels[i - 1], num_channels[i], 1))
            else:
                self.add_module(
                    layer_name,
                    ConvBnReLU(num_channels[i - 1], num_channels[i], 3, stride=2))

    def forward(self, x):
        for i, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel):
        super().__init__()
        self.conv_layer = ConvBnReLU(in_channels, out_channel, 3)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnReLU(out_channel, out_channel, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class ContextPath(nn.Module):
    """Context Path to provide sufficient receptive field.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        context_channels (Tuple[int]): The number of channel numbers
            of various modules in Context Path.
            Default: (128, 256, 512).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
    Returns:
        x_16_up, x_32_up (torch.Tensor, torch.Tensor): Two feature maps
            undergoing upsampling from 1/16 and 1/32 downsampling
            feature maps. These two feature maps are used for Feature
            Fusion Module and Auxiliary Head.
    """

    def __init__(self,
                 backbone_cfg,
                 context_channels=(128, 256, 512),
                 align_corners=False):
        super().__init__()
        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.backbone = build(**backbone_cfg)

        self.align_corners = align_corners
        self.arm16 = AttentionRefinementModule(context_channels[1],
                                               context_channels[0])
        self.arm32 = AttentionRefinementModule(context_channels[2],
                                               context_channels[0])
        self.conv_head32 = ConvBnReLU(context_channels[0], context_channels[0], 3)
        self.conv_head16 = ConvBnReLU(context_channels[0], context_channels[0], 3)
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnReLU(context_channels[2], context_channels[0], 1))

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)
        x_gap = self.gap_conv(x_32)

        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = resize(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)

        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = resize(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)

        return x_16_up, x_32_up


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels, 1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            ConvBnReLU(out_channels, out_channels, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        # Note: No BN and more 1x1 conv in paper.
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out


class BiSeNetV1(nn.Module):
    """BiSeNetV1 backbone.

    This backbone is the implementation of `BiSeNet: Bilateral
    Segmentation Network for Real-time Semantic
    Segmentation <https://arxiv.org/abs/1808.00897>`_.

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
                 in_channels=3,
                 spatial_channels=(64, 64, 64, 128),
                 context_channels=(128, 256, 512),
                 out_indices=(0, 1, 2),
                 align_corners=False,
                 out_channels=256, output_num_classes=45,
                 upsample_on=True,
                 *args, **kwargs):

        super().__init__()
        assert len(spatial_channels) == 4, 'Length of input channels \
                                           of Spatial Path must be 4!'

        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.out_indices = out_indices
        self.align_corners = align_corners
        self.context_path = ContextPath(backbone_cfg, context_channels,
                                        self.align_corners)
        self.spatial_path = SpatialPath(in_channels, spatial_channels)
        self.ffm = FeatureFusionModule(context_channels[1], out_channels)
        self.outConv = nn.Sequential(
            nn.Conv2d(out_channels, output_num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) if upsample_on else nn.Identity()
        )
        self.context_8_conv = nn.Sequential(
            nn.Conv2d(context_channels[0], output_num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) if upsample_on else nn.Identity()
        )
        self.context_16_conv = nn.Sequential(
            nn.Conv2d(context_channels[0], output_num_classes, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True) if upsample_on else nn.Identity()
        )

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        x_context8, x_context16 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_context8)

        outs = [x_fuse, x_context8, x_context16]
        outs = [outs[i] for i in self.out_indices]
        feat_1 = self.outConv(outs[0])
        feat_2 = self.context_8_conv(outs[1])
        feat_3 = self.context_16_conv(outs[2])
        return dict(scale_1=feat_1, scale_2=feat_2, scale_3=feat_3)

