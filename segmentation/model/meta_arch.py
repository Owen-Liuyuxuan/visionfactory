from typing import List
import torch
import torch.nn as nn
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from vision_base.utils.builder import build
from .u_net import Seg_UNet_Core

class MultiScaleCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, inputs:List[torch.Tensor], target:torch.Tensor):
        total_loss = 0
        for inp in inputs:
            loss = super().forward(inp, target)
            loss = torch.where(
                loss < 1e-5,
                loss * 1e-2,
                loss
            )
            total_loss += loss
        return total_loss


class UNetSeg(BaseMetaArch):
    """ MonoDepthDorn modified from
        https://arxiv.org/pdf/1806.02446.pdf
    """
    def __init__(self, network_cfg):
        super(UNetSeg, self).__init__()
    
        self.output_channel = getattr(network_cfg, 'output_channel', 34)

        if 'name' not in network_cfg:
            network_cfg['name'] = 'segmentation.model.u_net.Seg_UNet_Core'
        self.core = build(**network_cfg)

        self.loss = MultiScaleCrossEntropyLoss(ignore_index=0, reduction='none')

    def forward_train(self, data, meta):
        """
        """
        img_batch = data['image']
        gts = data['gt_image'].long()
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch)
        gts[gts==-1] = 0
        losses = self.loss([feat[f'scale_{i+1}'] for i in range(3) ], gts)
        losses = losses.mean()

        return_dict = dict(
            loss=losses,
            loss_dict=dict(total_loss=losses),
            loss_hm=dict()
        )
        return return_dict
 
    def forward_test(self, data, meta):
        """
        """
        img_batch = data['image']
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch)['scale_1']
        result_dict = dict(
            logits=feat,
            pred_seg=torch.argmax(feat, dim=1)
        )
        return result_dict

    def dummy_forward(self, img_batch):
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch)['scale_1']
        return torch.argmax(feat, dim=1)
