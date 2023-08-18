
import torch
import torch.nn as nn
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from .u_net import Seg_UNet_Core

class UNetSeg(BaseMetaArch):
    """ MonoDepthDorn modified from
        https://arxiv.org/pdf/1806.02446.pdf
    """
    def __init__(self, network_cfg):
        super(UNetSeg, self).__init__()
    
        self.output_channel = getattr(network_cfg, 'output_channel', 34)
        self.backbone_arguments = getattr(network_cfg, 'backbone')

        self.core = Seg_UNet_Core(3, self.output_channel, backbone_arguments=self.backbone_arguments)

        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        #self.loss = DiceLoss(ignore_index=0)

    def forward_train(self, data, meta):
        """
        """
        img_batch = data['image']
        gts = data['gt_image'].long()
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch)
        gts[gts==-1] = 0
        losses = self.loss(feat['scale_1'], gts)
        losses = torch.where(
            losses < 1e-5,
            losses * 1e-2,
            losses
        )
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
