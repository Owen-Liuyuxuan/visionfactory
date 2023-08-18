
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from vision_base.utils.builder import build
from vision_base.networks.blocks.blocks import ConvBnReLU

class NaiveBEVMetaArch(BaseMetaArch):
    def __init__(self, network_cfg):
        super(NaiveBEVMetaArch, self).__init__()
        self.network_cfg = network_cfg
        self.output_channel = getattr(network_cfg, 'output_channel', 45)

        self._build_model()
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def _build_model(self):
        self.backbone = build(**self.network_cfg.backbone_cfg)
        self.transform = nn.Sequential(
            ConvBnReLU(512, 32),
            nn.Flatten(),
            nn.Linear(
            (192 // 32) * (640 // 32) * 32,
            (768 // 16) * (704 // 16) * 32),
        )
        self.upsample = nn.Sequential(
            ConvBnReLU(32, 16, (3, 3)),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnReLU(16, 16, (3, 3)),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 45, 1),
            nn.Upsample(scale_factor=4)
        )

    def forward_train(self, data, meta):
        """
        """
        img_batch = data['image']
        gts = data['bev_msk'].long()
        N, C, H, W = img_batch.shape
        feat = self.backbone(img_batch)[0] #[12 * 4]
        bev_features = self.transform(feat).reshape(N, 32, 768 // 16, 704 // 16)
        output = self.upsample(bev_features)

        gts[gts==-1] = 0
        losses = self.loss(output, gts)
        losses = torch.where(
            losses < 1e-5,
            losses * 1e-2,
            losses
        )
        losses = losses.mean()

        return_dict = dict(
            loss=losses,
            loss_dict=dict(total_loss=losses),
            loss_hm=dict() # add images-like features to visualize on tensorboard
        )
        return return_dict
 
    def forward_test(self, data, meta):
        """
        """
        img_batch = data['image']
        N, C, H, W = img_batch.shape
        feat = self.backbone(img_batch)[0] #[12 * 4]
        bev_features = self.transform(feat).reshape(N, 32, 768 // 16, 704 // 16)
        output = self.upsample(bev_features)
        result_dict = dict(
            logits=feat,
            pred_seg=torch.argmax(output, dim=1)
        )
        return result_dict

    def dummy_forward(self, img_batch):
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch)['scale_1']
        return torch.argmax(feat, dim=1)
