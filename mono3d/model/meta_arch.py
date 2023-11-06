import torch.nn as nn
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from vision_base.utils.builder import build
from vision_base.networks.models.backbone.dla import DLA
from vision_base.networks.models.backbone.yolo import YOLOPAFPN
from vision_base.networks.models.backbone.dla_utils import DLASegUpsample

class MonoFlex_core(nn.Module):
    """Some Information about RTM3D_core"""
    def __init__(self, backbone_arguments=dict(), **kwargs):
        super(MonoFlex_core, self).__init__()
        self.backbone = build(**backbone_arguments)
        
        if isinstance(self.backbone, DLA):
            feature_size = 64
            print(f"Apply DLA Upsampling instead, feature_size={feature_size}")
            self.deconv_layers = DLASegUpsample(
                input_channels=[16, 32, 64, 128, 256, 512],
                down_ratio=4,
                final_kernel=1,
                last_level=5,
                out_channel=64, ## Notice that if in DLA the head_feature_size should be 256 and input features should be 64 for the heads.
            )
        elif isinstance(self.backbone, YOLOPAFPN):
            feature_size = 256
            output_features = int(256 * self.backbone.width) 
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(output_features, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.SiLU(inplace=True),
            )
        else:
            feature_size = 256
            output_features = 512
            print(f"Apply Baseline ConvTranspose Upsampling instead, feature_size={feature_size}")
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(output_features, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(feature_size, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(feature_size, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
            )
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                
    def forward(self, x):
        x = self.backbone(x['image'])
        if isinstance(self.backbone, DLA):
            x = self.deconv_layers(x)
        elif isinstance(self.backbone, YOLOPAFPN):
            x = self.deconv_layers(x[0])
        else:
            x = self.deconv_layers(x[-1])
        return x
    

class MonoFlex(BaseMetaArch):
    """
        KM3D
    """
    def __init__(self, network_cfg):
        super(MonoFlex, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg


    def build_core(self, network_cfg:dict):
        self.core = MonoFlex_core(**network_cfg)

    def build_head(self, network_cfg):
        self.bbox_head = build(**network_cfg.head) # This should be RTM3DHead

    def forward_train(self, data, meta):
        """
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        img_batch = data['image']
        annotations = dict()
        for key in data:
            if isinstance(key, tuple) and key[0] == 'target':
                annotations[key[1]] = data[key]

        features  = self.core(dict(image=img_batch, P2=data['P']))
        output_dict = self.bbox_head(features)

        loss, loss_dict = self.bbox_head.loss(output_dict, annotations, meta, P2=data['P'])

        return dict(loss=loss, loss_dict=loss_dict)
    
    def resize_bboxes_to_original(self, bboxes, P, original_P):
        original_bboxes = bboxes.clone()
        scale_x = original_P[0, 0] / P[0, 0]
        scale_y = original_P[1, 1] / P[1, 1]
        
        shift_left = original_P[0, 2] / scale_x - P[0, 2]
        shift_top  = original_P[1, 2] / scale_y - P[1, 2]
        original_bboxes[:, 0:4:2] += shift_left
        original_bboxes[:, 1:4:2] += shift_top

        original_bboxes[:, 0:4:2] *= scale_x
        original_bboxes[:, 1:4:2] *= scale_y
        return original_bboxes

    def forward_test(self, data, meta):
        """
        """
        img_batch = data['image']
        P2 = data['P']

        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        features  = self.core(dict(image=img_batch, P2=P2))
        output_dict = self.bbox_head(features)

        scores, bboxes, cls_names = self.bbox_head.get_bboxes(output_dict, P2, img_batch)

        original_bboxes = self.resize_bboxes_to_original(bboxes, P2[0], data['original_P'][0])

        return dict(scores=scores, bboxes=bboxes, cls_names=cls_names, original_bboxes=original_bboxes)

    def dummy_forward(self, image, P2):
        """
        """
        features = self.core(dict(image=image, P2=P2))
        output_dict = self.bbox_head(features)
        scores, bboxes, cls_indexes = self.bbox_head.export_get_bboxes(output_dict, P2, image)
        return scores, bboxes, cls_indexes