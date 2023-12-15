import clip
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from vision_base.utils.builder import build
from segmentation.model.meta_arch import MultiScaleCrossEntropyLoss

def load_clip_model(model='RN50', device='cuda', jit=False) -> Tuple[torch.nn.Module, int]:
    assert model in clip.available_models()
    model, _ = clip.load(model, device=device, jit=jit)
    if model == 'RN50x16':
        output_features = 768
    else:
        output_features = 512
    return model, output_features

def featurize_texts(texts: Union[str, List[str]], model, device='cuda') -> torch.Tensor:
    if isinstance(texts, str):
        texts = [texts]
    text_features = model.encode_text(clip.tokenize(texts).to(device))
    return text_features # [N, 512]

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / (features.norm(dim=-1, keepdim=True) + 1e-6)

class LanguageSeg(BaseMetaArch):
    """ LSeg Learned from 
    https://github.com/isl-org/lang-seg/tree/main
    """
    def __init__(self, network_cfg):
        super(LanguageSeg, self).__init__()
        base_clip_model = getattr(network_cfg, 'base_clip_model', 'RN50')
        self.max_label_set = getattr(network_cfg, 'max_label_set', 80)
        self.text_learning_ratio = getattr(network_cfg, 'text_learning_ratio', 0.1)
        self.upsamples = nn.ModuleDict({
            'scale_1': nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            'scale_2': nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            'scale_3': nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        })
        self.clip_model, self.output_features = load_clip_model(base_clip_model)
        for param in self.clip_model.visual.parameters():
            param.requires_grad=False
        self.frozen_clip_model, _ = load_clip_model(base_clip_model)
        for param in self.frozen_clip_model.parameters():
            param.requires_grad=False
        self.regularization = getattr(network_cfg, 'regularization', 1.0)

        self.core = build(**network_cfg)
        self.register_parameter('logit_scale', nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp()))
        self.loss = MultiScaleCrossEntropyLoss(ignore_index=0, reduction='none')
        self.default_text_list = ['unlabeled', 'car', 'pedestrian', 'sidewalk', 'road', 'motocycle', 'sky', 'building']
    
    @torch.no_grad()
    def create_text_feature_batch(self, label_sets):
        mask = torch.zeros((len(label_sets), self.max_label_set), dtype=torch.bool)
        text_features = torch.zeros((len(label_sets),
                                     self.max_label_set,
                                     self.output_features))
        for i, label_set in enumerate(label_sets):
            text_features[i, :len(label_set)] = normalize_features(
                featurize_texts(label_set, self.clip_model, text_features.device) # [n, 512]
            )
            mask[i, :len(label_set)] = True
        return text_features, mask
    

    def regularize_loss_text_features(self, text_features, frozen_text_features):
        loss = torch.nn.functional.relu(1 - (text_features * frozen_text_features).sum(-1)) # [B, n]
        loss = loss.mean().detach()
        return loss

    # @torch.autocast(device_type='cuda')
    def forward_train(self, data, meta):
        """
        """
        img_batch = data['image']
        gts = data['gt_image'].long()
        label_sets = data['label_sets']

        B = img_batch.shape[0]
        image_feature = self.core(img_batch) # Dict[B, C, H, W]
        
        segmentation_loss = 0
        regularization_loss = 0
        loss_dict = {}
        outputs = []
        for i in range(B):
            label_set = label_sets[i]
            text_features = normalize_features(
                featurize_texts(label_set, self.clip_model, img_batch.device) # [n, C]
            )
            with torch.no_grad():
                frozen_text_features = normalize_features(
                    featurize_texts(label_set, self.frozen_clip_model, img_batch.device) # [n, C]
                )
            ## Using loss to limit the variation of text features
            regularization_loss = regularization_loss + self.regularize_loss_text_features(text_features, frozen_text_features)
 
            ## Lowering the segmentation-related learning rate for text features
            text_features = text_features * self.text_learning_ratio + \
                        text_features.detach() * (1 - self.text_learning_ratio)

            n = text_features.shape[0]
            outputs.append([])
            for key in image_feature.keys():
                feat = image_feature[key][i].clone() # [C, H, W]
                _, H, W = feat.shape
                feat = feat.permute(1, 2, 0).reshape(-1, self.output_features) # [HW, C]
                feat = normalize_features(feat) # [HW, C]
                logits = self.logit_scale * feat.half() @ text_features.t() # [HW, n]
                out = logits.float().view(H, W, n).permute(2, 0, 1) # [n, H, W]
                outputs[i].append(self.upsamples[key](out[None]))
            segmentation_loss = segmentation_loss + self.loss(outputs[i], gts[i:i+1]).mean()

        total_loss = (segmentation_loss + regularization_loss * self.regularization ) / B

        loss_dict['total_loss'] = total_loss
        loss_dict['segmentation_loss'] = segmentation_loss
        loss_dict['regularization_loss'] = regularization_loss

        return_dict = dict(
            loss=total_loss,
            loss_dict=loss_dict,
            loss_hm=dict()
        )
        return return_dict
    
    def forward_test(self, data, meta):
        """
        """
        img_batch = data['image']
        if 'label_sets' not in data:
            label_sets = [self.default_text_list for _ in range(img_batch.shape[0])]
        label_sets = data['label_sets']

        B = img_batch.shape[0]
        image_feature = self.core(img_batch) # Dict[B, C, H, W]
        
        pred_seg = []
        for i in range(B):
            label_set = label_sets[i]
            text_features = normalize_features(
                featurize_texts(label_set, self.clip_model, img_batch.device) # [n, C]
            )
            n = text_features.shape[0]
            logits = []
            feat = image_feature['scale_1'][i]
            _, H, W = feat.shape
            feat = feat.permute(1, 2, 0).reshape(-1, self.output_features) # [HW, C]
            feat = normalize_features(feat) # [HW, C]
            logits = self.logit_scale * feat.half() @ text_features.t() # [HW, n]
            out = logits.float().view(H, W, n).permute(2, 0, 1) # [n, H, W]
            out = self.upsamples['scale_1'](out[None]) #[1, n, H, W]
            pred_seg.append(torch.argmax(out, dim=1))

        pred_seg = torch.cat(pred_seg, dim=0) # [B, H, W]
        result_dict = dict(
            pred_seg=pred_seg
        )
        return result_dict
