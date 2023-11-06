from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from easydict import EasyDict
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from mono3d.model.utils import ClipBoxes, calc_iou, alpha2theta_3d, BackProjection
from mono3d.model.rtm3d_utils import _transpose_and_gather_feat,\
     compute_rot_loss, _nms, _topk, decode_depth_from_keypoints, decode_depth_inv_sigmoid_calibration

class IoULoss(nn.Module):
    """Some Information about IoULoss"""
    def forward(self, preds:torch.Tensor, targets:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """IoU Loss

        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targets (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]

        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        
        # overlap
        lt = torch.max(preds[..., :2], targets[..., :2])
        rb = torch.min(preds[..., 2:], targets[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
        ag = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)
        return -ious.log()


class MonoFlexHead(nn.Module):
    def __init__(self, num_joints:int=9,
                       max_objects:int=32,
                       learned_classes:List[str]=[],
                       data3d_json:str="/data/whl.json",
                       layer_cfg=EasyDict(),
                       loss_cfg=EasyDict(),
                       test_cfg=EasyDict()):
        super(MonoFlexHead, self).__init__()
        self._init_layers(**layer_cfg)
        self.build_loss(**loss_cfg)
        self.test_cfg = test_cfg
        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1],
            [-1, 0], [0, -1], [-1, 0], [0, -1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('const', const) # self.const

        self.learned_classes = learned_classes
        self.num_classes = len(learned_classes)
        self.num_joints  = num_joints
        self.max_objects = max_objects
        self.clipper = ClipBoxes()
        self.backprojector = BackProjection()
                
        self.empirical_whl = json.load(
            open(data3d_json, 'r')
        )
        self.register_buffer('cls_mean_size', torch.from_numpy(
            np.array(
                [self.empirical_whl[class_name]['whl'] for class_name in self.learned_classes], dtype=np.float32
            ))
        )
    
    @staticmethod
    def _neg_loss(pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float() * gt.gt(-1e-4).float() # define less than 0 points to be masked out

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred_prob = torch.sigmoid(pred)

        pos_loss = nn.functional.logsigmoid(pred) * torch.pow(1 - pred_prob, 2) * pos_inds
        pos_loss = torch.where(
            pred_prob > 0.99,
            torch.zeros_like(pos_loss),
            pos_loss
        )
        neg_loss = nn.functional.logsigmoid(- pred) * torch.pow(pred_prob, 2) * neg_weights * neg_inds
        neg_loss = torch.where(
            pred_prob < 0.01,
            torch.zeros_like(neg_loss),
            neg_loss
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    @staticmethod
    def _RegWeightedL1Loss(output, mask, ind, target, dep):
        dep=dep.squeeze(2)
        dep[dep<5]=dep[dep<5]*0.01
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        #losss=torch.abs(pred * mask-target * mask)
        #loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss=torch.abs(pred * mask-target * mask)
        loss=torch.sum(loss,dim=2)*dep
        loss=loss.sum()
        loss = loss / (mask.sum() + 1e-4)

        return loss

    @staticmethod
    def _RegL1Loss(output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

    @staticmethod
    def _RotLoss(output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

    def _init_layers(self,
                    input_features=256,
                    head_features=64,
                    head_dict=dict(),
                     **kwargs):
        # self.head_dict = head_dict
        self.head_layers = nn.ModuleDict()
        for head_name, num_output in head_dict.items():
            self.head_layers[head_name] = nn.Sequential(
                    nn.Conv2d(input_features, head_features, 3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features, num_output, 1)
                )
            
            if 'hm' in head_name:
                output_layer = self.head_layers[head_name][-1]
                nn.init.constant_(output_layer.bias, -2.19)

            else:
                output_layer = self.head_layers[head_name][-1]
                nn.init.normal_(output_layer.weight, std=0.001)
                nn.init.constant_(output_layer.bias, 0)

    def forward(self, x):
        ret = {}
        for head in self.head_layers:
            ret[head] = self.head_layers[head](x)
        return ret

    def build_loss(self,
                   uncertainty_range=[-10, 10],
                   uncertainty_weight=1.0,
                   loss_weight:dict=dict(),
                   **kwargs):
        assert uncertainty_range[1] >= uncertainty_range[0]
        self.bbox2d_loss = IoULoss()
        self.uncertainty_range = uncertainty_range
        self.uncertainty_weight = uncertainty_weight
        self.loss_weight = {'hm_loss': 1, 'hp_loss': 1,
                       'box2d_loss': 1,  'off_loss': 0.5, 'dim_loss': 1,
                       'depth_loss': 1, 'kpd_loss': 0.2,
                       'rot_loss': 0.2, 'soft_depth_loss': 0.2}
        self.loss_weight.update(loss_weight)
        print(self.loss_weight)

    @torch.no_grad()
    def update_2d_pseudo_label(self, output, annotations):
        """
        What we want, all pixels outside 2D bboxes be 0, selected maximum points to be 1.
        All points inside 2D bboxes to be -1.

        Seperate the task into two parts:
            All pixels to be 0 / All points inside 2D bboxes to be -1  : done in dataset.
            select maximum point to be 1.
        """
        B = output['hm'].shape[0]

        for i in range(B):
            if annotations['is_labeled_3d'][i].item():
                continue
            ## Only consider frame instance without 3D labels
            hm = torch.sigmoid(output['hm'][i:i+1]) #[1, C, H, W]
            _, hm_w = hm.shape[2], hm.shape[3]
            heat = _nms(hm) #[1, C, H, W]
            K = 200
            scores, inds, clses, ys, xs = _topk(heat, K=K)
            output_batch_d = dict()
            for key in output:
                output_batch_d[key] = output[key][i:i+1]

            gathered_output = self._gather_output(output_batch_d, inds.long(), torch.ones_like(scores).bool())

            bbox2d = self._decode(gathered_output['bbox2d'], torch.stack([xs, ys], dim=-1)) # [1, K, 4]
            iou_matrix = calc_iou(annotations['bboxes2d'][i], bbox2d[0]) # [max_objects, K]
            iou_matrix = iou_matrix * scores # allow existing large score matrix to become TP
            iou_matrix[iou_matrix < 1e-5] = -1

            max_objects, K = iou_matrix.shape
            cost_matrix = np.zeros([max_objects, K + max_objects])
            cost_matrix[0:max_objects, 0:K] = - iou_matrix.cpu().numpy()

            row_inds, col_inds = linear_sum_assignment(cost_matrix) # [K, ]
            keep_mask = col_inds < K #[max_object, ] numpy bool
            row_inds = row_inds[keep_mask] #[m, ]
            col_inds = col_inds[keep_mask] #[m, ] usable indexes
            M = col_inds.shape[0]

            target_hm = annotations['hm'][i].clone()
            bboxes2d_target = torch.zeros_like(annotations['bboxes2d_target'][i])
            ind = torch.zeros_like(annotations['ind'][i])
            reg_mask = torch.zeros_like(annotations['reg_mask'][i])
            cls_indexes = torch.zeros_like(annotations['cls_indexes'][i])

            reg_mask[0:M] = 1
            gt_bbox2ds = annotations['bboxes2d'][i][row_inds] #[m, 4]
            center_xs = xs[0][col_inds]
            center_ys = ys[0][col_inds]
            cls_indexes[0:M] = annotations['cls_indexes'][i][row_inds]
            ind[0:M] = center_ys * hm_w + center_xs
            bboxes2d_target[0:M, 0] = center_xs - gt_bbox2ds[:, 0]
            bboxes2d_target[0:M, 1] = center_ys - gt_bbox2ds[:, 1]
            bboxes2d_target[0:M, 2] = gt_bbox2ds[:, 2] - center_xs
            bboxes2d_target[0:M, 3] = gt_bbox2ds[:, 3] - center_ys

            # hs = (gt_bbox2ds[:, 3] - gt_bbox2ds[:, 1]).cpu().numpy()
            # ws = (gt_bbox2ds[:, 2] - gt_bbox2ds[:, 0]).cpu().numpy()
            for m in range(M):
                cls_idx = cls_indexes[m].item()
                target_hm[cls_idx][center_ys[m].long().item(), center_xs[m].long().item()] = 1
            annotations['hm'][i] = target_hm.float().to(hm.device)
            annotations['ind'][i] = ind
            annotations['reg_mask'][i] = reg_mask
            annotations['bboxes2d_target'][i] = bboxes2d_target
            annotations['cls_indexes'][i] = cls_indexes
        return annotations


    def _bbox2d_loss(self, output:torch.Tensor,
                           target:torch.Tensor)->torch.Tensor:
        pred_box = torch.cat([output[..., 0:2] * -1, output[..., 2:]], dim=-1)
        targ_box = torch.cat([target[..., 0:2] * -1, target[..., 2:]], dim=-1)
        loss = self.bbox2d_loss(pred_box, targ_box).sum()
        loss = loss / (len(output) + 1e-4)
        return loss

    def _laplacian_l1(self, output, target, uncertainty):
        loss = F.l1_loss(output, target, reduction='none') * torch.exp(-uncertainty) + \
            uncertainty * self.uncertainty_weight
        
        return loss.sum() / (len(output) + 1e-4)
    
    @staticmethod
    def _L1Loss(output, target):
        loss = F.l1_loss(output, target, reduction='none')
        return loss.sum() / (len(output) + 1e-4)

    def _gather_output(self, output, ind, mask, mask_for_2d=None):
        
        if mask_for_2d is None:
            mask_for_2d = mask
        bbox2d = _transpose_and_gather_feat(output['bbox2d'], ind)[mask_for_2d]
        rot = _transpose_and_gather_feat(output['rot'], ind)[mask]
        hps = _transpose_and_gather_feat(output['hps'], ind)[mask]
        offset = _transpose_and_gather_feat(output['reg'], ind)[mask]
        depth = _transpose_and_gather_feat(output['depth'], ind)[mask]
        dim = _transpose_and_gather_feat(output['dim'], ind)[mask]
        depth_uncer = _transpose_and_gather_feat(output['depth_uncertainty'], ind)[mask]
        corner_uncer = _transpose_and_gather_feat(output['corner_uncertainty'], ind)[mask]

        n, num_kps = hps.shape #[N, 20]
        hps = hps.reshape(n, num_kps//2, 2) #[N, 10, 2]

        flatten_reg_mask_gt = mask.view(-1).bool()
        batch_idxs = torch.arange(len(mask)).view(-1, 1).expand_as(mask).reshape(-1).cuda()
        batch_idxs = batch_idxs[flatten_reg_mask_gt].long().cuda()

        decoded_dict = dict(
            bbox2d = bbox2d,
            log_dim=dim,
            rot=rot,
            hps=hps,
            offset = offset,
            depth=depth,
            depth_uncer=depth_uncer,
            corner_uncer=corner_uncer,
            batch_idxs=batch_idxs
        )
        return decoded_dict

    def _keypoints_depth_loss(self, depths, target, validmask, uncertainty):

        loss = F.l1_loss(depths, target.repeat(1, 3), reduction='none') * torch.exp(-uncertainty) + \
            uncertainty * self.uncertainty_weight #[N, 3]
        
        valid_loss = loss * validmask.float() + (1 - validmask.float()) * loss.detach() #[N, 3]

        return valid_loss.mean(dim=1).sum() / (len(depths) + 1e-4)

    @staticmethod
    def merge_depth(depth, depth_uncer):
        pred_uncertainty_weights = 1 / depth_uncer
        pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
        depth = torch.sum(depth * pred_uncertainty_weights, dim=1)
        return depth
    
    def _decode(self, reg_preds, points):
        xs = points[..., 0] #[N]
        ys = points[..., 1] #[N]

        lefts   = xs - reg_preds[..., 0] #[N, ]
        tops    = ys - reg_preds[..., 1] #[N, ]
        rights  = xs + reg_preds[..., 2] #[N, ]
        bottoms = ys + reg_preds[..., 3] #[N, ]

        bboxes = torch.stack([lefts, tops, rights, bottoms], dim=-1)

        return bboxes

    def _decode_alpha(self, rot):
        alpha_idx = rot[..., 1] > rot[..., 5]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan(rot[..., 2] / rot[..., 3]) + (-0.5 * np.pi)
        alpha2 = torch.atan(rot[..., 6] / rot[..., 7]) + (0.5 * np.pi)
        alpha_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        return alpha_pre.unsqueeze(-1)
    
    def export_get_bboxes(self, output:dict, P2, img_batch=None):
        output['hm'] = torch.sigmoid(output['hm'])

        heat = _nms(output['hm'])
        scores, inds, clses, ys, xs = _topk(heat, K=100)

        gathered_output = self._gather_output(output, inds.long(), torch.ones_like(scores).bool())

        scores = scores[0] #[1, N] -> [N]
        clses = clses[0] #[1, N] -> [N]
        ys = ys[0] #[1, N] -> [N]
        xs = xs[0] #[1, N] -> [N]

        bbox2d = self._decode(gathered_output['bbox2d'], torch.stack([xs, ys], dim=1))

        gathered_output['dim'] = self.cls_mean_size[clses.long()] * torch.exp(gathered_output['log_dim'])
        gathered_output['depth_decoded'] = decode_depth_inv_sigmoid_calibration(gathered_output['depth'], P2[0, 0, 0], constant_scale=30.0)
        expanded_P2 = P2[gathered_output['batch_idxs'], :, :] #[N, 4, 4]
        gathered_output['kpd_depth'] = decode_depth_from_keypoints(gathered_output['hps'], gathered_output['dim'], expanded_P2) #[N, 3]
        gathered_output['depth_uncer'] = torch.clamp(gathered_output['depth_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])
        gathered_output['corner_uncer'] = torch.clamp(gathered_output['corner_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])

        pred_combined_uncertainty = torch.cat((gathered_output['depth_uncer'], gathered_output['corner_uncer']), dim=1).exp()
        pred_combined_depths = torch.cat((gathered_output['depth_decoded'], gathered_output['kpd_depth']), dim=1)
        gathered_output['merged_depth'] = self.merge_depth(pred_combined_depths, pred_combined_uncertainty)

        score_threshold = getattr(self.test_cfg, 'score_thr', 0.1)
        mask = scores > score_threshold#[K]
        bbox2d = bbox2d[mask]
        scores = scores[mask].unsqueeze(1) #[K, 1]
        dims   = gathered_output['dim'][mask] #[w, h, l] ?
        cls_indexes = clses[mask].long()
        alpha = self._decode_alpha(gathered_output['rot'][mask])

        cx3d = (xs[mask] + gathered_output['offset'][mask][..., 0]).unsqueeze(-1)
        cy3d = (ys[mask] + gathered_output['offset'][mask][..., 1]).unsqueeze(-1)
        z3d = gathered_output['merged_depth'][mask].unsqueeze(-1)  #[N, 1]
    
        ## upsample back
        bbox2d *= 4
        cx3d *= 4
        cy3d *= 4

        center3d = self.backprojector(
            torch.cat([cx3d, cy3d, z3d], dim=1), P2[0]
        ) # [N, 2] [x3d, y3d]
        theta = alpha2theta_3d(alpha, center3d[:, 0:1], center3d[:, 2:3], P2[0])

        if img_batch is not None:
            bbox2d = self.clipper(bbox2d, img_batch)

        bbox3d_3d = torch.cat(
            [bbox2d,  center3d, dims, alpha, theta], dim=1     #x3d, y3d, z, w, h, l, alpha, theta
        )

        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        

        if cls_agnostic:
            keep_inds = nms(bbox3d_3d[:, :4], scores[:, 0], nms_iou_thr)
        else:
            max_coordinate = bbox3d_3d.max()
            nms_bbox = bbox3d_3d[:, :4] + cls_indexes.float() * (max_coordinate)
            keep_inds = nms(nms_bbox, scores, nms_iou_thr)
            
        scores = scores[keep_inds, 0]
        bbox3d_3d = bbox3d_3d[keep_inds]
        cls_indexes = cls_indexes[keep_inds]
        return scores, bbox3d_3d, cls_indexes

    def get_bboxes(self, output:dict, P2, img_batch=None):
        output['hm'] = torch.sigmoid(output['hm'])

        heat = _nms(output['hm'])
        scores, inds, clses, ys, xs = _topk(heat, K=100)

        gathered_output = self._gather_output(output, inds.long(), torch.ones_like(scores).bool())

        scores = scores[0] #[1, N] -> [N]
        clses = clses[0] #[1, N] -> [N]
        ys = ys[0] #[1, N] -> [N]
        xs = xs[0] #[1, N] -> [N]

        bbox2d = self._decode(gathered_output['bbox2d'], torch.stack([xs, ys], dim=1))

        gathered_output['dim'] = self.cls_mean_size[clses.long()] * torch.exp(gathered_output['log_dim'])
        gathered_output['depth_decoded'] = decode_depth_inv_sigmoid_calibration(gathered_output['depth'], P2[0, 0, 0], constant_scale=30.0)
        expanded_P2 = P2[gathered_output['batch_idxs'], :, :] #[N, 4, 4]
        gathered_output['kpd_depth'] = decode_depth_from_keypoints(gathered_output['hps'], gathered_output['dim'], expanded_P2) #[N, 3]
        gathered_output['depth_uncer'] = torch.clamp(gathered_output['depth_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])
        gathered_output['corner_uncer'] = torch.clamp(gathered_output['corner_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])

        pred_combined_uncertainty = torch.cat((gathered_output['depth_uncer'], gathered_output['corner_uncer']), dim=1).exp()
        pred_combined_depths = torch.cat((gathered_output['depth_decoded'], gathered_output['kpd_depth']), dim=1)
        gathered_output['merged_depth'] = self.merge_depth(pred_combined_depths, pred_combined_uncertainty)

        score_threshold = getattr(self.test_cfg, 'score_thr', 0.1)
        mask = scores > score_threshold#[K]
        bbox2d = bbox2d[mask]
        scores = scores[mask].unsqueeze(1) #[K, 1]
        dims   = gathered_output['dim'][mask] #[w, h, l] ?
        cls_indexes = clses[mask].long()
        alpha = self._decode_alpha(gathered_output['rot'][mask])

        cx3d = (xs[mask] + gathered_output['offset'][mask][..., 0]).unsqueeze(-1)
        cy3d = (ys[mask] + gathered_output['offset'][mask][..., 1]).unsqueeze(-1)
        z3d = gathered_output['merged_depth'][mask].unsqueeze(-1)  #[N, 1]
    
        ## upsample back
        bbox2d *= 4
        cx3d *= 4
        cy3d *= 4

        center3d = self.backprojector(
            torch.cat([cx3d, cy3d, z3d], dim=1), P2[0]
        ) # [N, 2] [x3d, y3d]
        theta = alpha2theta_3d(alpha, center3d[:, 0:1], center3d[:, 2:3], P2[0])

        if img_batch is not None:
            bbox2d = self.clipper(bbox2d, img_batch)

        bbox3d_3d = torch.cat(
            [bbox2d,  center3d, dims, alpha, theta], dim=1     #x3d, y3d, z, w, h, l, alpha, theta
        )

        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        

        if cls_agnostic:
            keep_inds = nms(bbox3d_3d[:, :4], scores[:, 0], nms_iou_thr)
        else:
            max_coordinate = bbox3d_3d.max()
            nms_bbox = bbox3d_3d[:, :4] + cls_indexes.float() * (max_coordinate)
            keep_inds = nms(nms_bbox, scores, nms_iou_thr)
            
        scores = scores[keep_inds, 0]
        bbox3d_3d = bbox3d_3d[keep_inds]
        cls_indexes = cls_indexes[keep_inds]
        cls_names = [self.learned_classes[cls_index] for cls_index in cls_indexes]
        return scores, bbox3d_3d, cls_names

    def loss(self, output, annotations, meta, P2):
        # epoch = meta['epoch']
        # annotations = self.update_2d_pseudo_label(output, annotations)

        annotations['ind'] = annotations['ind'].long()
        annotations['reg_mask'] = annotations['reg_mask'].bool() # [B, N]
        annotations['3D_reg_mask'] = annotations['reg_mask'] * annotations['is_labeled_3d'][:, None] #[B, N]
        # heatmap center loss
        label_mask = annotations['labeled_mask']
        hm_loss = self._neg_loss(output['hm'][label_mask], annotations['hm'][label_mask])
        # keypoint L1 loss
        hp_loss = self._RegWeightedL1Loss(output['hps'],annotations['hps_mask'], annotations['ind'], annotations['hps'],annotations['dep'].clone())
        # rotations from RTM3D
        rot_loss = self._RotLoss(output['rot'], annotations['3D_reg_mask'], annotations['ind'], annotations['rotbin'], annotations['rotres'])

        # gather output
        gathered_output = self._gather_output(output, annotations['ind'], annotations['3D_reg_mask'], annotations['reg_mask'])
        gathered_output['dim'] = annotations['mean_dim'][annotations['3D_reg_mask']] * torch.exp(gathered_output['log_dim'])
        expanded_P2 = P2[gathered_output['batch_idxs'], :, :] #[N, 3, 4]
        gathered_output['depth_decoded'] = decode_depth_inv_sigmoid_calibration(gathered_output['depth'], expanded_P2[:, 0:1, 0], constant_scale=30.0)
        gathered_output['kpd_depth'] = decode_depth_from_keypoints(gathered_output['hps'], gathered_output['dim'], expanded_P2) #[N, 3]
        gathered_output['depth_uncer'] = torch.clamp(gathered_output['depth_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])
        gathered_output['corner_uncer'] = torch.clamp(gathered_output['corner_uncer'], self.uncertainty_range[0], self.uncertainty_range[1])

        pred_combined_uncertainty = torch.cat((gathered_output['depth_uncer'], gathered_output['corner_uncer']), dim=1).exp()
        pred_combined_depths = torch.cat((gathered_output['depth_decoded'], gathered_output['kpd_depth']), dim=1)
        gathered_output['merged_depth'] = self.merge_depth(pred_combined_depths, pred_combined_uncertainty)

        # FCOS style regression
        box2d_loss = self._bbox2d_loss(gathered_output['bbox2d'], annotations['bboxes2d_target'][annotations['reg_mask']])
        # dimensions
        dim_loss = self._L1Loss(gathered_output['log_dim'], annotations['log_dim'][annotations['3D_reg_mask']])
        # offset for center heatmap
        off_loss = self._L1Loss(gathered_output['offset'], annotations['reg'][annotations['3D_reg_mask']])

        # direct depth regression
        depth_loss = self._laplacian_l1(gathered_output['depth_decoded'], annotations['dep'][annotations['3D_reg_mask']], gathered_output['depth_uncer'])

        keypoint_depth_loss = self._keypoints_depth_loss(gathered_output['kpd_depth'], annotations['dep'][annotations['3D_reg_mask']],
                                                         annotations['kp_detph_mask'][annotations['3D_reg_mask']], gathered_output['corner_uncer'])
        soft_depth_loss = self._L1Loss(gathered_output['merged_depth'].unsqueeze(-1), annotations['dep'][annotations['3D_reg_mask']])


        loss_stats = {'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'box2d_loss': box2d_loss, 'off_loss': off_loss,'dim_loss': dim_loss,
                      'depth_loss': depth_loss, 'kpd_loss': keypoint_depth_loss,
                      'rot_loss':rot_loss, 'soft_depth_loss': soft_depth_loss}

        weight_dict = self.loss_weight

        loss = 0
        for key, weight in weight_dict.items():
            if key in loss_stats:
                loss = loss + loss_stats[key] * weight
                loss_stats[key] = loss_stats[key].detach()
        loss_stats['total_loss'] = loss.detach()
        return loss, loss_stats


class CenterNet2DHead(MonoFlexHead):
    def _decode(self, heat, reg, wh, K=100, **kwargs):

        batch, cat, height, width = heat.size()
        # num_joints = kps.shape[1] // 2
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        # hm_show,_=torch.max(hm_hp,1)
        # hm_show=hm_show.squeeze(0)
        # hm_show=hm_show.detach().cpu().numpy().copy()
        # plt.imshow(hm_show, 'gray')
        # plt.show()

        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)

        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)

        bboxes *= 4 # restore back to scale 1
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections


    def get_bboxes(self, output:dict, P2=None, img_batch=None):
        output['hm'] = torch.sigmoid(output['hm'])
        reg = output['reg']
        dets = self._decode(
            output['hm'], reg, output['wh'], K=100
        )[0]

        score_threshold = getattr(self.test_cfg, 'score_thr', 0.1)
        mask = dets[:, 4] > score_threshold#[K]
        bbox2d = dets[mask, 0:4]
        scores = dets[mask, 4:5] #[K, 1]
        cls_indexes = dets[mask, 5:6].long()
        
        if img_batch is not None:
            bbox2d = self.clipper(bbox2d, img_batch)
        
        cls_agnostic = getattr(self.test_cfg, 'cls_agnositc', True) # True -> directly NMS; False -> NMS with offsets different categories will not collide
        nms_iou_thr  = getattr(self.test_cfg, 'nms_iou_thr', 0.5)
        

        if cls_agnostic:
            keep_inds = nms(bbox2d[:, :4], scores[:, 0], nms_iou_thr)
        else:
            max_coordinate = bbox2d.max()
            nms_bbox = bbox2d[:, :4] + cls_indexes.float() * (max_coordinate)
            keep_inds = nms(nms_bbox, scores, nms_iou_thr)
            
        scores = scores[keep_inds, 0]
        bbox2d = bbox2d[keep_inds]
        cls_indexes = cls_indexes[keep_inds]
        
        return scores, bbox2d, cls_indexes

    def loss(self, output, annotations, meta=dict()):

        hm_loss = self._neg_loss(output['hm'], annotations['hm'])

        wh_loss = self._RegL1Loss(output['wh'], annotations['reg_mask'],annotations['ind'], annotations['wh'])

        off_loss = self._RegL1Loss(output['reg'], annotations['reg_mask'], annotations['ind'], annotations['reg'])

        loss_stats = {'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}

        weight_dict = {'hm_loss': 1, 'wh_loss': 0.1, 'off_loss': 1}

        loss = 0
        for key, weight in weight_dict.items():
            if key in loss_stats:
                loss = loss + loss_stats[key] * weight
                loss_stats[key] = loss_stats[key].detach()
        loss_stats['total_loss'] = loss.detach()
        return loss, loss_stats
