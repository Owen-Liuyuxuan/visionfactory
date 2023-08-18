from typing import List, Tuple, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F
def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

def MonoFlexMultiBin_loss(vector_ori, gt_ori, num_bin=4):
    gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    for i in range(num_bin):
        #bin cls loss
        cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_bin * 2 + i * 2
            e = s + 2
            pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
            reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
                        F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

    return cls_losses / num_bin + reg_losses / reg_cnt

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def compute_radius(det_size, min_overlap=0.7):
    height, width = det_size[0], det_size[1]

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    return r2

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def gen_hm_radius(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d.astype(np.int)

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep

def decode_depth_inv_sigmoid(depth:torch.Tensor)->torch.Tensor:
    """Decode depth from network prediction to 3D depth

    Args:
        depth (torch.Tensor): depth from network prediction (un-activated)

    Returns:
        torch.Tensor: 3D depth for output
    """
    depth_decoded = torch.exp(-depth) #1 / torch.sigmoid(depth) - 1
    return depth_decoded

def encode_depth_inv_sigmoid(depth_decoded:torch.Tensor)->torch.Tensor:
    """Decode depth from network prediction to 3D depth

    Args:
        depth_decoded (torch.Tensor): depth from network prediction (un-activated)

    Returns:
        torch.Tensor: 3D depth for output
    """
    if isinstance(depth_decoded, torch.Tensor):
        depth = - torch.log(depth_decoded)
    if isinstance(depth_decoded, np.ndarray):
        depth = - np.log(depth_decoded)
    return depth

def decode_depth_inv_sigmoid_calibration(depth:torch.Tensor, fx:torch.Tensor, constant_scale:Union[float, torch.Tensor]=5)->torch.Tensor:
    """Decode depth from network prediction to 3D depth

    Args:
        depth (torch.Tensor): depth from network prediction (un-activated)
        fx (torch.Tensor): fx in calibration matrix
        constant_scale (float or torch.Tensor): scale factor for inv_sigmoid

    Returns:
        torch.Tensor: 3D depth for output
    """
    inv_sigmoid = decode_depth_inv_sigmoid(depth)
    depth_decoded = fx / (inv_sigmoid * constant_scale)
    return depth_decoded

def encode_depth_inv_sigmoid_calibration(depth_decoded:torch.Tensor, fx:torch.Tensor, constant_scale:Union[float, torch.Tensor]=5) -> torch.Tensor:
    """Encode depth into network prediction

    Args:
        depth_decoded (torch.Tensor): decoded depth
        fx (torch.Tensor): fx in calibration matrix
        constant_scale (Union[float, torch.Tensor], optional): scale factor for inv_sigmoid. Defaults to 5.

    Returns:
        torch.Tensor: network prediction gt.
    """
    inv_sigmoid = fx / depth_decoded / constant_scale
    depth = encode_depth_inv_sigmoid(inv_sigmoid)
    return depth

def decode_depth_from_keypoints(keypoints:torch.Tensor,
                                dimensions:torch.Tensor,
                                calib:torch.Tensor,
                                down_ratio:int=4,
                                group0_index:List[Tuple[int, int]]=[(7, 3), (0, 4)],
                                group1_index:List[Tuple[int, int]]=[(2, 6), (1, 5)],
                                min_depth:float=0.1,
                                max_depth:float=100,
                                EPS:float=1e-8)->torch.Tensor:
    """Decode depth from keypoints according to MonoFlex

    Args:
        keypoints (torch.Tensor): Tensor of shape [*, 10, 2], 8 vertices + top/bottom
        dimensions (torch.Tensor): Tensor of shape [*, 3], whl
        calibs (torch.Tensor): Calibration matrix P2 [*, 4, 4]
        down_ratio (int, optional): Down sample ratio of the predicted keypoints. Defaults to 4
        group0_index (List[Tuple[int, int]], optional): Group of index. Defaults to [0, 3, 4, 7].
        group1_index (List[Tuple[int, int]], optional): Group of index for depth 2. Defaults to [1, 2, 5, 6].
        min_depth (float, optional): min depth prediction. Defaults to 0.1
        max_depth (float, optional): max depth prediction. Defaults to 100
        EPS (float, optional): Small numbers. Defaults to 1e-8

    Returns:
        torch.Tensor: [*, 3]  depth computed from three groups of keypoints (top/bottom, group0, group1)
    """
    pred_height_3D = dimensions[..., 1].detach() #[*]

    center_height = keypoints[..., -2, 1] - keypoints[:, -1, 1] #[*]
    corner_02_height = keypoints[..., group0_index[0], 1] - keypoints[..., group0_index[1], 1] #[*, 2]
    corner_13_height = keypoints[..., group1_index[0], 1] - keypoints[..., group1_index[1], 1] #[*, 2]

    f = calib[..., 0, 0] #[*]
    center_depth = f * pred_height_3D / (F.relu(center_height) * down_ratio + EPS) #[*]
    corner_02_depth = (f * pred_height_3D).unsqueeze(-1) / (F.relu(corner_02_height) * down_ratio + EPS) #[*, 2]
    corner_13_depth = (f * pred_height_3D).unsqueeze(-1) / (F.relu(corner_13_height) * down_ratio + EPS) #[*, 2]

    corner_02_depth = corner_02_depth.mean(dim=1)
    corner_13_depth = corner_13_depth.mean(dim=1)

    depths = torch.stack([center_depth, corner_02_depth, corner_13_depth], dim=-1)
    depths = torch.clamp(depths, min_depth, max_depth)
    return depths

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def draw_bev_mask(mask, x, z, x_bounds, z_bounds, w, l, theta, cls):
    bev_corners_x = [x + l / 2 * np.cos(theta) - w / 2 * np.sin(theta),
                     x + l / 2 * np.cos(theta) + w / 2 * np.sin(theta),
                     x - l / 2 * np.cos(theta) + w / 2 * np.sin(theta),
                     x - l / 2 * np.cos(theta) - w / 2 * np.sin(theta),
                    ]
    bev_corners_z = [z - l / 2 * np.sin(theta) - w / 2 * np.cos(theta),
                     z - l / 2 * np.sin(theta) + w / 2 * np.cos(theta),
                     z + l / 2 * np.sin(theta) + w / 2 * np.cos(theta),
                     z + l / 2 * np.sin(theta) - w / 2 * np.cos(theta),
                    ]
    bev_corners_x = np.array(bev_corners_x)
    bev_corners_z = np.array(bev_corners_z)
    bev_corners_x_index = np.array((bev_corners_x - x_bounds[0]) / x_bounds[2], dtype=np.int64)
    bev_corners_z_index = np.array((bev_corners_z - z_bounds[0]) / z_bounds[2], dtype=np.int64)
    bev_points = np.stack([bev_corners_z_index, bev_corners_x_index], axis=-1) #[4, 2], (z, x) (axis=1, axis=0)
    cv2.fillPoly(mask, pts=[bev_points], color=cls)
    return mask

def gen_position(kps,dim,rot,meta,const):
    """ Decode rotation and generate position. Notice that
    unlike the official implementation, we do not transform back to pre-augmentation images.
    And we also compenstate for the offset in camera in this function.

    We also change the order of the keypoints to the default projection order in this repo,
    therefore the way we construct least-square matrix also changed.

    Args:
        kps [torch.Tensor]: [B, C, 9, 2], keypoints relative offset from the center_int in augmented scale 4. network prediction.
        dim [torch.Tensor]: [B, C, 3], width/height/length, the order is different.
        rot [torch.Tensor]: [B, C, 8], rotation prediction from the network.
        meta [Dict]: meta['calib'].shape = [B, 3, 4] -> calibration matrix for augmented images.
        const [torch.Tensor]: const.shape = [1, 1, 16], constant helping parameter used in optimization.
    Returns:
        position [torch.Tensor]: [B, C, 3], 3D position.
        rot_y [torch.Tensor]: [B, C, 1], 3D rotation theta. Decoded.
        alpna_pre [torch.Tensor]: [B, C, 1], observation angle alpha decoded. The typo is consistent with the official typo.
        kps [torch.Tensor]: [B, C, 18], basically same with the input (not transformed here).
    """
    b=kps.size(0)
    c=kps.size(1)
    calib=meta['calib']

    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]

    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)

    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = kps[:, :, :16]
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f

    l = dim[:, :, 2:3]
    h = dim[:, :, 1:2]
    w = dim[:, :, 0:1]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)

    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)

    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    Tx_fx = (calib[..., 0, 3] / calib[..., 0, 0]).unsqueeze(-1) # [B, C, 1]
    B[:, :, 0:1]    = - l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 1:2]    = - h * 0.5
    B[:, :, 2:3]    = - l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 3:4]    = - h * 0.5
    B[:, :, 4:5]    = - l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 5:6]    =   h * 0.5
    B[:, :, 6:7]    =   l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 7:8]    =   h * 0.5
    B[:, :, 8:9]    =   l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 9:10]   = - h * 0.5
    B[:, :, 10:11]  =   l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 11:12]  = - h * 0.5
    B[:, :, 12:13]  =   l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 13:14]  =   h * 0.5
    B[:, :, 14:15]  = - l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 15:16]  =   h * 0.5

    C[:, :, 0:1]    =   l * 0.5 * sinori - w * 0.5 * cosori # - l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 1:2]    =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 2:3]    =   l * 0.5 * sinori + w * 0.5 * cosori # - l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 3:4]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 4:5]    =   l * 0.5 * sinori + w * 0.5 * cosori # - l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 5:6]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 6:7]    = - l * 0.5 * sinori + w * 0.5 * cosori # l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 7:8]    = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9]    = - l * 0.5 * sinori + w * 0.5 * cosori # l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 9:10]   = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11]  = - l * 0.5 * sinori - w * 0.5 * cosori # l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 11:12]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13]  = - l * 0.5 * sinori - w * 0.5 * cosori #  l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 13:14]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15]  =   l * 0.5 * sinori - w * 0.5 * cosori # - l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 15:16]  =   l * 0.5 * sinori - w * 0.5 * cosori

    B = B - kp_norm * C

    # A=A*kps_mask1
    A = A.double()
    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()
    # mask = mask.unsqueeze(2)
    
    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv + torch.randn_like(pinv) * 1e-8)  # b*c 3 3
    pinv = torch.bmm(pinv, AT).float()
    pinv = torch.bmm(pinv, B)
    
    pinv = pinv.view(b, c, 3, 1).squeeze(3)
        
    return pinv,rot_y,alpna_pre, kps

def gen_position_direct_alpha(kps,dim,rot,meta):
    """ Decode rotation and generate position. Notice that
    unlike the official implementation, we do not transform back to pre-augmentation images.
    And we also compenstate for the offset in camera in this function.

    We also change the order of the keypoints to the default projection order in this repo,
    therefore the way we construct least-square matrix also changed.

    Args:
        kps [torch.Tensor]: [B, C, 9, 2], keypoints relative offset from the center_int in augmented scale 4. network prediction.
        dim [torch.Tensor]: [B, C, 3], width/height/length, the order is different.
        rot [torch.Tensor]: [B, C, 1], rotation prediction / alpha from the network.
        meta [Dict]: meta['calib'].shape = [B, 3, 4] -> calibration matrix for augmented images.
        const [torch.Tensor]: const.shape = [1, 1, 16], constant helping parameter used in optimization.
    Returns:
        position [torch.Tensor]: [B, C, 3], 3D position.
        rot_y [torch.Tensor]: [B, C, 1], 3D rotation theta. Decoded.
        alpna_pre [torch.Tensor]: [B, C, 1], observation angle alpha decoded. The typo is consistent with the official typo.
        kps [torch.Tensor]: [B, C, 18], basically same with the input (not transformed here).
    """
    const = torch.Tensor(
                [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
                [-1, 0], [0, -1], [-1, 0], [0, -1]]).unsqueeze(0).unsqueeze(0).cuda()

    b=kps.size(0)
    c=kps.size(1)
    calib=meta['calib']

    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]

    rot_y = rot + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = kps[:, :, :16]
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f

    l = dim[:, :, 2:3]
    h = dim[:, :, 1:2]
    w = dim[:, :, 0:1]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)

    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)

    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    Tx_fx = (calib[..., 0, 3] / calib[..., 0, 0]).unsqueeze(-1) # [B, C, 1]
    B[:, :, 0:1]    = - l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 1:2]    = - h * 0.5
    B[:, :, 2:3]    = - l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 3:4]    = - h * 0.5
    B[:, :, 4:5]    = - l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 5:6]    =   h * 0.5
    B[:, :, 6:7]    =   l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 7:8]    =   h * 0.5
    B[:, :, 8:9]    =   l * 0.5 * cosori + w * 0.5 * sinori + Tx_fx
    B[:, :, 9:10]   = - h * 0.5
    B[:, :, 10:11]  =   l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 11:12]  = - h * 0.5
    B[:, :, 12:13]  =   l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 13:14]  =   h * 0.5
    B[:, :, 14:15]  = - l * 0.5 * cosori - w * 0.5 * sinori + Tx_fx
    B[:, :, 15:16]  =   h * 0.5

    C[:, :, 0:1]    =   l * 0.5 * sinori - w * 0.5 * cosori # - l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 1:2]    =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 2:3]    =   l * 0.5 * sinori + w * 0.5 * cosori # - l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 3:4]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 4:5]    =   l * 0.5 * sinori + w * 0.5 * cosori # - l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 5:6]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 6:7]    = - l * 0.5 * sinori + w * 0.5 * cosori # l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 7:8]    = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9]    = - l * 0.5 * sinori + w * 0.5 * cosori # l * 0.5 * cosori + w * 0.5 * sinori
    C[:, :, 9:10]   = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11]  = - l * 0.5 * sinori - w * 0.5 * cosori # l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 11:12]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13]  = - l * 0.5 * sinori - w * 0.5 * cosori #  l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 13:14]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15]  =   l * 0.5 * sinori - w * 0.5 * cosori # - l * 0.5 * cosori - w * 0.5 * sinori
    C[:, :, 15:16]  =   l * 0.5 * sinori - w * 0.5 * cosori

    B = B - kp_norm * C

    # A=A*kps_mask1
    A = A.double()
    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()
    # mask = mask.unsqueeze(2)
    
    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv + torch.randn_like(pinv) * 1e-8)  # b*c 3 3
    pinv = torch.bmm(pinv, AT).float()
    pinv = torch.bmm(pinv, B)
    
    pinv = pinv.view(b, c, 3, 1).squeeze(3)
        
    return pinv

def stereo_optimized_position(l_kps, r_kps, P2, P3, dim, rot):
    """ Decode rotation and generate position. Notice that
    unlike the official implementation, we do not transform back to pre-augmentation images.
    And we also compenstate for the offset in camera in this function.

    We also change the order of the keypoints to the default projection order in this repo,
    therefore the way we construct least-square matrix also changed.

    Args:
        l_kps [torch.Tensor]: [B, C, 16+], keypoints position on left image.
        r_kps [torch.Tensor]: [B, C, 16+], keypoints position on right image.
        dim [torch.Tensor]: [B, C, 3], width/height/length, the order is different from the official implementation.
        rot [torch.Tensor]: [B, C, 1], 3D rotation of the object.
    Returns:
        position [torch.Tensor]: [B, C, 3], 3D position.
    """
    b=l_kps.size(0)
    c=l_kps.size(1)

    l_calib = P2.unsqueeze(1)
    l_calib = l_calib.expand(b, c, -1, -1).contiguous()
    r_calib = P3.unsqueeze(1)
    r_calib = r_calib.expand(b, c, -1, -1).contiguous()

    # form left kp_norm (kpx - cx) / fx
    l_kpoint = l_kps[:, :, :16] #[B, C, 16]
    f = l_calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(l_kpoint)
    l_cx, l_cy = l_calib[:, :, 0, 2].unsqueeze(2), l_calib[:, :, 1, 2].unsqueeze(2)
    l_cxy = torch.cat((l_cx, l_cy), dim=2)
    l_cxy = l_cxy.repeat(1, 1, 8)  # b,c,16
    l_kp_norm = (l_kpoint[:] - l_cxy) / f

    # form right kp_norm
    r_kpoint = r_kps[:, :, :16] #[B, C, 16]
    r_cx, r_cy = r_calib[:, :, 0, 2].unsqueeze(2), r_calib[:, :, 1, 2].unsqueeze(2)
    r_cxy = torch.cat((r_cx, r_cy), dim=2)
    r_cxy = r_cxy.repeat(1, 1, 8)  # b,c,16
    r_kp_norm = (r_kpoint[:] - r_cxy) / f

    # concat
    kp_norm = torch.cat([l_kp_norm, r_kp_norm], dim=-1) #[b, c, 32]


    w = dim[:, :, 0:1]
    h = dim[:, :, 1:2]
    l = dim[:, :, 2:3]
    cosori = torch.cos(rot)
    sinori = torch.sin(rot)

    B = torch.zeros_like(kp_norm)
    C = torch.zeros_like(kp_norm)

    kp = kp_norm.unsqueeze(3)  # b,c,32,1
    const = kp.new_zeros([b, c, 32, 2])
    const[..., ::2, 0] = -1
    const[..., 1::2, 1] = -1
    A = torch.cat([const, kp], dim=3) #[b, c, 32, 3]

    l_Tx_fx = (l_calib[..., 0, 3] / l_calib[..., 0, 0]).unsqueeze(-1) # [B, C, 1]
    r_Tx_fx = (r_calib[..., 0, 3] / r_calib[..., 0, 0]).unsqueeze(-1) # [B, C, 1]
    B[:, :, 0:1]    = - l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 1:2]    = - h * 0.5
    B[:, :, 2:3]    = - l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 3:4]    = - h * 0.5
    B[:, :, 4:5]    = - l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 5:6]    =   h * 0.5
    B[:, :, 6:7]    =   l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8]    =   h * 0.5
    B[:, :, 8:9]    =   l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10]   = - h * 0.5
    B[:, :, 10:11]  =   l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12]  = - h * 0.5
    B[:, :, 12:13]  =   l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14]  =   h * 0.5
    B[:, :, 14:15]  = - l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 15:16]  =   h * 0.5
    B[:, :, 16:32]  =  B[:, :, 0:16].clone()
    B[:, :, 0:16:2] += l_Tx_fx
    B[:, :, 16:32:2]+= r_Tx_fx

    C[:, :, 0:1]    =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 1:2]    =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 2:3]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 3:4]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 4:5]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 5:6]    =   l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 6:7]    = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8]    = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9]    = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10]   = - l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14]  = - l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15]  =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 15:16]  =   l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 16:32] = C[:, :, 0:16].clone()

    B = B - kp_norm * C

    # A=A*kps_mask1
    A = A.double()
    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 32)
    A = A.view(b * c, 32, 3)
    B = B.view(b * c, 32, 1).float()
    # mask = mask.unsqueeze(2)
    
    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv + torch.randn_like(pinv) * 1e-8)  # b*c 3 3
    pinv = torch.bmm(pinv, AT).float()
    pinv = torch.bmm(pinv, B)
    
    pinv = pinv.view(b, c, 3, 1).squeeze(3)
    return pinv
