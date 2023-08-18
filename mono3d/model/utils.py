import torch
import torch.nn as nn
import numpy as np
def theta2alpha_3d(theta, x, z, P2):
    """ Convert theta to alpha with 3D position
    Args:
        theta [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        alpha []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(theta, torch.Tensor):
        alpha = theta - torch.atan2(x + offset, z)
    else:
        alpha = theta - np.arctan2(x + offset, z)
    return alpha

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        theta = alpha + torch.atan2(x + offset, z)
    else:
        theta = alpha + np.arctan2(x + offset, z)
    return theta


class BBox3dProjector(nn.Module):
    """
        forward methods
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
    """
    def __init__(self):
        super(BBox3dProjector, self).__init__()
        self.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]]
        ).float()  )# 8, 3

    def forward(self, bbox_3d, tensor_p2):
        """
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
        """
        relative_eight_corners = 0.5 * self.corner_matrix * bbox_3d[:, 3:6].unsqueeze(1)  # [N, 8, 3]
        # [batch, N, ]
        thetas = alpha2theta_3d(bbox_3d[..., 6], bbox_3d[..., 0], bbox_3d[..., 2], tensor_p2)
        _cos = torch.cos(thetas).unsqueeze(1)  # [N, 1]
        _sin = torch.sin(thetas).unsqueeze(1)  # [N, 1]
        rotated_corners_x, rotated_corners_z = (
            relative_eight_corners[:, :, 2] * _cos + relative_eight_corners[:, :, 0] * _sin,
            - relative_eight_corners[:, :, 2] * _sin + relative_eight_corners[:, :, 0] * _cos
        )  # relative_eight_corners == [N, 8, 3]
        rotated_corners = torch.stack([rotated_corners_x, relative_eight_corners[:,:,1], rotated_corners_z], dim=-1) #[N, 8, 3]
        abs_corners = rotated_corners + \
            bbox_3d[:, 0:3].unsqueeze(1)  # [N, 8, 3]
        camera_corners = torch.cat([abs_corners,
            abs_corners.new_ones([abs_corners.shape[0], self.corner_matrix.shape[0], 1])],
            dim=-1).unsqueeze(3)  # [N, 8, 4, 1]
        camera_coord = torch.matmul(tensor_p2, camera_corners).squeeze(-1)  # [N, 8, 3]

        homo_coord = camera_coord / (camera_coord[:, :, 2:] + 1e-6) # [N, 8, 3]

        return abs_corners, homo_coord, thetas

class BackProjection(nn.Module):
    """
        forward method:
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
    """
    def forward(self, bbox3d, p2):
        """
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
        """
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]

        z3d = bbox3d[:, 2:3] #[N, 1]
        x3d = (bbox3d[:,0:1] * z3d - cx * z3d - tx) / fx #[N, 1]
        y3d = (bbox3d[:,1:2] * z3d - cy * z3d - ty) / fy #[N, 1]
        return torch.cat([x3d, y3d, bbox3d[:, 2:]], dim=1)


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def compute_occlusion(box2d, z):
    """
        Compute intersection / area_2d, max with mask z > zi.
        Args:
            box2d: torch.tensor[N, 4] -> [x1, y1, x2, y2]
            z:     torch.tensor[N, ]
        Return:
            occlusions: torch.Tensor[N, ]
    """
    area = (box2d[:, 2] - box2d[:, 0]) * (box2d[:, 3] - box2d[:, 1]) #[N, ]

    iw = torch.min(torch.unsqueeze(box2d[:, 2], dim=1), box2d[:, 2]) - torch.max(torch.unsqueeze(box2d[:, 0], 1), box2d[:, 0])
    ih = torch.min(torch.unsqueeze(box2d[:, 3], dim=1), box2d[:, 3]) - torch.max(torch.unsqueeze(box2d[:, 1], 1), box2d[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    intersection = iw * ih # [N, N]

    inter_over_area = intersection / (area[:, None] + 1e-4) #[N, N]   inter_over_area[i, :] = all intersection / area[i]  [N, ]

    eps = 1e-4
    z_mask = (z.reshape(-1, 1) - z.reshape(1, -1)) > eps # z_mask[i, j] == True -> z[i] > z[j]
    inter_over_area = torch.where(
        z_mask,
        inter_over_area,
        torch.zeros_like(inter_over_area)
    )
    occlusion, _ = torch.max(inter_over_area, dim=1)
    return occlusion


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)

        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)
      
        return boxes
