from segmentation_utils.labels import PALETTE
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def ColorizeSeg(pred_seg, rgb_image, opacity=1.0, palette=PALETTE):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    h, w = pred_seg.shape
    for i in range(h):
        for j in range(w):
            color_seg[i, j] = palette[pred_seg[i, j]]
    new_image = rgb_image * (1 - opacity) + color_seg * opacity
    new_image = new_image.astype(np.uint8)
    return new_image


@jit(nopython=True, cache=True)
def Colorizevoxel(semantic):
    """ Colorize Voxels
    Args:
        semantic: np.array, length shape [N]
    Return:
        return np.array, shape [B, N, 3] colorized semantics
    """
    N  = semantic.shape[0]
    color_semantic = np.zeros((N, 3), dtype=np.uint8)

    for n in range(N):
        color_semantic[n] = PALETTE[semantic[n]]
    return color_semantic


