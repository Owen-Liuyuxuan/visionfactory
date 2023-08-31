"""
Convert Panoptic BEV labels to KITTI360 official Segmentation Label
"""
import umsgpack
import numpy as np
import cv2
import os
import json
from PIL import Image

KITTI360_panoptic_dir = "/data/kitti360_panopticbev"

meta_file = f"{KITTI360_panoptic_dir}/metadata_ortho.bin"
list_file = f"{KITTI360_panoptic_dir}/split/train.txt"
front_mask_dir = f"{KITTI360_panoptic_dir}/front_msk_trainid/front"
bev_mask_dir = f"{KITTI360_panoptic_dir}/bev_msk/bev_ortho"
class_weights = f"{KITTI360_panoptic_dir}/class_weights"
img_json_file = f"{KITTI360_panoptic_dir}/img/front.json"
output_dir = f"{KITTI360_panoptic_dir}/bev_msk/original_id"

with open(meta_file, "rb") as fid:
    metadata = umsgpack.unpack(fid, encoding="utf-8")
    
panoptic_id2original = np.zeros(256, dtype=np.uint8)
for i, original_id in enumerate(metadata['meta']['original_ids']):
    panoptic_id2original[i] = original_id
panoptic_id2original[6] = 0

with open(list_file, "r") as fid:
    lst = fid.readlines()
    lst = [line.strip() for line in lst]

os.makedirs(output_dir, exist_ok=True)

from numba import jit
import tqdm
@jit
def convert_to_original(bev_mask, img_desc_cat, id_mapper):
    h, w = bev_mask.shape
    output_bev_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output_bev_mask[i, j] = id_mapper[img_desc_cat[bev_mask[i, j]]]
    return output_bev_mask

for img_desc in tqdm.tqdm(metadata["images"]):
    bev_msk_file = os.path.join(bev_mask_dir, "{}.png".format(img_desc['id']))
    cat = np.array(img_desc['cat'])
    if os.path.isfile(bev_msk_file):
        bev_msk = np.array(Image.open(bev_msk_file)) #[H, W]
        original_id_mask = convert_to_original(bev_msk, cat, panoptic_id2original)
    cv2.imwrite(os.path.join(output_dir, "{}.png".format(img_desc['id'])), original_id_mask)
