import sys
import os
import numpy as np
import numba
from fire import Fire
import tqdm
import cv2

sys.path.append(os.getcwd())
from segmentation.bdd100k import original_int_type as bdd100k_label_dict
from segmentation.evaluation.labels import labels as kitti360_labels

def find_similar_label(bdd100k_label, kitti360_labels):
    name:str = bdd100k_label

    ## Completely same
    for kitti360_label in kitti360_labels:
        if kitti360_label.name == name:
            return name, kitti360_label
    ## Similar
    for kitti360_label in kitti360_labels:
        if kitti360_label.name in name:
            return name, kitti360_label
    return name, None

@numba.jit(nopython=True)
def remapping_label(original_seg, mapping):
    remapped_seg = np.zeros((original_seg.shape[0], original_seg.shape[1]), dtype=np.uint8)
    h, w = original_seg.shape
    for i in range(h):
        for j in range(w):
            remapped_seg[i, j] = mapping[original_seg[i, j]]

    return remapped_seg

def remap_bdd100k_to_kitti360(base_path="/data/bdd100k", output_path="/home/remapped_bdd100k"):
    os.makedirs(output_path, exist_ok=True)
    mapping = {}
    for label in bdd100k_label_dict:
        extracted_name, similar_label = find_similar_label(bdd100k_label_dict[label], kitti360_labels)
        mapping[label] = similar_label.id if similar_label is not None else -1
        if similar_label is None:
            print("Not found similar label for {}/{}".format(bdd100k_label_dict[label], extracted_name))
    
    numpy_mapping = np.zeros((256), dtype=np.uint8)
    for k, v in mapping.items():
        numpy_mapping[k] = v

    label_dir = os.path.join(base_path, "labels", 'sem_seg', 'masks')
    output_dir = os.path.join(output_path, "labels", 'sem_seg', 'remapped_masks')
    splits = os.listdir(label_dir)
    for split in splits:
        split_path = os.path.join(label_dir, split)
        if not os.path.isdir(split_path):
            continue
        images = os.listdir(split_path)
        print("Remapping {} images in {}".format(len(images), split_path))
        for image in tqdm.tqdm(images):
            image_path = os.path.join(split_path, image)
            if not os.path.isfile(image_path):
                continue
            output_image_path = os.path.join(output_dir, split, image)
            original_seg = cv2.imread(image_path, -1)
            remapped_seg = remapping_label(original_seg, numpy_mapping)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, remapped_seg)

if __name__ == "__main__":
    Fire(remap_bdd100k_to_kitti360)