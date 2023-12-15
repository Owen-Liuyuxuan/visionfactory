import sys
import os
import numpy as np
import numba
from fire import Fire
import tqdm
import cv2
import json

from segmentation.data_utils.labels_bdd100k import original_int_type as bdd100k_label_dict
sys.path.append(os.getcwd())

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
    cls_names = ['unlabeled']
    cls_index = 1
    for label in bdd100k_label_dict:
        if bdd100k_label_dict[label] == 'unlabeled':
            mapping[label] = 0
            continue
        cls_names.append(bdd100k_label_dict[label])
        mapping[label] = cls_index
        cls_index += 1
    
    with open("seg_dataset_meta/bdd100k_class_list.txt", 'w') as f:
        for key in cls_names:
            f.write(key + '\n')
    print(mapping)
    numpy_mapping = np.zeros((256), dtype=np.uint8)
    for k, v in mapping.items():
        numpy_mapping[k] = v

    image_dir = os.path.join(base_path, "images", '10k')
    label_dir = os.path.join(base_path, "labels", 'sem_seg', 'masks')
    output_dir = os.path.join(output_path, "labels", 'sem_seg', 'remapped_masks')
    splits = os.listdir(label_dir)
    for split in splits:
        json_path = f"seg_dataset_meta/jsonified_bdd100k_seg_{split}.json"
        json_obj = dict()
        json_obj['label_sets'] = [key for key in cls_names]
        images = []
        labels = []
        remapped_labels = []
        split_path = os.path.join(label_dir, split)
        raw_image_split_path = os.path.join(image_dir, split)
        if not os.path.isdir(split_path):
            continue
        original_labels = os.listdir(split_path)
        original_labels.sort()
        original_images = os.listdir(raw_image_split_path)
        original_images.sort()
        print("Remapping {} images in {}".format(len(original_labels), split_path))
        for label, image in tqdm.tqdm(zip(original_labels, original_images), dynamic_ncols=True):
            label_path = os.path.join(split_path, label)
            image_path = os.path.join(raw_image_split_path, image)
            if not os.path.isfile(label_path):
                continue
            output_label_path = os.path.join(output_dir, split, label)
            original_seg = cv2.imread(label_path, -1)
            remapped_seg = remapping_label(original_seg, numpy_mapping)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            cv2.imwrite(output_label_path, remapped_seg)

            images.append(image_path)
            labels.append(label_path)
            remapped_labels.append(output_label_path)
        json_obj['images'] = images
        json_obj['original_labels'] = labels
        json_obj['labels'] = remapped_labels
        with open(json_path, 'w') as f:
            json.dump(json_obj, f, indent=2)

if __name__ == "__main__":
    Fire(remap_bdd100k_to_kitti360)