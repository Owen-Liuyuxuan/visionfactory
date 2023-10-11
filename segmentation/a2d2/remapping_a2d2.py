import sys
import os
import numpy as np
import numba 
from fire import Fire
import tqdm
import json
import cv2
sys.path.append(os.getcwd())
from segmentation.evaluation.labels import labels as kitti360_labels


def convert_a2d2_color_string_to_rgb(color_string):
    color_string = color_string.replace("#", "")
    return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4))

def a2d2_find_similar_label(name, kitti360_labels):
    name = name.strip('1234').strip().lower()
    name = name.replace('pedestrian', 'person')
    name = name.replace('small vehicles', 'car')
    name = name.replace('sidebars', 'guard rail')
    name = name.replace('speed bumper', 'unlabeled')
    name = name.replace('solid line', 'road')

    ## Completely same
    for kitti360_label in kitti360_labels:
        if kitti360_label.name == name:
            return name, kitti360_label
    ## Similar
    for kitti360_label in kitti360_labels:
        if kitti360_label.name in name:
            return name, kitti360_label
    return name, None

def remapping_label(original_seg, mapping):
    output = np.zeros((original_seg.shape[0], original_seg.shape[1]), dtype=np.uint8)
    for key, value in mapping.items():
        mask = (original_seg == np.array(key)).all(axis=-1)
        output[mask] = value
    return output

def remapping_a2d2_to_kitti360(base_path="/data/a2d2/camera_lidar_semantic_bboxes", output_path="/home/remapped_a2d2"):
    os.makedirs(output_path, exist_ok=True)
    a2d2_class_color = json.load(open("./segmentation/a2d2/class_list.json"))
    class_to_color = {}
    for key in a2d2_class_color:
        class_to_color[a2d2_class_color[key]] = convert_a2d2_color_string_to_rgb(key)

    mapping = {}
    for label in class_to_color:
        extracted_name, similar_label = a2d2_find_similar_label(label, kitti360_labels)
        mapping[label] = similar_label.id if similar_label is not None else 0

    color_to_class = {}
    for i, key in enumerate(a2d2_class_color):
        color_to_class[convert_a2d2_color_string_to_rgb(key)] = a2d2_class_color[key]

    color_to_kitti_id = {}
    for color in color_to_class:
        a2d2_class_name = color_to_class[color]
        kitti_id = mapping[a2d2_class_name]
        color_to_kitti_id[color] = kitti_id
    
    sequence_times = os.listdir(base_path)
    for sequence_time in sequence_times:
        sequence_dir = os.path.join(base_path, sequence_time)
        if not os.path.isdir(sequence_dir):
            continue
        label_dir = os.path.join(sequence_dir, "label", "cam_front_center")
        output_sequence_dir = os.path.join(output_path, sequence_time)
        output_label_dir  = os.path.join(output_sequence_dir, "remapped_label", "cam_front_center")
        os.makedirs(output_label_dir, exist_ok=True)
        labels = os.listdir(label_dir)
        labels = [label for label in labels if label.endswith(".png")]
        labels.sort()
        print("Processing {}".format(sequence_time))
        for label in tqdm.tqdm(labels):
            label_path = os.path.join(label_dir, label)
            label_image = cv2.imread(label_path)[..., ::-1]
            output_label = remapping_label(label_image, color_to_kitti_id)
            cv2.imwrite(os.path.join(output_label_dir, label), output_label)

if __name__ == "__main__":
    Fire(remapping_a2d2_to_kitti360)
