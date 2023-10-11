import sys
import os
import numpy as np
import numba
from fire import Fire
import tqdm
import cv2

sys.path.append(os.getcwd())
from segmentation.data_utils.labels_apollo import labels as apollo_labels
from segmentation.evaluation.labels import labels as kitti360_labels

def find_similar_label(apollo_label, kitti360_labels):
    name:str = apollo_label.name
    name = name.replace("_", " ")
    name =name.replace("rover", "ego vehicle")
    name =name.replace("others", "unlabeled")
    name =name.replace("siderwalk", "sidewalk")
    name =name.replace("traffic cone", "pole")
    name =name.replace("overpass", "bridge")
    name =name.replace("tricycle", "motorcycle")
    name =name.replace("dustbin", "trash bin")
    name =name.replace("dustbin", "trash bin")
    name =name.replace("billboard", "unlabeled")
    name =name.replace("vegatation", "vegetation")

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

def remapping_apollo_to_kitti360(base_path="/data/ApolloScene", output_path="/home/remapped_apollo"):
    os.makedirs(output_path, exist_ok=True)
    mapping = {}
    for label in apollo_labels:
        extracted_name, similar_label = find_similar_label(label, kitti360_labels)
        mapping[label.id] = similar_label.id if similar_label is not None else -1
        if similar_label is None:
            print("Not found similar label for {}/{}".format(label.name, extracted_name))
    numpy_mapping = np.zeros((256), dtype=np.uint8)
    for k, v in mapping.items():
        numpy_mapping[k] = v
    label_dir = os.path.join(base_path, "Label")
    records = os.listdir(label_dir)
    for record in records:
        record_path = os.path.join(label_dir, record)
        if not os.path.isdir(record_path):
            continue
        output_record_dir = os.path.join(output_path, record)
        os.makedirs(output_record_dir, exist_ok=True)
        cameras = os.listdir(record_path)
        for camera in cameras:
            camera_path = os.path.join(record_path, camera)
            if not os.path.isdir(camera_path):
                continue
            output_camera_dir = os.path.join(output_record_dir, camera)
            os.makedirs(output_camera_dir, exist_ok=True)
            original_seg_paths = os.listdir(camera_path)
            print("Processing {}/{}".format(record, camera))
            for seg_path in tqdm.tqdm(original_seg_paths):
                full_seg_path = os.path.join(camera_path, seg_path)
                output_seg_path = os.path.join(output_camera_dir, seg_path)
                original_seg = cv2.imread(full_seg_path, -1)
                remapped_seg = remapping_label(original_seg, numpy_mapping)
                cv2.imwrite(output_seg_path, remapped_seg)

if __name__ == "__main__":
    Fire(remapping_apollo_to_kitti360)