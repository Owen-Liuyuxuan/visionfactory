import sys
import os
import numpy as np
import numba
from fire import Fire
import tqdm
import cv2
import json

sys.path.append(os.getcwd())
from segmentation.data_utils.labels_apollo import labels as apollo_labels

def label_preprocessing(name):
    name =name.replace("vegatation", "vegetation")
    name = name.replace('_groups', "")
    name = name.replace('_group', "")
    # name =name.replace("others", "unlabeled")
    return name

@numba.jit(nopython=True)
def remapping_label(original_seg, mapping):
    remapped_seg = np.zeros((original_seg.shape[0], original_seg.shape[1]), dtype=np.uint8)
    h, w = original_seg.shape
    for i in range(h):
        for j in range(w):
            remapped_seg[i, j] = mapping[original_seg[i, j]]

    return remapped_seg

broken_images = ["171206_031956529_Camera_5.jpg",
                      "171206_032007529_Camera_5.jpg",
                      "171206_032028255_Camera_5.jpg",
                      "171206_032031281_Camera_5.jpg",
                      "171206_032038560_Camera_5.jpg"]

def remapping_apollo_to_kitti360(base_path="/data/ApolloScene", output_path="/home/remapped_apollo"):
    os.makedirs(output_path, exist_ok=True)
    mapping = {}
    cls_names = []
    cls_index = 0
    for label in apollo_labels:
        process_name = label_preprocessing(label.name)
        if process_name == 'unlabeled':
            mapping[label.id] = 0
            continue
        if process_name == 'sky':
            mapping[label.id] = 0 # apollo indeed does not label sky
            continue
        if process_name in cls_names:
            mapping[label.id] = cls_names.index(process_name)
            continue
        cls_names.append(process_name)
        mapping[label.id] = cls_index
        cls_index += 1

    with open("seg_dataset_meta/apollo_class_list.txt", 'w') as f:
        for key in cls_names:
            f.write(key + '\n')

    numpy_mapping = np.zeros((256), dtype=np.uint8)
    for k, v in mapping.items():
        numpy_mapping[k] = v
    print(mapping)

    json_path = "seg_dataset_meta/jsonified_apollo_seg.json"
    json_obj = dict()
    json_obj['label_sets'] = [key for key in cls_names]
    images = []
    labels = []
    remapped_labels = []

    label_dir = os.path.join(base_path, "Label")
    image_dir = os.path.join(base_path, 'ColorImage')
    records = os.listdir(label_dir)
    for record in records:
        record_path = os.path.join(label_dir, record)
        record_image_path = os.path.join(image_dir, record)
        if not os.path.isdir(record_path):
            continue
        output_record_dir = os.path.join(output_path, record)
        os.makedirs(output_record_dir, exist_ok=True)
        cameras = os.listdir(record_path)
        for camera in cameras:
            camera_path = os.path.join(record_path, camera)
            image_camera_path = os.path.join(record_image_path, camera)
            if not os.path.isdir(camera_path):
                continue
            output_camera_dir = os.path.join(output_record_dir, camera)
            os.makedirs(output_camera_dir, exist_ok=True)
            original_seg_paths = os.listdir(camera_path)
            original_seg_paths = [seg_path for seg_path in original_seg_paths if seg_path.endswith('.png')]
            original_seg_paths.sort()
            image_paths = os.listdir(image_camera_path)
            image_paths = [image_path for image_path in image_paths if image_path.endswith('.jpg')]
            image_paths.sort()
            print("Processing {}/{}".format(record, camera))
            for seg_path, image_path in tqdm.tqdm(zip(original_seg_paths, image_paths), dynamic_ncols=True):
                if image_path in broken_images:
                    print(f"{image_path} found and skip")
                    continue
                image_path = os.path.join(image_camera_path, image_path)
                full_seg_path = os.path.join(camera_path, seg_path)
                output_seg_path = os.path.join(output_camera_dir, seg_path)
                original_seg = cv2.imread(full_seg_path, -1)
                remapped_seg = remapping_label(original_seg, numpy_mapping)
                cv2.imwrite(output_seg_path, remapped_seg)

                images.append(image_path)
                labels.append(full_seg_path)
                remapped_labels.append(output_seg_path)
    json_obj['images'] = images
    json_obj['original_labels'] = labels
    json_obj['labels'] = remapped_labels
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=2)

if __name__ == "__main__":
    Fire(remapping_apollo_to_kitti360)