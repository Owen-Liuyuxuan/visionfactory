import sys
import os
import numpy as np
import numba 
from fire import Fire
import tqdm
import json
import cv2
sys.path.append(os.getcwd())

def convert_a2d2_color_string_to_rgb(color_string):
    color_string = color_string.replace("#", "")
    return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4))

def a2d2_label_preprocessing(name):
    name = name.strip('1234').strip().lower()
    return name

@numba.jit(nopython=True)
def remapping_label(original_seg, mapping):
    output = np.zeros((original_seg.shape[0], original_seg.shape[1]), dtype=np.uint8)
    for i in range(original_seg.shape[0]):
        for j in range(original_seg.shape[1]):
            r, g, b = original_seg[i, j, 0], original_seg[i, j, 1], original_seg[i, j, 2]
            output[i, j] = mapping[r, g, b]
    return output

def remapping_a2d2_to_kitti360(base_path="/data/a2d2/camera_lidar_semantic", output_path="/home/remapped_a2d2"):
    os.makedirs(output_path, exist_ok=True)
    a2d2_class_color = json.load(open("./segmentation/data_utils/class_list.json"))
    class_to_color = {}
    for key in a2d2_class_color:
        class_to_color[a2d2_class_color[key]] = convert_a2d2_color_string_to_rgb(key)

    mapping = {}
    mapping['unlabeled'] = 0
    cls_index = 0
    for label in class_to_color:
        processed_name = a2d2_label_preprocessing(label)
        if processed_name not in mapping:
            cls_index += 1
            mapping[processed_name] = cls_index

    color_to_class = {}
    for i, key in enumerate(a2d2_class_color):
        color_to_class[convert_a2d2_color_string_to_rgb(key)] = a2d2_class_color[key]

    color_to_kitti_id = {}
    for color in color_to_class:
        a2d2_class_name = color_to_class[color]
        kitti_id = mapping[a2d2_label_preprocessing(a2d2_class_name)]
        color_to_kitti_id[color] = kitti_id
    
    with open("seg_dataset_meta/a2d2_class_list.txt", 'w') as f:
        for key in mapping:
            f.write(key + '\n')
    print(color_to_kitti_id)
    # numpy version
    numpy_mapping = np.zeros((256, 256, 256), dtype=np.uint8)
    for k, v in color_to_kitti_id.items():
        numpy_mapping[k] = v

    json_path = "seg_dataset_meta/jsonified_a2d2_seg.json"
    json_obj = dict()
    json_obj['label_sets'] = [key for key in mapping]
    images = []
    labels = []
    remapped_labels = []
    sequence_times = os.listdir(base_path)
    for sequence_time in sequence_times:
        sequence_dir = os.path.join(base_path, sequence_time)
        if not os.path.isdir(sequence_dir):
            continue
        label_dir = os.path.join(sequence_dir, "label")
        image_dir = os.path.join(sequence_dir, "camera")
        camera_names = os.listdir(label_dir)
        for camera in camera_names:
            label_cam_dir = os.path.join(label_dir, camera)
            camera_cam_dir = os.path.join(image_dir, camera)
            output_sequence_dir = os.path.join(output_path, sequence_time)
            output_label_cam_dir  = os.path.join(output_sequence_dir, "remapped_label", camera)
            os.makedirs(output_label_cam_dir, exist_ok=True)
            label_files = os.listdir(label_cam_dir)
            label_files = [label for label in label_files if label.endswith(".png")]
            label_files.sort()
            image_files = os.listdir(camera_cam_dir)
            image_files = [image for image in image_files if image.endswith(".png")]
            image_files.sort()
            assert len(label_files) == len(image_files)
            print("Processing {}".format(sequence_time))
            for label, image in tqdm.tqdm(zip(label_files, image_files), dynamic_ncols=True):
                label_path = os.path.join(label_cam_dir, label)
                image_path = os.path.join(camera_cam_dir, image)
                label_image = cv2.imread(label_path)[..., ::-1]
                output_label = remapping_label(label_image, numpy_mapping)
                cv2.imwrite(os.path.join(output_label_cam_dir, label), output_label)
                images.append(image_path)
                labels.append(label_path)
                remapped_labels.append(os.path.join(output_label_cam_dir, label))
    json_obj['images'] = images
    json_obj['original_labels'] = labels
    json_obj['labels'] = remapped_labels
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=2)

if __name__ == "__main__":
    Fire(remapping_a2d2_to_kitti360)
