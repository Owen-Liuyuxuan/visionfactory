"""
The core idea is to produce a unified json data description file for KITTI and nuScenes dataset. 

1. Unify classes annotations. We know there are categories in nuScenes not labeled in KITTI. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.

Suggested unified Types: 

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

in KITTI, we mainly have this mapping dictionary {'Car': 'car', 'Pedestrian': 'pedestrian', 'Van': 'truck', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Tram': 'bus'}. We preserve all other informations, visibility we will preserve occluded
"""
import numpy as np
import os
import json
from PIL import Image
import tqdm

CATE_MAPPING = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Van': 'truck', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Tram': 'bus'}

LABELED_OBJECTS = ['car', 'pedestrian', 'truck', 'bicycle', 'bus']

def parse_label(path, img_id, width, height, task):
    """
    convert KITTI label to json format
    :param path:
    :param img_id:
    :param width:
    :param height:
    :return: annotations
    """
    with open(path, 'r') as f:
        str_list = f.readlines()
    str_list = [itm.rstrip() for itm in str_list if itm != '\n']
    annotations = []
    # 找到此张图像上对应的bb
    for idx, label in enumerate(str_list):

        (truncated, occluded, alpha, bbox_l, bbox_t, bbox_r, bbox_b,
            h, w, l, x, y, z, ry) = [
                float(itm) for itm in label.split()[1:15]
            ]
        type_name = label.split()[0]
        if type_name not in CATE_MAPPING:
            continue
        unified_type_name = CATE_MAPPING[type_name]
        annotations.append(dict(
            whl = [w,h,l],
            xyz = [x,y - 0.5 * h,z],
            alpha = alpha,
            theta = ry,
            image_id = img_id,
            bbox2d = [bbox_l, bbox_t, bbox_r, bbox_b],
            visibility_level = occluded,
            category_name = unified_type_name,
        ))
    return annotations


def read_calib_file(path):
    '''
    read KITTI calib file
    '''
    calib = dict()
    with open(path, 'r') as f:
        str_list = f.readlines()
    str_list = [itm.rstrip() for itm in str_list if itm != '\n']
    for itm in str_list:
        calib[itm.split(':')[0]] = itm.split(':')[1]
    for k, v in calib.items():
        calib[k] = [float(itm) for itm in v.split()]
    return calib
    

def main(kitti_obj_dir, json_path):
    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = True
    main_object['total_frames'] = 0
    image_2_dir = os.path.join(kitti_obj_dir, 'training', 'image_2')
    #image_3_dir = os.path.join(kitti_obj_dir, 'training', 'image_3')
    calib_dir   = os.path.join(kitti_obj_dir, 'training', 'calib')
    label_2_dir = os.path.join(kitti_obj_dir, 'training', 'label_2')
    num_frames = len(os.listdir(image_2_dir))
    for idx in tqdm.tqdm(range(num_frames)):
        image_2_path = os.path.join(image_2_dir, '{:06d}.png'.format(idx))
        #image_3_path = os.path.join(image_3_dir, '{:06d}.png'.format(idx))
        calib_path   = os.path.join(calib_dir, '{:06d}.txt'.format(idx))
        label_2_path = os.path.join(label_2_dir, '{:06d}.txt'.format(idx))
        calib = read_calib_file(calib_path)
        main_object['calibrations'].append(calib)
        main_object['images'].append(image_2_path)
        pil_image = Image.open(image_2_path)
        width, height = pil_image.size
        annotations = parse_label(label_2_path, idx, width, height, 'instance')
        main_object['annotations'].append(annotations)
        main_object['total_frames'] += 1
    
    json.dump(main_object, open(json_path, 'w'))

if __name__ == '__main__':
    #kitti_obj_dir = '/data/kitti_obj'
    kitti_obj_dir = '/data/kitti_obj'
    json_path = 'kitti_object.json'
    main(kitti_obj_dir, json_path)