"""
The core idea is to produce a unified json data description file for bdd100k dataset. 

1. Unify classes annotations. We know there are categories in nuScenes not labeled in KITTI/cityscape. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.
6. In BDD100k, there are only 2D labels and .

Suggested unified Types: 

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

in KITTI, we mainly have this mapping dictionary {'Car': 'car', 'Pedestrian': 'pedestrian', 'Van': 'truck', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Tram': 'bus'}. We preserve all other informations, visibility we will preserve occluded
"""
import numpy as np
import os
import json
from PIL import Image
import tqdm

LABELED_OBJECTS = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'motorcycle', 'trailer']


def main(bdd100k_base_path, json_path):
    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = False
    main_object['total_frames'] = 0

    print("Start Loading BDD100k Label file, may take some time.")
    train_json_path = os.path.join(bdd100k_base_path, 'labels', 'det_20', 'det_train.json')
    image_dir       = os.path.join(bdd100k_base_path, 'images', '100k', 'train')
    
    bdd_annotation_data = json.load(
        open(train_json_path, 'r')
    )
    print("BDD100k Label file loaded.")

    for index in tqdm.tqdm(range(len(bdd_annotation_data))):
        main_object['images'].append(
            os.path.join(
                image_dir, bdd_annotation_data[index]['name']
            )
        )

        main_object['calibrations'].append(
            dict(
                P = [640, 0, 640, 0,
                    0,  640, 340, 0,
                    0, 0, 1, 0]
            ) # we have to use a fake calibration file here because BDD100k have no calibration data 
        )
        annotations = []
        frame = bdd_annotation_data[index]
        if 'labels' not in frame:
            main_object['annotations'].append(annotations)
            continue
        
        for obj in frame['labels']:
            category = obj['category']
            if category in LABELED_OBJECTS:
                box_dic = obj['box2d']
                obj_dict=dict(
                    image_id=index,
                    bbox2d = [box_dic[key] for key in ['x1', 'y1', 'x2', 'y2']],
                    visibility_level = int(obj['attributes']['occluded']) * 3,
                    category_name=category
                )
                annotations.append(obj_dict)
        main_object['annotations'].append(annotations)

    json.dump(main_object, open(json_path, 'w'))

if __name__ == '__main__':
    bdd100k_base_path = '/data/bdd100k/'
    json_path = 'bdd100k_object.json'
    main(bdd100k_base_path, json_path)