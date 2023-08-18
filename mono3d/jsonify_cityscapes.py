"""
The core idea is to produce a unified json data description file for Cityscape dataset.

1. Unify classes annotations. We know there are categories in nuScenes not labeled in KITTI/cityscape. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.
6. In cityscape, we suggest only use the 2D label.

Suggested unified Types:

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

in KITTI, we mainly have this mapping dictionary {'Car': 'car', 'Pedestrian': 'pedestrian', 'Van': 'truck', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Tram': 'bus'}. We preserve all other informations, visibility we will preserve occluded
"""
import os
import json

LABELED_OBJECTS = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'motorcycle', 'trailer', 'pedestrian']


def main(cityscape_base_path, json_path):
    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = False
    main_object['total_frames'] = 0
    
    image_base_path = os.path.join(cityscape_base_path, 'leftImg8bit', 'train')
    gtbox3d_base_path = os.path.join(cityscape_base_path, 'gtBbox3d', 'train')
    peroson_base_path = os.path.join(cityscape_base_path, 'gtBboxCityPersons', 'train')
    mode_cities = sorted(os.listdir(image_base_path))
    image_paths = []
    gtbox3d_paths = []
    person_paths = []
    for city_name in mode_cities:
        image_city_path = os.path.join(image_base_path, city_name)
        image_list = sorted(os.listdir(image_city_path))
        image_paths += [os.path.join(image_city_path, image_path) for image_path in image_list]
        
        gtbox3d_city_path = os.path.join(gtbox3d_base_path, city_name)
        gtbox3d_list = sorted(os.listdir(gtbox3d_city_path))
        gtbox3d_paths += [os.path.join(gtbox3d_city_path, gtbox3d_path) for gtbox3d_path in gtbox3d_list]

        person_city_path = os.path.join(peroson_base_path, city_name)
        person_list = sorted(os.listdir(person_city_path))
        person_paths += [os.path.join(person_city_path, person_path) for person_path in person_list]

    for index in range(len(image_paths)):
        main_object['images'].append(image_paths[index])
        bbox_data = json.load(open(gtbox3d_paths[index], 'r'))
        person_data = json.load(open(person_paths[index], 'r'))

        fx = bbox_data['sensor']['fx']
        fy = bbox_data['sensor']['fy']
        u0 = bbox_data['sensor']['u0']
        v0 = bbox_data['sensor']['v0']
        main_object['calibrations'].append(
            dict(
                P = [fx, 0, u0, 0,
                    0,  fy, v0, 0,
                    0, 0, 1, 0]
            )
        )
        annotations = []
        for obj in bbox_data['objects']:
            label = obj['label']
            if label in LABELED_OBJECTS:
                obj_dict=dict(
                    image_id = index,
                    bbox2d = obj['2d']['amodal'],
                    visibility_level=obj['occlusion'],
                    category_name=label
                )
                annotations.append(obj_dict)
        for obj in person_data['objects']:
            obj_dict=dict(
                image_id=index,
                bbox2d = obj['bbox'],
                visibility_level=0.0,
                category_name='pedestrian'
            )
            annotations.append(obj_dict)
        main_object['annotations'].append(annotations)
    
    json.dump(main_object, open(json_path, 'w'))


if __name__ == '__main__':
    kitti_obj_dir = '/data/cityscapes'
    json_path = 'cityscape_object.json'
    main(kitti_obj_dir, json_path)
