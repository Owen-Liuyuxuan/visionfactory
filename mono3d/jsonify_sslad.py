"""
The core idea is to produce a unified json data description file for SSLAD dataset / similar to COCO dataset.

1. Unify classes annotations. We know there are categories in nuScenes not labeled in SSLAC. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.
6. In SSLAD, there are only 2D labels and no calibration.

Suggested unified Types:

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

"""
import os
import json
from typing import List, Dict, Any
import tqdm

LABELED_OBJECTS = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'motorcycle']
CATE_MAPPING = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Truck': 'truck',
                 'Cyclist': 'bicycle', 'Tram': 'bus', 'Tricycle': 'motorcycle'}


def get_category_id_to_name_mapping(categories:List[Dict[str, Any]], map_to_nuscenes:bool=False):
    """
    Example:
        [{'supercategory': 'Pedestrian', 'id': 1, 'name': 'Pedestrian'},
        {'supercategory': 'Cyclist', 'id': 2, 'name': 'Cyclist'},
        {'supercategory': 'Car', 'id': 3, 'name': 'Car'},
        {'supercategory': 'Truck', 'id': 4, 'name': 'Truck'},
        {'supercategory': 'Tram', 'id': 5, 'name': 'Tram'},
        {'supercategory': 'Tricycle', 'id': 6, 'name': 'Tricycle'}]
    Return:
        {1: 'Pedestrian', 2: 'Cyclist', 3: 'Car', 4: 'Truck', 5: 'Tram', 6: 'Tricycle'}
    """
    category_id_to_name_mapping = {}
    for category in categories:
        if map_to_nuscenes:
            category_id_to_name_mapping[category['id']] = CATE_MAPPING[category['name']]
        else:
            category_id_to_name_mapping[category['id']] = category['name']
    return category_id_to_name_mapping

def get_image_id_to_annotation_mapping(annotations:List[Dict[str, Any]], num_images)->List[List[int]]:
    """
        annotation example:
            annotations[0] = {'image_id': 1,
                                'category_id': 3,
                                'bbox': [65, 667, 174, 126],
                                'area': 21924,
                                'id': 1,
                                'iscrowd': 0}
        return example:
            [[0]]
    """
    image_id_to_annotation_mapping = [[] for _ in range(num_images)]
    for anno_num, annotation in enumerate(annotations):
        image_id = annotation['image_id'] - 1 # SSLAD image_id starts from 1
        image_id_to_annotation_mapping[image_id].append(anno_num)
    return image_id_to_annotation_mapping

def bbox_xywh_to_xyxy(bbox_xywh:List[float])->List[float]:
    """
    Args:
        bbox_xywh: [x, y, w, h]
    Return:
        bbox_xyxy: [x1, y1, x2, y2]
    """
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]
    

def main(SSLAD_base_path="/data/SSLAD-2D", split='train', json_path='sslad_object.json'):
    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = False
    main_object['total_frames'] = 0

    print("Start Loading SSLAD Label file, may take some time.")
    SSLAD_json_file = os.path.join(SSLAD_base_path, 'labeled', 'annotations', f'instance_{split}.json')
    image_base_path = os.path.join(SSLAD_base_path, 'labeled', split)


    annotation_data = json.load(
        open(SSLAD_json_file, 'r')
    )

    id_cate_mapping = get_category_id_to_name_mapping(
        annotation_data['categories'], map_to_nuscenes=True
    )
    image_id_to_annotation_mapping = get_image_id_to_annotation_mapping(
        annotation_data['annotations'], len(annotation_data['images'])
    )

    print("SSLAD Label file loaded.")

    for index in tqdm.tqdm(range(len(annotation_data['images']))):
        main_object['images'].append(
            os.path.join(
                image_base_path, annotation_data['images'][index]['file_name']
            )
        )

        main_object['calibrations'].append(
            dict(
                P = [640, 0, 960, 0,
                    0,  640, 720, 0,
                    0, 0, 1, 0]
            ) # we have to use a fake calibration file here because SSLAD have no calibration data
        )
        annotations = []
        annotation_indexes = image_id_to_annotation_mapping[index]
        if len(annotation_indexes) == 0:
            main_object['annotations'].append(annotations)
            continue
        
        for annotation_index in annotation_indexes:
            obj = annotation_data['annotations'][annotation_index]
            bbox = bbox_xywh_to_xyxy(obj['bbox']) # SSLAD use xywh format
            obj_dict=dict(
                image_id = index,
                bbox2d = bbox,
                visibility_level = 0,
                category_name = id_cate_mapping[obj['category_id']]
            )
            annotations.append(obj_dict)

        main_object['annotations'].append(annotations)

    json.dump(main_object, open(json_path, 'w'))


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
