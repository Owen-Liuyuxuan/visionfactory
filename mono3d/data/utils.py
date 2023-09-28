"""
    Unified Json Dataset:
    Json format:
    {
        "labeled_objects": List[str]; list of annotated object names; should be a subset of  the unified Types:
                            ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        "images": List[str]; list of image paths(full path)
        "is_labeled_3d": bool; whether the dataset is 3D labeled
        "total_frames": int; total number of frames
        "calibrations": Dict[str, List[float]]: camera calibration matrixes, each matrix is a list of 12 floats
        "annotations": List[List[Dict]]:
            annotation=annotations[i][j]; the j-th object in the i-th image
            for 3D datasets (Using the JsonMonoDataset / Object3DKittiMetricEvaluateHook APIs):
                annotation['xyz' | 'whl' | 'alpha' | 'theta' | 'bbox2d' | 'category_name' | 'visibility_level' | 'image_id']
            for 2D labeled dataset (Using the Json2DDataset / Object2DEvaluateHook APIs):
                annotation['bbox2d' | 'visibility_level' | 'category_name' | 'image_id']
            key description:
                "bbox2d": List[float]; [x1, y1, x2, y2]
                "xyz": List[float]; [x, y, z] in the camera coordinate
                "whl": List[float]; [w, h, l]
                "alpha": float; observation angle of object, ranging [-pi..pi]
                "theta": float; rotation ry around Y-axis in camera coordinates [-pi..pi]
                "visibility_level": int; 0: fully visible, 1: partly occluded, 2: largely occluded, 3: un-usable following kitti
                "category_name": str; one of the unified Types
                "image_id": int; the index of the image in "images"
    }
    for unlabeled dataset just for demonstration (Using the JsonTestDataset API):
        only "images", "calibrations" are needed.
    Jsonfile that could run with 3D data APIs also work with 2D data APIs/test APIs.
    Jsonfile that could run with 2D data APIs also work with test APIs.
    

    KITTI dataset:

    -training
        --image_2
            000000.png
            000001.png
            ...

        --label_2
                #Values    Name      Description
            ----------------------------------------------------------------------------
            1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                                'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                'Misc' or 'DontCare'
            1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                                truncated refers to the object leaving image boundaries
            1    occluded     Integer (0,1,2,3) indicating occlusion state:
                                0 = fully visible, 1 = partly occluded
                                2 = largely occluded, 3 = unknown
            1    alpha        Observation angle of object, ranging [-pi..pi]
            4    bbox         2D bounding box of object in the image (0-based index):
                                contains left, top, right, bottom pixel coordinates
            3    dimensions   3D object dimensions: height, width, length (in meters)
            3    location     3D object location x,y,z in camera coordinates (in meters)
            1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
            1    score        Only for results: Float, indicating confidence in
                                detection, needed for p/r curves, higher is better.

    COCO dataset for detection.
    Json format:
    {
        "info" : {
            # can be an empty dictionary. COCO.info in pycocoAPI will print out all the key-value pair of this dictionary without checking if "info" is in dataset.
        },
        "images" :[image],
        "annotations: [annotation],
        "licenses": [licence], # not nesessary, not directly used in COCO API. 
        "categories": [category]
    }
    image{
        "id": int,
        "width": int, "height": int,
        "file_name": str,
        "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime,
    }
    annotation{
        "id": int, 
        "image_id": int, 
        "category_id": int, 
        "segmentation": RLE or [polygon], # For bounding box detection, omit such field
        "area": float, 
        "bbox": [x,y,width,height], #[top_left_x, top_left_y, w, h]
        "iscrowd": 0 or 1, #ignore or not
    }
"""


import json
from typing import List
from PIL import Image
import numpy as np


def bbox_xyxy_to_xywh(bbox_xyxy:List[float])->List[float]:
    """
    Args:
        bbox_xyxy: [x1, y1, x2, y2]
    Return:
        bbox_xywh: [x, y, w, h]
    """
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def Json2annotation(gt_json_file):
    """
    gt_json: Dict
        gt_json['images'] = List[str]
        gt_json['annotations'] = List[List[Dict]]
            obj = gt_json['annotations'][0]
            obj['xyz' | 'whl' | 'alpha' | 'theta' | 'box2d' | 'category_name' | 'visibility_level']
    kitti_eval_annotation: List[Dict]
        frames = List[Dict[str, List|np.ndarray]]
            frame = Dict[str, List|np.ndarray]
            frame['name' | 'truncated' | 'occluded' | 'alpha' | bbox' | 'dimensions' | 'location' | 'rotation_y' | 'score']
            bbox: xyxy; dimension: lhw; location: xyz
    """
    if isinstance(gt_json_file, str):
        with open(gt_json_file, 'r') as f:
            gt_json = json.load(f)
    elif isinstance(gt_json_file, dict):
        gt_json = gt_json_file

    kitti_eval_annotation = []
    for json_anno in gt_json['annotations']:
        frame = dict()
        frame['name'] = np.array([obj['category_name'] for obj in json_anno])
        frame['truncated'] = np.array([0 for _ in json_anno])
        frame['occluded'] = np.array([obj['visibility_level'] for obj in json_anno])
        frame['alpha'] = np.array([obj['alpha'] for obj in json_anno])
        frame['bbox'] = np.array([obj['bbox2d'] for obj in json_anno]).reshape(-1, 4)
        frame['dimensions'] = np.array([obj['whl'] for obj in json_anno]).reshape(-1, 3)[:, [2, 1, 0]] #[N, 3]
        frame['location'] = np.array([obj['xyz'] for obj in json_anno]).reshape(-1, 3) #[N, 3]
        frame['location'][:, 1] = frame['location'][:, 1] + 0.5*frame['dimensions'][:, 1] # kitti receive bottom center
        frame['rotation_y'] = np.array([obj['theta'] for obj in json_anno]).reshape(-1) #[N, 1]
        if len(json_anno) > 0 and 'score' in json_anno[0]:
            frame['score'] = np.array([obj['score'] for obj in json_anno]).reshape(-1)
        else:
            frame['score'] = np.array([1 for obj in json_anno]).reshape(-1)
        kitti_eval_annotation.append(frame)
    return kitti_eval_annotation


def JsonToCoCo(gt_json_file, read_image=False, image_w=None, image_h=None):
    if isinstance(gt_json_file, str):
        with open(gt_json_file, 'r') as f:
            gt_data = json.load(f)
    else:
        assert isinstance(gt_json_file, dict)
        gt_data = gt_json_file

    coco_data = {}
    coco_data['info'] = {}
    coco_data['images'] = []
    coco_data['annotations'] = []
    coco_data['licenses'] = []
    coco_data['categories'] = [
        dict(id=i, name=category_name, supercategory=category_name)
        for i, category_name in enumerate(gt_data['labeled_objects'])
    ]

    for i in range(len(gt_data['images'])):
        ## Image Object
        image = dict()
        image['id'] = i
        if read_image:
            image['file_name'] = gt_data['images'][i]
            image['height'], image['width'] = Image.open(image['file_name']).size
        else:
            image['height'], image['width'] = image_h, image_w
        image['file_name'] = gt_data['images'][i]
        image['license'] = 0

        coco_data['images'].append(image)
        ## Annotation Object
        for j in range(len(gt_data['annotations'][i])):
            annotation = dict()
            annotation['id'] = len(coco_data['annotations'])
            annotation['image_id'] = i
            category_name = gt_data['annotations'][i][j]['category_name']
            annotation['category_id'] = gt_data['labeled_objects'].index(category_name)
            annotation['segmentation'] = []
            annotation['area'] = 0
            annotation['bbox'] = bbox_xyxy_to_xywh(gt_data['annotations'][i][j]['bbox2d'])
            annotation['iscrowd'] = 0
            coco_data['annotations'].append(annotation)

    return coco_data
