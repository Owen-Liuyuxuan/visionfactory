# Dataset Explained

## Unified Json Dataset:

```python
"""
    Unified Json Dataset:
    Json format:
    {
        "labeled_objects": List[str]; list of annotated object names; should be a subset of the unified Types:
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
"""
```

## KITTI Dataset:
KITTI dataset only annotated a subset of nuScenes annotations and the labeling style is significantly different (for example KITTI Cyclist includes both the bike and the people; while nuScenes separate them as two objects). 

We simply [map the classes](../jsonify_kitti.py) together and ignore these in-evitable labeling noise.

```python
"""
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
"""
```

## COCO datasaet for detection

We use COCO dataset format mainly for evaluating 2D detection results. It is also simple guides for writing codes to transform COCO dataset format to the unified format.

```python
"""
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
```


## ONCE dataset

[ONCE Dataset](https://once-for-auto-driving.github.io/) is a large-scale datasets in Chinese citys. Training with this dataset can significantly improve detection results deployed in China.

```python
"""
    each scene is contained in a JSON file:
    Json Format:
    {
        'calib':Dict[str, Dict[str]]
            calib['cam0[1|3|5|6|7|8]'] : {
                "cam_to_velo": List of List -> 4x4 transformation
                "cam_intrinsic": List of List -> 3x3 Projection matrix
                "distortion": List -> 1x7 distortion in openCV
            }
        'frames': List[Dict]
            [{
                "sequence_id": str,
                "frame_id": str,
                "pose": List[7] -> qx,qy,qz,qw,x,y,z
                "annos": List[Dict]
            }]
            distorted_raw_image_path = base_path + f['sequence_id'] + cam + f['frame_id'].jpg 
            anno = frames[i]['annos']
            anno = {
                "name": List[str], list of N object types ['Car'|'Truck'|'Bus'|'Pedestrian'|'Cyclist'].
                "boxes_3d": List[List[float]] -> Nx7 (x,y,z,l,w,h,theta) -> all in LiDAR frame
                "boxes_2d": Dict[str, List[List[float]]]
                    bbox2d = anno['boxes_2d']['cam01'][i] -> [xyxy] format in the "rectified" image
            }
    }
"""
```
In order to transform ONCE dataset to the unified Json dataset. Here are the additional works to notice:

- Map the categories to nuScenes similar to KITTI.
- **Undistort the image and save the rectified image in a different directory**
- Convert objects' rotation and position to each camera's frame. Filter out out-of-scope objects.
