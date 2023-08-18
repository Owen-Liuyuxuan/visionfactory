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
import tqdm
from typing import List
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

import sys
package_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, package_path)
from vision_base.data.datasets.nuscenes_utils import NuScenes



LABELED_OBJECTS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def _split_to_samples(nusc, split_logs: List[str]) -> List[str]:
    """
    Convenience function to get the samples in a particular split.
    :param split_logs: A list of the log names in this split.
    :return: The list of samples.
    """
    samples = []
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            samples.append(sample['token'])
    return samples

def main(dataroot, json_path, version='v1.0-trainval'):
    nusc = NuScenes(dataroot, version)

    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse # noqa: F841

    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = True
    main_object['total_frames'] = 0

    imsize = (1600, 900)

    split_logs = create_splits_logs('train', nusc)
    # Use only the samples from the current split.
    sample_tokens = _split_to_samples(nusc, split_logs)

    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        lidar_token = sample['data']['LIDAR_TOP']
        cam_tokens = [sample['data'][cam] for cam in CAMERA_NAMES]
        
        # sensor records
        sd_record_cams = [nusc.get('sample_data', cam_token) for cam_token in cam_tokens]
        sd_record_lid = nusc.get('sample_data', lidar_token)
        cs_record_cams = [nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token']) for sd_record in sd_record_cams]
        cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        # transforms
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
        ego_to_cams = [transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                        inverse=True) for cs_record_cam in cs_record_cams]
        velo_to_cams = [np.dot(ego_to_cam, lid_to_ego) for ego_to_cam in ego_to_cams]

        velo_to_cams_kitti = [np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix) for velo_to_cam in velo_to_cams]
        
        ## Intrinsics
        p_left_kittis = [np.zeros((3, 4)) for _ in range(len(cs_record_cams))]
        for i, p in enumerate(p_left_kittis):
            p[:3, :3] = cs_record_cams[i]['camera_intrinsic']
        
        filename_cam_fulls = [sd_record['filename'] for sd_record in sd_record_cams]

        lidar_annotations = []
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            # get box in LIDAR frame
            _, box_lidar_nusc, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.ANY, selected_anntokens=[sample_annotation_token])
            box_lidar_nusc = box_lidar_nusc[0]
            visibility_level = 4 - int(sample_annotation['visibility_token'])
            lidar_annotations.append(dict(box = box_lidar_nusc, detection_name = detection_name, visibility_level = visibility_level))

        for i, cam in enumerate(CAMERA_NAMES):
            main_object['images'].append(
                os.path.join(nusc.dataroot, filename_cam_fulls[i])
            )
            main_object['calibrations'].append(
                dict(
                    P2=p_left_kittis[i].tolist(),
                    R0_rect=np.eye(3).tolist(),
                    Tr_velo_to_cam=velo_to_cams_kitti[i][0:3].tolist(),
                )
            )
            main_object['total_frames'] += 1
            cam_annotations = []
            for lidar_anno in lidar_annotations:
                box_cam_kitti = KittiDB.box_nuscenes_to_kitti(lidar_anno['box'],
                    Quaternion(matrix=velo_to_cams_kitti[i][:3, :3]),
                    velo_to_cams_kitti[i][:3, 3],
                    Quaternion(axis=[1, 0, 0], angle=0))

                bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kittis[i], imsize=imsize)

                if bbox_2d is None:
                    continue

                # Compute alpha for kitti camera based model
                v = np.dot(box_cam_kitti.rotation_matrix, np.array([1, 0, 0]))
                yaw = -np.arctan2(v[2], v[0])

                alpha = yaw - np.arctan2(-box_cam_kitti.center[2], box_cam_kitti.center[0]) - 0.5 * np.pi
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                if alpha <= -np.pi:
                    alpha += 2 * np.pi

                xyz = box_cam_kitti.center.tolist()
                xyz[1] = xyz[1] - 0.5 * box_cam_kitti.wlh[2]
                cam_annotations.append(
                    dict(
                        category_name = lidar_anno['detection_name'],
                        bbox2d = bbox_2d,
                        visibility_level = lidar_anno['visibility_level'],
                        theta = yaw,
                        alpha = alpha,
                        xyz = xyz,
                        whl = [box_cam_kitti.wlh[0], box_cam_kitti.wlh[2], box_cam_kitti.wlh[1]],
                    )
                )
            main_object['annotations'].append(cam_annotations)
    
    
    json.dump(main_object, open(json_path, 'w'))


if __name__ == '__main__':
    kitti_obj_dir = '/data/nuscene'
    json_path = 'nusc_object.json'
    main(kitti_obj_dir, json_path)
