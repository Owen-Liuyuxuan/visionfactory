"""
The core idea is to produce a unified json data description file for ONCE dataset.

1. Unify classes annotations. We know there are categories in nuScenes not labeled in ONCE. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.

Suggested unified Types:

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

in ONCE, we mainly have this mapping dictionary {'Car': 'car', 'Pedestrian': 'pedestrian', 'Van': 'truck', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Tram': 'bus'}. We preserve all other informations
"""
import numpy as np
import math
import os
import json
import tqdm
from pyquaternion import Quaternion
import cv2
import torch
from model.utils import compute_occlusion

CATE_MAPPING = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Truck': 'truck', 'Cyclist': 'bicycle', 'Bus': 'bus'}

LABELED_OBJECTS = ['car', 'pedestrian', 'truck', 'bicycle', 'bus']

CAMS = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']

SCENES = [f"{index:06d}" for index in [76, 80, 92, 104, 113, 121]]


 
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def main(ONCE_base_dir, json_path):
    main_object = {}
    main_object['labeled_objects'] = LABELED_OBJECTS
    main_object['images'] = []
    main_object['annotations'] = []
    main_object['calibrations'] = []
    main_object['is_labeled_3d'] = True
    main_object['total_frames'] = 0
    
    train_data_base_dir = os.path.join(ONCE_base_dir, 'train', 'data')

    for scene in SCENES:
        dataset_annotations = json.load(
            open(
                os.path.join(train_data_base_dir, scene, f"{scene}.json"), 'r'
            )
        )
        calib_dict = {}
        for cam in CAMS:
            calib_dict[cam] = dict()
            calib_dict[cam]['P'] = np.array(dataset_annotations['calib'][cam]['cam_intrinsic'])
            calib_dict[cam]['T_cam2velo'] = np.array(dataset_annotations['calib'][cam]['cam_to_velo'])
            T_velo2cam = np.linalg.inv(calib_dict[cam]['T_cam2velo'])
            calib_dict[cam]['T_velo2cam'] = T_velo2cam
            calib_dict[cam]['T_velo2cam_tran'] = T_velo2cam[0:3, 3]
            calib_dict[cam]['T_velo2cam_rot'] = Quaternion(matrix=T_velo2cam)
            calib_dict[cam]['distortion'] = np.array(dataset_annotations['calib'][cam]['distortion'])
        for frame in tqdm.tqdm(dataset_annotations['frames']):
            if 'annos' not in frame:
                continue
            for cam in CAMS:
                # image = cv2.imread(os.path.join(
                #         train_data_base_dir, frame['sequence_id'], cam, f"{frame['frame_id']}.jpg"
                #     ))
                cam_calib = calib_dict[cam]
                # h, w = image.shape[:2]
                h, w = 1020, 1920
                P = np.array(cam_calib['P'])
                new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(P,
                                                np.array(cam_calib['distortion']),
                                                (w, h), alpha=0.0, newImgSize=(w, h))
                # image = cv2.undistort(image, P,
                #                                 np.array(cam_calib['distortion']),
                #                                 newCameraMatrix=new_cam_intrinsic)
                target_dir = os.path.join('/home/rectified_once/', frame['sequence_id'], cam)
                # os.makedirs(target_dir, exist_ok=True)
                # cv2.imwrite(
                #      os.path.join(target_dir, f"{frame['frame_id']}.jpg"), image
                # )

                main_object['images'].append(
                    os.path.join(target_dir, f"{frame['frame_id']}.jpg")
                )
                P34 = np.zeros([3, 4])
                P34[0:3, 0:3] = new_cam_intrinsic
                main_object['calibrations'].append(dict(P=P34.reshape(-1).tolist()))
                annotations = []
                for i in range(len(frame['annos']['boxes_3d'])):
                    bbox2d = np.array(frame['annos']['boxes_2d'][cam][i])
                    if np.all(bbox2d < 0):
                        continue
                    obj = frame['annos']['boxes_3d'][i]
                    theta = obj[6]
                    center = np.array(obj[0:3])
                    whl = np.array([obj[idx] for idx in [4, 5, 3]])
                    box_rot = Quaternion._from_axis_angle(np.array([0, 0, 1]), theta)
                    velo2cam_rot = calib_dict[cam]['T_velo2cam_rot']
                    velo2cam_tran = calib_dict[cam]['T_velo2cam_tran']
                    cam_center = np.dot(velo2cam_rot.rotation_matrix, center) + velo2cam_tran
                    cam_orientation = velo2cam_rot * box_rot
                    if cam_center[2] < 3:
                        continue

                    temp_R = np.array([[ 1.,  0.,  0.],
                        [ 0.,  0.,  1.],
                        [ 0., -1.,  0.]], dtype=np.float32)
                    R = temp_R @ cam_orientation.rotation_matrix
                    q = Quaternion(matrix=R)
                    _,  _, cam_angle = euler_from_quaternion(*[getattr(q, key) for key in ['x', 'y', 'z', 'w']])
                    cam_angle =  -cam_angle

                    alpha = cam_angle - np.arctan2(-cam_center[2], cam_center[0]) - 0.5 * np.pi
                    if alpha > np.pi:
                        alpha -= 2 * np.pi
                    if alpha <= -np.pi:
                        alpha += 2 * np.pi
                    obj_dict = dict(
                        category_name = CATE_MAPPING[frame['annos']['names'][i]],
                        bbox2d = bbox2d.tolist(),
                        visibility_level = 0,
                        theta = cam_angle,
                        alpha = alpha,
                        xyz = cam_center.tolist(),
                        whl = whl.tolist(),
                    )
                    annotations.append(obj_dict)
                N = len(annotations)
                if N == 0:
                    main_object['annotations'].append([])
                    continue
                bboxes_2d_tensor = torch.tensor(
                     [obj_dict['bbox2d'] for obj_dict in annotations]
                ).float().reshape(N, 4)
                z_tensor = torch.tensor(
                     [obj_dict['xyz'][2] for obj_dict in annotations]
                ).float().reshape(N)
                
                occlusions = compute_occlusion(bboxes_2d_tensor, z_tensor).cpu().numpy()
                main_object['annotations'].append(
                    [annotations[idx] for idx in range(len(occlusions)) if occlusions[idx] < 0.6]
                )

    print(len(main_object['images']))
    json.dump(main_object, open(json_path, 'w'))


if __name__ == '__main__':
    #kitti_obj_dir = '/data/kitti_obj'
    base_dir = '/data/ONCE'
    json_path = 'once_object.json'
    main(base_dir, json_path)
