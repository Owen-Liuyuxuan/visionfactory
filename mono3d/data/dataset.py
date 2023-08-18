from __future__ import print_function, division
import os
import torch
import numpy as np
from easydict import EasyDict
import json
from typing import List, Tuple, Dict

from copy import deepcopy
from dgp.datasets.synchronized_dataset import SynchronizedScene, SynchronizedSceneDataset
from vision_base.utils.builder import build
from mono3d.model.utils import BBox3dProjector, theta2alpha_3d
from mono3d.model.rtm3d_utils import gen_hm_radius, gaussian_radius
from vision_base.data.datasets.utils import read_image


def select_by_split_file(dataset_base, split_file, selectable_keys=['images', 'annotations', 'calibrations']):
    new_dataset_base = dict()
    for key in dataset_base:
        if key not in selectable_keys:
            new_dataset_base[key] = dataset_base[key]
        else:
            new_dataset_base[key] = []
    with open(split_file, 'r') as f:
        split_lines = f.readlines()
        for line in split_lines:
            index = int(line)
            for key in selectable_keys:
                new_dataset_base[key].append(dataset_base[key][index])
    return new_dataset_base

class JsonMonoDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(JsonMonoDataset, self).__init__()
        json_path = getattr(data_cfg, 'json_path', '/data/data.json')
        self.dataset_base = json.load(
            open(json_path, 'r')
        )
        self.labeled_objects = self.dataset_base['labeled_objects']
        self.is_labeled_3d = np.array(self.dataset_base['is_labeled_3d'])
        self.split_file = getattr(data_cfg, 'split_file', None)
        if self.split_file is not None:
            self.dataset_base = select_by_split_file(self.dataset_base, self.split_file)
        self.images = self.dataset_base['images']
        self.annotatations = self.dataset_base['annotations']
        for frame_index in range(len(self.annotatations)):
            new_object_list = []
            for obj_index in range(len(self.annotatations[frame_index])):
                for key in self.annotatations[frame_index][obj_index]:
                    attr = self.annotatations[frame_index][obj_index][key]
                    if isinstance(attr, list):
                        self.annotatations[frame_index][obj_index][key] = np.array(attr)
                visibility_level = self.annotatations[frame_index][obj_index]['visibility_level']
                if visibility_level <= 2:
                    new_object_list.append(self.annotatations[frame_index][obj_index])
            self.annotatations[frame_index] = new_object_list

        self.calibs = self.dataset_base['calibrations']
        self.training_types = getattr(data_cfg, 'training_types', ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'])
        self.label_training_mask = np.array(
            [class_name in self.labeled_objects for class_name in self.training_types], dtype=np.bool_
        )

        ## MonoFlex Params
        self.num_classes = len(self.training_types)
        self.num_vertexes = 10
        self.max_objects = getattr(data_cfg, 'max_objects', 32)
        self.projector = BBox3dProjector()
        self.projector.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1], #0
            [ 1, -1, -1],  #1
            [ 1,  1, -1],  #2
            [ 1,  1,  1],  #3
            [ 1, -1,  1],  #4
            [-1, -1,  1],  #5
            [-1,  1,  1],  #6
            [-1,  1, -1],  #7
            [ 0,  1,  0],  #8
            [ 0, -1,  0],  #9
            [ 0,  0,  0]]  #10
        ).float()  )# 10, 3

        self.main_calibration_key = getattr(data_cfg, 'main_calibration_key', 'P2')

        data3d_json = getattr(data_cfg, 'data3d_json', '/data/whl.json')
        self.empirical_whl = json.load(
            open(data3d_json, 'r')
        )

        self.transform = build(**data_cfg.augmentation)
    
    def _build_target(self, image:np.ndarray, P2:np.ndarray, transformed_label:List[Dict], scale=4)-> dict:
        """Encode Targets for MonoFlex

        Args:
            image (np.ndarray): augmented image [3, H, W]
            P2 (np.ndarray): Calibration matrix [3, 4]
            transformed_label (List[KittiObj]): A list of kitti objects.
            scale (int, optional): Downsampling scale. Defaults to 4.

        Returns:
            dict: label dicts
        """
        num_objects = len(transformed_label)
        hm_h, hm_w = image.shape[1] // scale, image.shape[2] // scale

        # setup empty targets
        hm_main_center = np.zeros((self.num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((self.num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)
        bboxes2d = np.zeros((self.max_objects, 4), dtype=np.float32)
        cls_indexes = np.zeros((self.max_objects, 1), dtype=np.int64)
        fcos_bbox2d_target = np.zeros((self.max_objects, 4), dtype=np.float32)
        location = np.zeros((self.max_objects, 3), dtype=np.float32)
        orientation = np.zeros((self.max_objects, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objects, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objects, 2), dtype=np.float32)
        ver_coor = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((self.max_objects * self.num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((self.max_objects * self.num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((self.max_objects * self.num_vertexes), dtype=np.int64)
        keypoints_depth_mask = np.zeros((self.max_objects, 3), dtype=np.float32)

        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)
        log_dimension = np.zeros((self.max_objects, 3), dtype=np.float32)
        mean_dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        rots = np.zeros((self.max_objects, 2), dtype=np.float32) #[sin, cos]

        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        whs = np.zeros((self.max_objects, 2), dtype=np.float32)

        # compute vertexes
        for obj in transformed_label:
            obj['alpha'] = theta2alpha_3d(obj['theta'], obj['xyz'][0], obj['xyz'][2], P2)
        bbox3d_origin = torch.tensor([[obj['xyz'][0], obj['xyz'][1], obj['xyz'][2], obj['whl'][0], obj['whl'][1], obj['whl'][2], obj['alpha']] for obj in transformed_label], dtype=torch.float32).reshape(-1, 7)
        abs_corner, homo_corner, theta = self.projector.forward(bbox3d_origin, P2.clone())

        for k in range(num_objects):
            obj = transformed_label[k]
            cls_id = self.training_types.index(obj['category_name'])
            bbox = obj['bbox2d']
            orientation[k] = obj['theta']
            dim  = obj['whl']
            mean_dim = self.empirical_whl[obj['category_name']]['whl']
            log_dim  = np.log((obj['whl']) / mean_dim)
            alpha= obj['alpha']

            if np.sin(alpha) < 0.5: #alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                rotbin[k, 0] = 1
                rotres[k, 0] = alpha - (-0.5 * np.pi)
            if np.sin(alpha) > -0.5: # alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                rotbin[k, 1] = 1
                rotres[k, 1] = alpha - (0.5 * np.pi)

            bbox = bbox / scale  # on the heatmap
            bboxes2d[k] = bbox
            cls_indexes[k] = cls_id
            
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                radius = 1  # Just dummy

                location[k] = bbox3d_origin[k, 0:3].float().cpu().numpy()

                radius = gaussian_radius((np.ceil(bbox_h), np.ceil(bbox_w)))
                radius = max(0, int(radius))
                ## Different from RTM3D:
                # Generate heatmaps for 10 vertexes
                vertexes_2d = homo_corner[k, 0:10, 0:2].numpy()

                vertexes_2d = vertexes_2d / scale  # on the heatmap

                # keypoints mask: keypoint must be inside the image and in front of the camera
                keypoints_x_visible = (vertexes_2d[:, 0] >= 0) & (vertexes_2d[:, 0] <= hm_w)
                keypoints_y_visible = (vertexes_2d[:, 1] >= 0) & (vertexes_2d[:, 1] <= hm_h)
                keypoints_z_visible = (abs_corner[k, 0:10, 2].numpy() > 0)
                keypoints_visible   = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
                keypoints_visible = np.append(
                    np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2)
                ) # "modified keypoint visible from monoflex"
                keypoints_depth_valid = np.stack(
                    (keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all())
                ).astype(np.float32)
                keypoints_visible = keypoints_visible.astype(np.float32)

                ## MonoFlex use the projected 3D as the center
                #center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center = homo_corner[k, 10, 0:2].numpy() / scale
                center_int = center.astype(np.int32)
                
                if not (0 <= center_int[0] < hm_w and 0 <= center_int[1] < hm_h):
                    continue

                # Generate heatmaps for main center
                gen_hm_radius(hm_main_center[cls_id], center, radius)
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                for ver_idx, ver in enumerate(vertexes_2d):
                    ver_int = ver.astype(np.int32)
                    
                    # targets for vertexes coordinates
                    ver_coor[k, ver_idx * 2: (ver_idx + 1) * 2] = ver - center_int  # Don't take the absolute values
                    ver_coor_mask[k, ver_idx * 2: (ver_idx + 1) * 2] = 1
                    
                    if (0 <= ver_int[0] < hm_w) and (0 <= ver_int[1] < hm_h):
                        gen_hm_radius(hm_ver[ver_idx], ver_int, radius)
                        
                        # targets for vertexes offset
                        ver_offset[k * self.num_vertexes + ver_idx] = ver - ver_int
                        ver_offset_mask[k * self.num_vertexes + ver_idx] = 1
                        # Indices of vertexes
                        indices_vertexes[k * self.num_vertexes + ver_idx] = ver_int[1] * hm_w + ver_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                ## targets for fcos 2d
                fcos_bbox2d_target[k] = np.array(
                    [center_int[0] - bbox[0], center_int[1] - bbox[1], bbox[2] - center_int[0], bbox[3] - center_int[1]]
                )
                # targets for dimension
                dimension[k] = dim
                log_dimension[k] = log_dim
                mean_dimension[k] = mean_dim

                # targets for orientation
                rots[k, 0] = np.sin(alpha)
                rots[k, 1] = np.cos(alpha)

                # targets for depth
                depth[k] = obj['xyz'][2]

                # targets for 2d bbox
                whs[k, 0] = bbox_w
                whs[k, 1] = bbox_h

                # Generate masks
                obj_mask[k] = 1
                keypoints_depth_mask[k] = keypoints_depth_valid

        # Follow official names
        targets = {
            'hm': hm_main_center,
            'hm_hp': hm_ver,
            'hps': ver_coor,
            'reg': cen_offset,
            'hp_offset': ver_offset,
            'dim': dimension, #whl
            'log_dim': log_dimension, #whl
            'mean_dim': mean_dimension, #whl
            'rots': rots, # sin cos alpha
            'rotbin': rotbin,
            'rotres': rotres,
            'dep': depth,
            'ind': indices_center,
            'hp_ind': indices_vertexes,
            'reg_mask': obj_mask,
            'hps_mask': ver_coor_mask,
            'hp_mask': ver_offset_mask,
            'kp_detph_mask': keypoints_depth_mask,
            'wh': whs,
            'bboxes2d': bboxes2d,
            'cls_indexes': cls_indexes,
            'bboxes2d_target': fcos_bbox2d_target,
            'location': location,
            'ori': orientation,
        }

        return targets


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = dict()
        
        data['image'] = read_image(self.images[index])
        data['objs_list'] = deepcopy(self.annotatations[index])
        calibration = self.calibs[index]
        data['P'] = deepcopy(
            np.array(calibration[self.main_calibration_key], dtype=np.float32)).reshape(3, 4)
        data['original_P'] = data['P'].copy()
        data = self.transform(data)

        target_dict = self._build_target(data['image'], data['P'], data['objs_list'])
        for key in target_dict:
            data[('target', key)] = target_dict[key]

        data[('target', 'labeled_mask')] = self.label_training_mask
        data[('target', 'is_labeled_3d')] = self.is_labeled_3d
        return data

class Json2DDataset(JsonMonoDataset):
    def _build_target(self, image:np.ndarray, P2:np.ndarray, transformed_label:List[Dict], scale=4)-> dict:
        """Encode Targets for MonoFlex. Make a Pseudo one that only deal with 2D data
        """
        num_objects = len(transformed_label)
        hm_h, hm_w = image.shape[1] // scale, image.shape[2] // scale

        # setup empty targets
        hm_main_center = np.zeros((self.num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((self.num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)
        bboxes2d = np.zeros((self.max_objects, 4), dtype=np.float32)
        cls_indexes = np.zeros((self.max_objects, 1), dtype=np.int64)
        fcos_bbox2d_target = np.zeros((self.max_objects, 4), dtype=np.float32)
        location = np.zeros((self.max_objects, 3), dtype=np.float32)
        orientation = np.zeros((self.max_objects, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objects, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objects, 2), dtype=np.float32)
        ver_coor = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((self.max_objects * self.num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((self.max_objects * self.num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((self.max_objects * self.num_vertexes), dtype=np.int64)
        keypoints_depth_mask = np.zeros((self.max_objects, 3), dtype=np.float32)

        dimension = np.ones((self.max_objects, 3), dtype=np.float32)
        log_dimension = np.zeros((self.max_objects, 3), dtype=np.float32)
        mean_dimension = np.ones((self.max_objects, 3), dtype=np.float32)

        rots = np.zeros((self.max_objects, 2), dtype=np.float32) #[sin, cos]

        depth = np.ones((self.max_objects, 1), dtype=np.float32)
        whs = np.ones((self.max_objects, 2), dtype=np.float32)

        for k in range(num_objects):
            obj = transformed_label[k]
            cls_id = self.training_types.index(obj['category_name'])
            bbox = obj['bbox2d']


            bbox = bbox / scale  # on the heatmap
            bboxes2d[k] = bbox
            cls_indexes[k] = cls_id

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                radius = 1  # Just dummy

                radius = gaussian_radius((np.ceil(bbox_h), np.ceil(bbox_w)))
                radius = max(0, int(radius))

                ## MonoFlex use the projected 3D as the center, but for dummy label, we only have image 2D.
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                
                if not (0 <= center_int[0] < hm_w and 0 <= center_int[1] < hm_h):
                    continue

                # Generate heatmaps for main center
                """
                In the case of 2D data, suppress all points outside of the 2D bounding boxes, 
                Default all points to be negative points. Heatmaps with correct class inside the 2D bounding boxes set to be -1. 
                """
                gen_hm_radius(hm_main_center[cls_id], center, radius)
                x_min, y_min, x_max, y_max = bbox.astype(np.int32)
                hm_main_center[cls_id][y_min:y_max, x_min:x_max] = -1
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                # targets for center offset
                cen_offset[k] = 0

                ## targets for fcos 2d
                fcos_bbox2d_target[k] = np.array(
                    [center_int[0] - bbox[0], center_int[1] - bbox[1], bbox[2] - center_int[0], bbox[3] - center_int[1]]
                )

                # Generate masks
                obj_mask[k] = 1

        # Follow official names
        targets = {
            'hm': hm_main_center,
            'hm_hp': hm_ver,
            'hps': ver_coor,
            'reg': cen_offset,
            'hp_offset': ver_offset,
            'dim': dimension, #whl
            'log_dim': log_dimension, #whl
            'mean_dim': mean_dimension, #whl
            'rots': rots, # sin cos alpha
            'rotbin': rotbin,
            'rotres': rotres,
            'dep': depth,
            'ind': indices_center,
            'hp_ind': indices_vertexes,
            'reg_mask': obj_mask,
            'hps_mask': ver_coor_mask,
            'hp_mask': ver_offset_mask,
            'kp_detph_mask': keypoints_depth_mask,
            'wh': whs,
            'bboxes2d': bboxes2d,
            'cls_indexes': cls_indexes,
            'bboxes2d_target': fcos_bbox2d_target,
            'location': location,
            'ori': orientation,
        }

        return targets

    def __getitem__(self, index):
        data = dict()
        
        data['image'] = read_image(self.images[index])
        data['objs_list'] = deepcopy(self.annotatations[index])
        calibration = self.calibs[index]
        data['P'] = deepcopy(
            np.array(calibration[self.main_calibration_key], dtype=np.float32)).reshape(3, 4)
        data['original_P'] = data['P'].copy()
        data = self.transform(data)

        target_dict = self._build_target(data['image'], data['P'], data['objs_list'])
        for key in target_dict:
            data[('target', key)] = target_dict[key]

        data[('target', 'labeled_mask')] = self.label_training_mask
        data[('target', 'is_labeled_3d')] = np.array(False)
        return data
class JsonTestDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(JsonTestDataset, self).__init__()
        json_path = getattr(data_cfg, 'json_path', '/data/data.json')
        self.dataset_base = json.load(
            open(json_path, 'r')
        )
        self.split_file = getattr(data_cfg, 'split_file', None)
        if self.split_file is not None:
            self.dataset_base = select_by_split_file(self.dataset_base, self.split_file)
        self.images = self.dataset_base['images']
        self.calibs = self.dataset_base['calibrations']

        self.main_calibration_key = getattr(data_cfg, 'main_calibration_key', 'P')

        self.transform = build(**data_cfg.augmentation)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = dict()
        
        data['image'] = read_image(self.images[index])
        calibration = self.calibs[index]
        data['P'] = deepcopy(
            np.array(calibration[self.main_calibration_key])).reshape(3, 4)
        data['original_P'] = data['P'].copy()
        data = self.transform(data)

        return data

class DGPTestDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(DGPTestDataset, self).__init__()
        json_path = getattr(data_cfg, 'json_path', '/data/data.json')
        split     = getattr(data_cfg, 'split', 'train')

        self.scene_files = json.load(open(json_path, 'r'))['scene_splits']['0']['filenames']
        self.base_dir_name = os.path.dirname(json_path)
        self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']

        self.dgp_dataset = SynchronizedSceneDataset(
            json_path, split=split, datum_names=self.camera_names
        )
        self.transform = build(**data_cfg.augmentation)

    def __len__(self):
        return len(self.dgp_dataset)

    def __getitem__(self, index):
        
        dgp_data = self.dgp_dataset[index]

        data = dict()
        data['image'] = np.array(dgp_data['rgb']) #[RGB]
        data['P'] = np.zeros([3, 4])
        data['P'][0:3, 0:3] = dgp_data['intrinsics']
        data['original_P'] = data['P'].copy()
        data = self.transform(data)

        return data