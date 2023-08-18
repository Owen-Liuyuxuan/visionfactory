import numpy as np
from numpy import random
import cv2
from mono3d.model.utils import theta2alpha_3d
from vision_base.data.augmentations.utils import flip_relative_pose

class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size, preserve_aspect_ratio=True,
                       force_pad=True,
                       image_keys=['image'],
                       calib_keys=[],
                       gt_image_keys=[],
                       object_keys=[],
                        **kwargs):
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.force_pad = force_pad
        self.image_keys=image_keys
        self.calib_keys=calib_keys
        self.gt_image_keys=gt_image_keys
        self.object_keys=object_keys

    def __call__(self, data):
        image = data[self.image_keys[0]]
        data[('image_resize', 'original_shape')] = np.array([image.shape[0], image.shape[1]]).astype(np.int64)
        ## Set up reshape output
        if self.preserve_aspect_ratio:
            scale_factor_x = self.size[0] / image.shape[0]
            scale_factor_y = self.size[1] / image.shape[1]
            if self.force_pad:
                scale_factor = min(scale_factor_x, scale_factor_y)
                mode = 'pad_0' if scale_factor_x > scale_factor_y else 'pad_1'
            else:
                scale_factor = scale_factor_x
                mode = 'crop_1' if scale_factor_x > scale_factor_y else 'pad_1'

            h = np.round(image.shape[0] * scale_factor).astype(int)
            w = np.round(image.shape[1] * scale_factor).astype(int)
            
            scale_factor_yx = (scale_factor, scale_factor)
        else:
            scale_factor_yx = (self.size[0] / image.shape[0], self.size[1] / image.shape[1])
            mode = 'none'
            h = self.size[0]
            w = self.size[1]
        
        data[('image_resize', 'effective_size')] = np.array([h, w]).astype(np.int64)

        # resize
        for key in self.image_keys:
            data[key] = cv2.resize(data[key], (w, h))

        for key in self.gt_image_keys:
            data[key] = cv2.resize(data[key], (w, h), interpolation=cv2.INTER_NEAREST)


        if len(self.size) > 1:

            for key in (self.image_keys + self.gt_image_keys):
                image = data[key]
                # crop in
                if mode=='crop_1':
                    data[key] = image[:, 0:self.size[1]]
                
                # pad
                if mode == 'pad_1':
                    padW = self.size[1] - image.shape[1]
                    if len(image.shape) == 2:
                        data[key] = np.pad(image,  [(0, 0), (0, padW)], 'constant')
                    
                    elif len(image.shape) == 3:
                        data[key] = np.pad(image,  [(0, 0), (0, padW), (0, 0)], 'constant')

                if mode == 'pad_0':
                    padH = self.size[0] - image.shape[0]
                    if len(image.shape) == 2:
                        data[key] = np.pad(image,  [(0, padH), (0, 0)], 'constant')
                    
                    elif len(image.shape) == 3:
                        data[key] = np.pad(image,  [(0, padH), (0, 0), (0, 0)], 'constant')

        for key in self.calib_keys:
            P = data[key]
            P[0, :] = P[0, :] * scale_factor_yx[1]
            P[1, :] = P[1, :] * scale_factor_yx[0]
            data[key] = P

        for key in self.object_keys:
            bbox_2d_ratio = np.array([scale_factor_yx[1], scale_factor_yx[0],
                                      scale_factor_yx[1], scale_factor_yx[0]])
            annotations = data[key]
            for i in range(len(annotations)):
                annotations[i]['bbox2d'] = annotations[i]['bbox2d'] * bbox_2d_ratio
            data[key] = annotations
        return data

class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty. It will also flip world in 3D

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob,
                image_keys=['image'],
                calib_keys=[],
                gt_image_keys=[],
                object_keys=[],
                lidar_keys=[],
                pose_axis_pairs=[],
                is_switch_left_right=True,
                stereo_image_key_pairs=[], #only used in "is_switch_left_right==True"
                stereo_calib_key_pairs=[], #only used in "is_switch_left_right==True"
                **kwargs
                ):
        self.mirror_prob = mirror_prob
        self.image_keys = image_keys
        self.calib_keys = calib_keys
        self.gt_image_keys = gt_image_keys
        self.object_keys = object_keys
        self.lidar_keys = lidar_keys
        self.pose_axis_pairs  = pose_axis_pairs
        self.is_switch_lr = is_switch_left_right
        self.stereo_image_key_pairs = stereo_image_key_pairs
        self.stereo_calib_key_pairs = stereo_calib_key_pairs

    def __call__(self, data):

        height, width, _ = data[self.image_keys[0]].shape

        if random.rand() <= self.mirror_prob:
            
            for key in (self.image_keys + self.gt_image_keys):
                data[key] = np.ascontiguousarray(data[key][:, ::-1])

            for key in self.calib_keys:
                P = data[key]
                P[0, 3] = -P[0, 3]
                P[0, 2] = width - P[0, 2] - 1
                data[key] = P

            for key in self.lidar_keys:
                data[key] = -data[key][..., 0] # Assume the last channel in lidar is "x" in the width direction
            
            for key, axis_num in self.pose_axis_pairs:
                data[key] = flip_relative_pose(data[key], axis_num)

            if self.is_switch_lr:
                for key_l, key_r in (self.stereo_image_key_pairs + self.stereo_calib_key_pairs):
                    data[key_l], data[key_r] = data[key_r], data[key_l]

            for key in self.object_keys:
                annotations = data[key]
                image_shape_array = np.array([width, height, width, height]) # noqa: F841
                for i in range(len(annotations)):
                    bbox_l = annotations[i]['bbox2d'][0]
                    bbox_r = annotations[i]['bbox2d'][2]

                    bbox_l, bbox_r = width - bbox_r - 1, width - bbox_l - 1
                    annotations[i]['bbox2d'][0] = bbox_l

                    annotations[i]['bbox2d'][2] = bbox_r

                    if 'xyz' in annotations[i]: # if we have 3D
                        x, y, z = annotations[i]['xyz']
                        x1 = -x
                        ry = annotations[i]['theta']
                        ry = - np.pi - ry if ry < 0 else np.pi - ry
                        while ry > np.pi: ry -= np.pi * 2
                        while ry < -np.pi: ry += np.pi * 2
                        annotations[i]['xyz'] = np.array([x1, y, z])
                        annotations[i]['theta'] = ry
                        annotations[i]['alpha'] = theta2alpha_3d(ry, x1, z, data[self.calib_keys[0]])
                data[key] = annotations
        
        return data

class RandomWarpAffine(object):
    """
        Randomly random scale and random shift the image. Then resize to a fixed output size.
    """
    def __init__(self, scale_lower=0.6, scale_upper=1.4, shift_border=128, output_w=1280, output_h=384,
                image_keys=['image'],
                gt_image_keys=[],
                calib_keys=[],
                object_keys=[],
                border_mode=cv2.BORDER_CONSTANT,
                random_seed = None,
                **kwargs):
        self.scale_lower    = scale_lower
        self.scale_upper    = scale_upper
        self.shift_border   = shift_border
        self.output_w       = output_w
        self.output_h       = output_h

        self.image_keys = image_keys
        self.gt_image_keys = gt_image_keys
        self.calib_keys = calib_keys
        self.object_keys    = object_keys
        self.border_mode = border_mode

        # If we provide a fix random seed, for different augmentation instances, they will provide random warp in the same way.
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]

        s_original = max(height, width)
        scale = s_original * self.rng.uniform(self.scale_lower, self.scale_upper)
        center_w = self.rng.integers(low=self.shift_border, high=width - self.shift_border)
        center_h = self.rng.integers(low=self.shift_border, high=height - self.shift_border)

        final_scale = max(self.output_w, self.output_h) / scale
        final_shift_w = self.output_w / 2 - center_w * final_scale
        final_shift_h = self.output_h / 2 - center_h * final_scale
        affine_transform = np.array(
            [
                [final_scale, 0, final_shift_w],
                [0, final_scale, final_shift_h]
            ], dtype=np.float32
        )

        for key in self.image_keys:
            data[key] = cv2.warpAffine(
                data[key], affine_transform, (self.output_w, self.output_h), flags=cv2.INTER_LINEAR, borderMode=self.border_mode
            )

        for key in self.gt_image_keys:
            data[key] = cv2.warpAffine(
                data[key], affine_transform, (self.output_w, self.output_h), flags=cv2.INTER_NEAREST, borderMode=self.border_mode
            )

        for key in self.calib_keys:
            P = data[key]
            P[0:2, :] *= final_scale
            P[0, 2] = P[0, 2] + final_shift_w               # cy' = cy - dv
            P[0, 3] = P[0, 3] + final_shift_w * P[2, 3] # ty' = ty - dv * tz
            P[1, 2] = P[1, 2] + final_shift_h               # cy' = cy - dv
            P[1, 3] = P[1, 3] + final_shift_h * P[2, 3] # ty' = ty - dv * tz
            data[key] = P

        shift = np.array([final_shift_w, final_shift_h, final_shift_w, final_shift_h])
        for key in self.object_keys:
            annotations = data[key]
            for i in range(len(annotations)):
                annotations[i]['bbox2d'] = annotations[i]['bbox2d'] * final_scale + shift
            data[key] = annotations

        return data

class FilterObject(object):
    """
        Filtering out object completely outside of the box;
    """
    def __init__(self,
                image_keys=['image'],
                object_keys=[],
                **kwargs
                ):
        self.image_keys = image_keys
        self.object_keys = object_keys


    def __call__(self, data):
        height, width, _ = data[self.image_keys[0]].shape

        for key in self.object_keys:
            annotations = data[key]
            new_annotations = []
            for obj in annotations:
                bbox2d = obj['bbox2d']
                is_outside = bbox2d[0] > width or bbox2d[1] > height or bbox2d[2] < 0 or bbox2d[3] < 0
                if not is_outside:
                    new_annotations.append(obj)
            data[key] = new_annotations
        
        return data
