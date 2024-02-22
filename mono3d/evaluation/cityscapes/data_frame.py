"""
    This is a interface class between the cityscape API and a definition similar to KITTI data
"""

from cityscapesscripts.helpers.annotation import Camera, CsBbox3d, CsBbox2d, Annotation, CsObjectType
from cityscapesscripts.helpers.box3dImageTransform import (
    CRS_S,
    Box3dImageTransform,
    get_projection_matrix
)
import json
from typing import List, Dict
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import numpy as np
import cv2

class CityScapeFrame:
    def __init__(self, gt_bbox_json:Dict):

        ## Camera:
        sensor = gt_bbox_json['sensor']
        self.baseline = sensor['baseline']
        self.camera = Camera(
            sensor['fx'], sensor['fy'], sensor['u0'], sensor['v0'],
            sensor['sensor_T_ISO_8855']
        )

        ## objects:
        self.objects:List[CsBbox3d] = []
        for json_obj in gt_bbox_json['objects']:
            obj = CsBbox3d()
            obj.fromJsonText(json_obj)
            if obj.center[0] < 100: # filter out far away objects
                self.objects.append(obj)

        self.init_transformers()

    def init_transformers(self):
        self.framed_objects:List[Box3dImageTransform] = []
        for i in range(len(self.objects)):
            frame_obj = Box3dImageTransform(self.camera)
            frame_obj.initialize_box_from_annotation(self.objects[i])
            self.framed_objects.append(frame_obj)

    def __len__(self):
        return len(self.framed_objects)

    def __getitem__(self, index):
        return self.framed_objects[index]

    def __str__(self):
        string = ""
        for obj in self.objects:
            string += obj.__str__() + "\n"
        return string[:-1]
    
    def update(self):
        for frame_obj in self.framed_objects:
            frame_obj.update()

    @property
    def P2(self):
        K_matrix = get_projection_matrix(self.camera) #[3, 3]
        P = np.eye(4)
        P[0:3, 0:3] = K_matrix
        return P
    
    @P2.setter
    def P2(self, P2):
        self.camera.fx = P2[0, 0]
        self.camera.fy = P2[1, 1]
        self.camera.u0 = P2[0, 2]
        self.camera.v0 = P2[1, 2]
        self.init_transformers()

    @property
    def bbox2d(self):
        """
            2D bounding box property
        """
        bboxes = []
        for frame_object in self.framed_objects:
            corner_bbox2d = frame_object._box_points_2d #[8, 2]
            left   = np.min(corner_bbox2d[:, 0])
            right  = np.max(corner_bbox2d[:, 0])
            top    = np.min(corner_bbox2d[:, 1])
            bottom = np.max(corner_bbox2d[:, 1])
            bboxes.append([left, top, right, bottom])
        return np.array(bboxes).reshape([-1, 4]) #[N, 4]

    @property
    def bbox3d(self) -> Dict[str, np.ndarray]:
        """get bbox 3d informations

        Returns:
            Dict:
                centers, #[N, 3] in camera frame, x,y,z
                camera_centers  #[N, 3] in camera frame, cx, cy, z
                quaternions  #[N, 4] w, x,y,z in camera frame
                whls  #[N, 3] in camera frame
        """
        centers = []
        camera_centers = []
        quaternions = []
        ypras = []
        whls = []

        K = get_projection_matrix(self.camera)
        for frame_object in self.framed_objects:
            size_S, center_S, rotation_S = frame_object.get_parameters(coordinate_system=CRS_S)
            l, w, h = size_S
            whls.append([w, h, l])
            centers.append(center_S)
            quaternions.append(list(rotation_S)) #[w, x, y, z]

            yaw, pitch, roll = Quaternion(matrix=frame_object._rotation_matrix).yaw_pitch_roll # in CRS_C
            x, y, z = center_S
            alpha = -np.pi + np.arctan2(z, x) - yaw # -(yaw_cs + \pi/ 2) = yaw_kitti; \alpha_kitti = \theta_kitti + arctan2(z, x) - \pi / 2; z,x in camera frame
            ypras.append([yaw, pitch, roll, alpha]) #[z, y, x]
            
            image_center = K@np.array(center_S)
            image_center /= image_center[2]
            camera_centers.append([image_center[0], image_center[1], center_S[2]]) #[cx, cy, z]

        return_dict = dict(
            centers=np.array(centers).reshape([-1, 3]), #[N, 3] in camera frame
            camera_centers=np.array(camera_centers).reshape([-1, 3]),  #[N, 3] in camera frame
            quaternions = np.array(quaternions).reshape([-1, 4]), #[N, 4] w, x,y,z in camera frame
            ypras = np.array(ypras).reshape([-1, 4]), #zyxa
            whls = np.array(whls).reshape([-1, 3]), #[N, 3] in camera frame
        )
        return return_dict

    @property
    def corner_bbox2d(self) -> np.ndarray:
        bboxes = []
        for frame_object in self.framed_objects:
            corner_bbox2d = frame_object._box_points_2d #[8, 2]
            bboxes.append(corner_bbox2d)

        return np.array(bboxes).reshape([-1, 8, 2])  #[N, 8, 2]
    
    @staticmethod
    def draw_corner_bbox2d(image, bboxes:np.ndarray, color = (255, 255, 0)):
        """ Draw the 3D corner on the image *inplace*
        image: np.ndarray

        # loc = ["BLB", "BRB", "FRB", "FLB", "BLT", "BRT", "FRT", "FLT"]
        """
        bboxes = bboxes.astype(np.int32)
        for box in bboxes:
            points = [tuple(box[i, :]) for i in range(8)]
            for i in range(0, 4):
                cv2.line(image, points[i], points[(i + 1) % 4], color, 2) #[bottom]
                cv2.line(image, points[(i + 4)], points[(i + 1) % 4 + 4], color, 2) #[top]
            cv2.line(image, points[0], points[4], color)
            cv2.line(image, points[1], points[5], color)
            cv2.line(image, points[2], points[6], color)
            cv2.line(image, points[3], points[7], color)
        return image

    @property
    def labels(self):
        return [obj.label for obj in self.objects]

    def filter(self, obj_types:List[str]):
        new_objects:List[CsBbox3d] = []
        for obj in self.objects:
            if obj.label in obj_types:
                new_objects.append(obj)
        self.objects = new_objects
        self.init_transformers()

    def get_label_indexes(self, obj_types:List[str]):
        indexes = []
        for label in self.labels:
            if label in obj_types:
                indexes.append(obj_types.index(label))
            else:
                indexes.append(-1)
        return indexes

    def flip_objects(self):
        """ Flip the image, only applicable in mono detection.
        """
        for i in range(len(self.objects)):
            # change
            quad = Quaternion(self.objects[i].rotation)
            yaw_pitch_roll = quad.yaw_pitch_roll
            rot = Rotation.from_euler('zyx', [yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2]])
            quat_flipped = list(rot.as_quat()[[3, 0, 1, 2]]) #final output [w, x, y, z]
            self.objects[i].rotation = quat_flipped
        
        self.init_transformers()

    def set_objects(self,
                    bbox2d:np.ndarray, #[N, 4]
                    whls:np.ndarray, #[N, 3]
                    centers:np.ndarray, #[N, 3]
                    #quaternions:np.ndarray, # [N, 4]
                    pitch_roll_alphas: np.ndarray,
                    scores:np.ndarray, #[N]
                    labels:List[str], #[N]
                    coordinate_system:int=CRS_S):
        self.framed_objects = []
        self.objects = []
        N = len(whls)
        assert len(centers) == N and len(pitch_roll_alphas) == N and len(labels) == N
        for i in range(N):
            if labels[i] not in ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'motorcycle', 'trailer', 'pedestrian']:
                continue
            frame_obj = Box3dImageTransform(self.camera)
            size = whls[i, [2, 0, 1]]
            pitch, roll, alpha = pitch_roll_alphas[i]
            center = centers[i]
            x, y, z = center
            yaw = -np.pi + np.arctan2(z, x) - alpha
            #print(yaw, pitch, roll, alpha)
            rot = Rotation.from_euler('zyx', [yaw, pitch, roll])
            quaternion = list(rot.as_quat()[[3, 0, 1, 2]])

            frame_obj.initialize_box(size, [1, 0, 0, 0], center, coordinate_system) #[dummy orientation]
            transfered_center = frame_obj._center
            transfered_quaternion = Quaternion(quaternion) # we directly predict orientation in vehicle frame
            frame_obj._rotation_matrix = transfered_quaternion.rotation_matrix

            obj_2d = CsBbox2d()
            bbox2d_list = bbox2d[i].tolist() #[x1, y1,  x2, y2]
            bbox2d_list[2] -= bbox2d_list[0]
            bbox2d_list[3] -= bbox2d_list[1] # becomes [x1, y1, w, h]
            obj_2d.bbox_amodal_xywh = bbox2d_list
            obj_2d.bbox_modal_xywh = bbox2d_list

            obj = CsBbox3d()
            obj.label = labels[i]
            obj.rotation = list(transfered_quaternion)
            obj.score = float(scores[i])
            obj.bbox_2d = obj_2d
            obj.center = transfered_center.tolist()
            obj.dims = size.tolist()

            self.framed_objects.append(frame_obj)
            self.objects.append(obj)

    def to_json(self, file_path:str):
        annotation = Annotation(objType=CsObjectType.BBOX3D)
        annotation.objects = self.objects

        jsonDict:dict = {}
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict:dict = {'2d':{}, '3d':{}}
            objDict['label'] = obj.label
            objDict['score'] = obj.score
            objDict['instanceId'] = obj.instanceId
            objDict['2d']['amodal'] = obj.bbox_2d.bbox_amodal_xywh
            objDict['2d']['modal'] = obj.bbox_2d.bbox_modal_xywh
            objDict['3d']['center'] = obj.center
            objDict['3d']['dimensions'] = obj.dims
            objDict['3d']['rotation'] = obj.rotation
            jsonDict['objects'].append(objDict)
        json.dump(jsonDict, open(file_path, 'w'))

        

if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    plt.interactive(True)
    left_image_path = "/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
    right_image_path = '/data/cityscapes/rightImg8bit/train/aachen/aachen_000000_000019_rightImg8bit.png'
    gtbox3d_path    = '/data/cityscapes/gtBbox3d/train/aachen/aachen_000000_000019_gtBbox3d.json'

    left_image = np.array(Image.open(left_image_path))
    right_image = np.array(Image.open(right_image_path))
    bbox_data = json.load(open(gtbox3d_path, 'r'))

    frame = CityScapeFrame(bbox_data)

    def draw_bbox2d_to_image(image, bboxes2d, color=(255, 0, 255)):
        """ Draw 2D bounding boxes on image not inplaced
        
        Inputs:
            image[np.ndArray] : image to be copied and drawed on. should be in [uint8 BGR] because we are using cv2.
            bboxes2d[np.ndArray] : bounding boxes of [N, 4] -> [x1, y1, x2, y2] pixel values.
            color[Tuple[uint8]] : [BGR] colors
        Returns:
            drawed_image [np.ndArray]: drawed image as a copy.
        """
        drawed_image = image.copy()
        for box2d in bboxes2d:
            cv2.rectangle(drawed_image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color, 3)
        return drawed_image

    # before flipping
    bbox2d = frame.bbox2d
    corner_points = frame.corner_bbox2d

    plt.subplot(3, 1, 1)
    drawed_image = draw_bbox2d_to_image(left_image, bbox2d)
    camera_centers = frame.bbox3d['camera_centers']
    for i in range(len(frame)):
        if frame.labels[i] == 'car':
            plt.plot(camera_centers[i, 0], camera_centers[i, 1], 'r.')
    plt.imshow(drawed_image)

    # Flipped
    P2 = frame.P2
    P2[0, :] *= -1
    P2[0, 2] += left_image.shape[1] - 1
    frame.P2 = P2
    frame.flip_objects()

    plt.subplot(3, 1, 2)
    bbox2d = frame.bbox2d
    corner_points = frame.corner_bbox2d
    drawed_image = draw_bbox2d_to_image(left_image[:, ::-1, :], bbox2d)
    camera_centers = frame.bbox3d['camera_centers']
    for i in range(len(frame)):
        if i > 0:
            break
        plt.plot(camera_centers[i, 0], camera_centers[i, 1], 'r.')
    plt.imshow(drawed_image)


    # Resize
    P2 = frame.P2
    P2[0, :] *= 0.5
    P2[1, :] *= 0.5
    frame.P2 = P2
    resized_image = cv2.resize(left_image, (1024, 512))

    plt.subplot(3, 1, 3)
    bbox2d = frame.bbox2d
    corner_points = frame.corner_bbox2d
    drawed_image = draw_bbox2d_to_image(resized_image[:, ::-1, :], bbox2d)
    camera_centers = frame.bbox3d['camera_centers']
    for i in range(len(frame)):
        if i > 0:
            break
        plt.plot(camera_centers[i, 0], camera_centers[i, 1], 'r.')
    plt.imshow(drawed_image)
    plt.show(block=True)