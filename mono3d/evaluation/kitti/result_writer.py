import shutil
import os
import numpy as np
class KittiResultWriter(object):
    def __init__(self, result_dir:str, 
                 obj_mapping={'car': 'Car', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist', 'truck': 'Truck'}):
        self.result_dir = result_dir
        self.obj_mapping = obj_mapping

        self.rebuild_dir()
    
    def rebuild_dir(self):
        if os.path.isdir(self.result_dir):
            shutil.rmtree(self.result_dir, ignore_errors=True)
            print("clean up the recorder directory of {}".format(self.result_dir))
        os.mkdir(self.result_dir)

    def write(self, index, scores, bbox_2d, obj_types, bbox_3d_state_3d=None):
        name = f"{index:06d}.txt"
        file_name = os.path.join(self.result_dir, name)
        text_to_write = ""
        if bbox_3d_state_3d is None:
            bbox_3d_state_3d = np.ones([bbox_2d.shape[0], 8], dtype=int)
            bbox_3d_state_3d[:, 3:6] = -1
            bbox_3d_state_3d[:, 0:3] = -1000
            bbox_3d_state_3d[:, 6:8]   = -10
        else:
            for i in range(len(bbox_2d)):
                bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5*bbox_3d_state_3d[i][4] # kitti receive bottom center
        if len(scores) > 0:
            for i in range(len(bbox_2d)):
                bbox = bbox_2d[i]
                if obj_types[i] not in self.obj_mapping:
                    continue
                else:
                    kitti_category = self.obj_mapping[obj_types[i]]
                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} \n').format(
                    kitti_category, bbox_3d_state_3d[i][-1], bbox[0], bbox[1], bbox[2], bbox[3],
                    bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                    bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                    bbox_3d_state_3d[i][7], scores[i])
        with open(file_name, 'w') as file:
            file.write(text_to_write)