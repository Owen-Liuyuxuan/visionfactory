import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

KITTI360_base = "/data/fusionportable_kitti360/"

for setting in ['room', 'quadro']:

    image_dir = os.path.join(KITTI360_base, setting, 'data_2d_raw')
    pose_dir  = os.path.join(KITTI360_base, setting, 'data_poses')
    calib_dir = os.path.join(KITTI360_base, setting, 'calibration')

    sequence_names = os.listdir(pose_dir)
    sequence_names.sort()

    np.random.seed(0)
    random_permutation = np.random.permutation(range(len(sequence_names)))
    split_ratio = 1.0
    validation_gaps = 20
    num_train_split = int(split_ratio * len(sequence_names))
    train_split = random_permutation[0:num_train_split]
    val_split   = random_permutation[num_train_split:]

    
    train_file = f'{setting}.txt'
    steps = [0,1,-1,3,-3]
    train_lines = ["sequence_name,pose_index,0,1,-1,3,-3\n"]
    for sequence_index in train_split:
        sequence_name = sequence_names[sequence_index]
        pose_file = os.path.join(pose_dir, sequence_name, 'poses.txt')
        key_frame_indexes = []
        with open(pose_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key_frame_indexes.append(int(line.strip().split(" ")[0]))
        image_dir_0 = os.path.join(image_dir, sequence_name, 'image_00')
        for i in range(max(steps), len(key_frame_indexes)-max(steps)):
            image_paths = [
                os.path.join(image_dir_0, f"{i:010d}.png") for i in [key_frame_indexes[i+j] for j in steps]
            ]
            image_exist=True
            for img in image_paths:
                if not os.path.isfile(img):
                    image_exist=False
            if image_exist:
                wline = f"{sequence_name},{i}"
                for k_i in [key_frame_indexes[i+j] for j in steps]:
                    wline += f",{k_i}"
                wline += "\n"
                train_lines.append(wline)
        
        with open(train_file, 'w') as f:
            f.writelines(train_lines)
