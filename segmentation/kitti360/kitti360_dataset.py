from __future__ import print_function, division
import os
from easydict import EasyDict
from copy import deepcopy

from torch.utils.data import Dataset # noqa: F401
from segmentation.evaluation.labels import labels as kitti360_labels
from vision_base.data.datasets.utils import read_image
from vision_base.utils.builder import build

def read_split_file(file_path, sample_over=1):
    imdb = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % sample_over == 0:
                splitted = line.strip().split(' ')
                obj = dict()
                obj['image_path'] = (splitted[0])
                obj['gt_path'] = (splitted[1])
                imdb.append(obj)
    return imdb

class KITTI360SegDataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(KITTI360SegDataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', '/data/KITTI-360')
        self.sample_over = getattr(data_cfg, 'sample_over', 1)
        self.meta_file   = getattr(data_cfg, 'split_file', '/data/KITTI-360/data_2d_semantics/train/2013_05_28_drive_train_frames.txt')
        self.imdb = read_split_file(self.meta_file, self.sample_over)

        self.transform = build(**data_cfg.augmentation)
        self.label_text_set = [label.name for label in kitti360_labels[:-1]]
    
    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):
        obj = self.imdb[index]
        data = dict()
        data['image'] = read_image(os.path.join(self.base_path, obj['image_path']))
        data['gt_image'] = read_image(os.path.join(self.base_path, obj['gt_path']))
        data['original_shape'] = data['image'].shape
        data['label_sets'] = self.label_text_set
        data = self.transform(deepcopy(data))
        return data
