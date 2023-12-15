from __future__ import print_function, division
from easydict import EasyDict
from copy import deepcopy
import json

from torch.utils.data import Dataset
from vision_base.data.datasets.utils import read_image
from vision_base.utils.builder import build



class JsonifiedSegDataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(JsonifiedSegDataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', 'jsonified_hkust_gz_seg_train.json')
        self.is_testing_dataset_flag = getattr(data_cfg, 'is_testing_dataset_flag', False)
        self.transform = build(**data_cfg.augmentation)
        self.imdb = json.load(open(self.base_path, 'r'))
        if not self.is_testing_dataset_flag:
            assert 'labels' in self.imdb, "Labels must be provided for training"
            assert len(self.imdb['images']) == len(self.imdb['labels']), "Number of images and annotations must be the same"
    
    def __len__(self):
        return len(self.imdb['images'])
    
    def __getitem__(self, index):
        data = dict()
        data['image'] = read_image(self.imdb['images'][index])
        if not self.is_testing_dataset_flag:
            data['gt_image'] = read_image(self.imdb['labels'][index])
        data['original_shape'] = data['image'].shape
        data['label_sets'] = self.imdb['label_sets']
        data = self.transform(deepcopy(data))
        return data

