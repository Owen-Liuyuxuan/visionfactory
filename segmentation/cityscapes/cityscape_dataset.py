from __future__ import print_function, division
import os
from easydict import EasyDict
from copy import deepcopy

from torch.utils.data import Dataset # noqa: F401
if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
from vision_base.data.datasets.utils import read_image
from vision_base.utils.builder import build
from segmentation.cityscapes.cityscape_labels import labels as cityscape_labels

def read_split_file(file_path, sample_over=1):
    file_name = os.path.basename(file_path)
    if 'train' in file_name:
        split = 'train'
    elif 'val' in file_name:
        split = 'val'
    elif 'test' in file_name:
        split = 'test'
    else:
        raise ValueError("Unknown split file name {}".format(file_name))
    imdb = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % sample_over == 0:
                line = line.strip()
                obj = dict()
                obj['image_path'] = os.path.join('leftImg8bit', split, f"{line}_leftImg8bit.png")
                obj['gt_path'] = os.path.join('gtFine', split, f"{line}_gtFine_labelIds.png")
                imdb.append(obj)
    return imdb

class CityscapeDataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(CityscapeDataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', '/data/cityscapes')
        self.sample_over = getattr(data_cfg, 'sample_over', 1)
        self.meta_file   = getattr(data_cfg, 'split_file', '/data/cityscapes/train.txt')
        self.imdb = read_split_file(self.meta_file, self.sample_over)
        self.transform = build(**data_cfg.augmentation)
        self.label_text_set = [label.name for label in cityscape_labels[:-1]]
    
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

if __name__ == "__main__":
    cfg = EasyDict()
    cfg.base_path = '/data/cityscapes'
    cfg.sample_over = 1
    cfg.split_file = '/data/cityscapes/train.txt'
    dataset = CityscapeDataset(**cfg)
    print(len(dataset))
    import tqdm
    for data in tqdm.tqdm(dataset, dynamic_ncols=True):
        if data['image'] is None:
            continue
        # print(data['image'].shape)
        # print(data['gt_image'].shape)
    
