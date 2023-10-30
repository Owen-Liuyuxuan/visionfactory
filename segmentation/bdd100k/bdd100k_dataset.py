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

def read_split_file(file_path, sample_over=1):
    imdb = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % sample_over == 0:
                splitted = line.strip().split()
                obj = dict()
                obj['image_path'] = splitted[0]
                obj['gt_path'] = splitted[1]
                imdb.append(obj)
    return imdb

def read_directory(base_path, split, sample_over=1):
    imdb = []
    image_base = os.path.join(base_path, 'images', '10k', split)
    images = os.listdir(image_base)
    images = [image for image in images if image.endswith('.jpg')]
    images.sort()

    label_base = os.path.join(base_path, 'labels', 'sem_seg', 'remapped_masks', split)
    labels = os.listdir(label_base)
    labels = [label for label in labels if label.endswith('.png')]
    labels.sort()

    assert len(images) == len(labels)
    for i, image in enumerate(images):
        if i % sample_over != 0:
            continue
        obj = dict()
        obj['image_path'] = os.path.join(image_base, image)
        obj['gt_path'] = os.path.join(label_base, image.replace('jpg', 'png'))
        imdb.append(obj)

    return imdb

class BDD100KDataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(BDD100KDataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', '/data/bdd100k/')
        self.sample_over = getattr(data_cfg, 'sample_over', 1)
        self.split   = getattr(data_cfg, 'split', 'train')
        self.imdb = read_directory(self.base_path, self.split, self.sample_over)
        self.transform = build(**data_cfg.augmentation)
    
    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):
        obj = self.imdb[index]
        data = dict()
        data['image'] = read_image(obj['image_path'])
        data['gt_image'] = read_image( obj['gt_path'])
        data['original_shape'] = data['image'].shape
        data = self.transform(deepcopy(data))
        return data

if __name__ == "__main__":
    cfg = EasyDict()
    cfg.base_path = '/data/bdd100k'
    cfg.sample_over = 1
    # cfg.split_file = '/data/ApolloScene/road03_ins_train.lst'
    dataset = BDD100KDataset(**cfg)
    print(len(dataset))
    import tqdm
    for data in tqdm.tqdm(dataset):
        if data['image'] is None:
            continue
        # print(data['image'].shape)
        # print(data['gt_image'].shape)
    
