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
                splitted = line.strip()
                obj = dict()
                obj['image_path'] = splitted[0]
                obj['gt_path'] = splitted[1]
                imdb.append(obj)
    return imdb

def read_directory(base_path, sample_over=1):
    imdb = []
    sequences = os.listdir(base_path)
    for sequence in sequences:
        sequence_path = os.path.join(base_path, sequence)
        if not os.path.isdir(sequence_path):
            continue
        image_dir = os.path.join(sequence_path, 'camera', 'cam_front_center')
        label_dir = os.path.join(sequence_path, 'remapped_label', 'cam_front_center')
        images = os.listdir(image_dir)
        images = [image for image in images if image.endswith('.png')]
        images.sort()
        labels = os.listdir(label_dir)
        labels = [label for label in labels if label.endswith('.png')]
        labels.sort()
        for i, image in enumerate(images):
            if i % sample_over != 0:
                continue
            obj = dict()
            obj['image_path'] = os.path.join(image_dir, image)
            obj['gt_path'] = os.path.join(label_dir, image.replace('camera', 'label'))
            imdb.append(obj)

    return imdb

class A2D2Dataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(A2D2Dataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', '/data/a2d2/camera_lidar_semantic_bboxes')
        self.sample_over = getattr(data_cfg, 'sample_over', 1)
        self.meta_file   = getattr(data_cfg, 'split_file', None)
        if self.meta_file is not None:
            self.imdb = read_split_file(self.meta_file, self.sample_over)
        else:
            self.imdb = read_directory(self.base_path, self.sample_over)
        self.transform = build(**data_cfg.augmentation)
    
    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):
        obj = self.imdb[index]
        data = dict()
        try:
            data['image'] = read_image(obj['image_path'])
            data['gt_image'] = read_image( obj['gt_path'])
            data['original_shape'] = data['image'].shape
        except:
            print("Error in reading {}".format(obj['image_path']))
            data['image'] = None
            data['gt_image'] = None
            data['original_shape'] = None
        data = self.transform(deepcopy(data))
        return data

if __name__ == "__main__":
    cfg = EasyDict()
    cfg.base_path = '/data/a2d2/camera_lidar_semantic_bboxes'
    cfg.sample_over = 1
    # cfg.split_file = '/data/ApolloScene/road03_ins_train.lst'
    dataset = A2D2Dataset(**cfg)
    print(len(dataset))
    import tqdm
    for data in tqdm.tqdm(dataset):
        if data['image'] is None:
            continue
        # print(data['image'].shape)
        # print(data['gt_image'].shape)
    
