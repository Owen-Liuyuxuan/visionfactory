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
                line = line.replace("road02_ins/", "").replace("road03_ins/", "") # not used road0X_ins folder
                splitted = line.strip().split('\t') # they will make split failed
                obj = dict()
                obj['image_path'] = splitted[0]
                obj['gt_path'] = splitted[1].replace('Label', 'remapped_label')
                imdb.append(obj)
    return imdb

def read_directory(base_path, sample_over=1):
    broken_images = ["171206_031956529_Camera_5.jpg",
                      "171206_032007529_Camera_5.jpg",
                      "171206_032028255_Camera_5.jpg",
                      "171206_032031281_Camera_5.jpg",
                      "171206_032038560_Camera_5.jpg"]
    imdb = []
    rgb_dir = os.path.join(base_path, 'ColorImage')
    label_dir = os.path.join(base_path, 'remapped_apollo')
    records = os.listdir(rgb_dir)
    for record in records:
        record_rgb_path = os.path.join(rgb_dir, record)
        record_label_path = os.path.join(label_dir, record)
        if not os.path.isdir(record_rgb_path):
            continue
        cameras = os.listdir(record_rgb_path)
        for camera in cameras:
            camera_rgb_path = os.path.join(record_rgb_path, camera)
            camera_label_path = os.path.join(record_label_path, camera)
            if not os.path.isdir(camera_rgb_path):
                continue
            rgb_image_names = os.listdir(camera_rgb_path)
            rgb_image_names.sort()
            label_image_names = os.listdir(camera_label_path)
            label_image_names.sort()
            for image_name, label_name in zip(rgb_image_names, label_image_names):
                if image_name.replace(base_path + "/", "") in broken_images:
                    continue
                obj = dict()
                obj['image_path'] = os.path.join(camera_rgb_path, image_name).replace(base_path + "/", "")
                obj['gt_path'] = os.path.join(camera_label_path, label_name).replace(base_path + "/", "")
                imdb.append(obj)

    return imdb

class ApolloSceneSegDataset(Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(ApolloSceneSegDataset, self).__init__()
        self.base_path = getattr(data_cfg, 'base_path', '/data/ApolloScene')
        self.sample_over = getattr(data_cfg, 'sample_over', 1)
        self.meta_file   = getattr(data_cfg, 'split_file', None) #'/data/ApolloScene/road03_ins_train.lst')
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
            data['image'] = read_image(os.path.join(self.base_path, obj['image_path']))
            data['gt_image'] = read_image(os.path.join(self.base_path, obj['gt_path']))
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
    cfg.base_path = '/data/ApolloScene'
    cfg.sample_over = 1
    # cfg.split_file = '/data/ApolloScene/road03_ins_train.lst'
    dataset = ApolloSceneSegDataset(**cfg)
    print(len(dataset))
    import tqdm
    for data in tqdm.tqdm(dataset):
        if data['image'] is None:
            continue
        # print(data['image'].shape)
        # print(data['gt_image'].shape)
    
