import os
import json
import numpy as np
from easydict import EasyDict
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from vision_base.utils.builder import build


class DGPTestDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(DGPTestDataset, self).__init__()
        json_path = getattr(data_cfg, 'json_path', '/data/data.json')
        split     = getattr(data_cfg, 'split', 'train')

        self.scene_files = json.load(open(json_path, 'r'))['scene_splits']['0']['filenames']
        self.base_dir_name = os.path.dirname(json_path)
        self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']

        self.dgp_dataset = SynchronizedSceneDataset(
            json_path, split=split, datum_names=self.camera_names
        )
        self.transform = build(**data_cfg.augmentation)

    def __len__(self):
        return len(self.dgp_dataset)

    def __getitem__(self, index):
        
        dgp_data = self.dgp_dataset[index]

        data = dict()
        data['image'] = np.array(dgp_data['rgb']) #[RGB]
        data['P'] = np.zeros([3, 4])
        data['P'][0:3, 0:3] = dgp_data['intrinsics']
        data['original_P'] = data['P'].copy()
        data = self.transform(data)

        return data
