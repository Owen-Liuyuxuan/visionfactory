import os
import umsgpack
import numpy as np
import json
from PIL import Image
import torch
from vision_base.utils.builder import build

class BEVKitti360Dataset(torch.utils.data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _FRONT_MSK_DIR = "front_msk_trainid"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _METADATA_FILE = "metadata_ortho.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, augmentation, is_temp_test=False):
        super(BEVKitti360Dataset, self).__init__()
        self.seam_root_dir = seam_root_dir
        self.kitti_root_dir = dataset_root_dir
        self.split_name = split_name
        self.transform = build(**augmentation)

        self._init_params()
        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_dir = os.path.join(self.kitti_root_dir, 'generated_bev_msk')
        self._front_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._FRONT_MSK_DIR, "front")
        self._lst_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)

        # Load meta-data and split
        self._meta, self._images, self._img_map = self._load_split(is_temp_test)

    def _init_params(self):
        T_34 = np.array([
            0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
            -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
            -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824
        ]).reshape([3, 4])
        self.T_cam2velo = np.eye(4)
        self.T_cam2velo[0:3] = T_34

        self.P =  np.array([552.554261, 0.000000, 682.049453, 0.000000,
                           0.000000, 552.554261, 238.769549, 0.000000,
                           0.000000, 0.000000, 1.000000, 0.000000]).reshape([3, 4])
        
        self.resolution = 25 / 336

    # Load the train or the validation split
    def _load_split(self, is_temp_test=False):
        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        front_msk_frames = os.listdir(self._front_msk_dir)
        front_msk_frames = [frame.split(".")[0] for frame in front_msk_frames]
        lst = [entry for entry in lst if entry in front_msk_frames]
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        with open(os.path.join(self._img_dir, "{}.json".format('front'))) as fp:
            map_list = json.load(fp)
            map_dict = {k: v for d in map_list for k, v in d.items()}
            img_map['front'] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]
        if is_temp_test:
            images = images[0:50] #[temp test]

        return meta, images, img_map

    def _load_item(self, item_idx):
        img_desc = self._images[item_idx]

        # Get the RGB file names
        img_file = os.path.join(self.kitti_root_dir, self._img_map['front']["{}.png".format(img_desc['id'])])

        data = {}
        # Load the images
        data['image'] = np.array(Image.open(img_file).convert(mode="RGB"))

        # Load the BEV mask
        bev_msk_file = os.path.join(self._bev_dir, "{}.png".format(img_desc['id']))
        data['bev_msk'] = np.array(Image.open(bev_msk_file))

        data['intrinsic'] = self.P.copy()
        data['T_cam2velo'] = self.T_cam2velo.copy()
        data['index'] = item_idx

        return data

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        data = self._load_item(item)
        data = self.transform(data)
        return data
