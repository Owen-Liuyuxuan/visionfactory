import os
import shutil
import cv2
import numpy as np
import umsgpack
from torch.utils.tensorboard import SummaryWriter
from vision_base.evaluation.base_evaluator import BaseEvaluator
from segmentation.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists, config
from .dataset import BEVKitti360Dataset

class KITTI360BEVEvaluator(BaseEvaluator):
    def __init__(self,
                 data_path, #path to kitti raw data
                 seam_root_dir,
                 split_name,
                 result_path,
                 sample_over=1,
                 is_temp_test=False,
                 is_evaluate_pseudo_label=True,
                 ):
        self.data_path = data_path

        with open(os.path.join(seam_root_dir, BEVKitti360Dataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")


        self._lst_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)
        with open(os.path.join(self._lst_dir, split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]
        if is_evaluate_pseudo_label:
            self._bev_dir = os.path.join(data_path, 'generated_bev_msk')
        else:
            self._bev_dir = os.path.join(seam_root_dir, 'bev_msk', 'original_id')
        self._front_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._FRONT_MSK_DIR, "front")

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        front_msk_frames = os.listdir(self._front_msk_dir)
        front_msk_frames = [frame.split(".")[0] for frame in front_msk_frames]
        lst = [entry for entry in lst if entry in front_msk_frames]
        lst = set(lst)

        self.images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        self.gt_image_lists = []
        for item_idx, img_desc in enumerate(self.images):
            if is_temp_test:
                if item_idx >= 50:
                    break
            img_desc = self.images[item_idx]
            bev_msk_file = os.path.join(self._bev_dir, "{}.png".format(img_desc['id']))
            if item_idx % sample_over == 0:
                self.gt_image_lists.append(bev_msk_file)

        self.result_path = result_path
        self.is_temp_test=is_temp_test
        self.pred_lists = []

    def reset(self):
        if os.path.isdir(self.result_path):
            shutil.rmtree(self.result_path, ignore_errors=True)
            print("clean up the recorder directory of {}".format(self.result_path))
        os.mkdir(self.result_path)
        self.pred_lists = []

    def step(self, index, output_dict, data):
        image_id = self.images[index]['id']
        pred_path = os.path.join(self.result_path, f'{image_id}.png')
        seg_result = output_dict['pred_seg'][0].cpu().numpy().astype(np.uint16) 
        cv2.imwrite(pred_path, seg_result)
        self.pred_lists.append(pred_path)

    def __call__(self, writer:SummaryWriter=None, global_step=0, epoch_num=0):
        print(len(self.gt_image_lists), len(self.pred_lists))
        result_dict = evaluateImgLists(self.pred_lists, self.gt_image_lists, config)
        if writer is not None:
            import pprint
            formatted_cfg = pprint.pformat(result_dict)
            writer.add_text("evaluation", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n'), global_step=epoch_num)
