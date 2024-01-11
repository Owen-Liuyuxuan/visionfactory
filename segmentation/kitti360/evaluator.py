import os
import shutil
import cv2
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from vision_base.evaluation.base_evaluator import BaseEvaluator
from segmentation.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists, config
from .kitti360_dataset import read_split_file

class KITTI360Evaluator(BaseEvaluator):
    def __init__(self,
                 data_path, #path to kitti raw data
                 split_file,
                 result_path,
                 sample_over=1
                 ):
        self.data_path = data_path
        self.imdb = read_split_file(file_path=split_file, sample_over=sample_over)
        self.image_lists = [os.path.join(self.data_path, obj['image_path']) for obj in self.imdb]
        self.gt_image_lists = [os.path.join(self.data_path, obj['gt_path']) for obj in self.imdb]
        self.result_path = result_path
        self.pred_lists = []

    def reset(self):
        if os.path.isdir(self.result_path):
            shutil.rmtree(self.result_path, ignore_errors=True)
            print("clean up the recorder directory of {}".format(self.result_path))
        os.mkdir(self.result_path)
        self.pred_lists = []

    def step(self, index, output_dict, data):
        pred_path = os.path.join(self.result_path, 'pred_{:010d}.png'.format(index))
        h, w, _ = data['original_shape']
        h_eff, w_eff = data[('image_resize', 'effective_size')]
        seg_result = output_dict['pred_seg'][0, 0:h_eff, 0:w_eff].cpu().numpy().astype(np.uint16)
        seg_result = cv2.resize(seg_result, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(pred_path, seg_result)
        self.pred_lists.append(pred_path)

    def __call__(self, writer:SummaryWriter=None, global_step=0, epoch_num=0):
        result_dict = evaluateImgLists(self.pred_lists, self.gt_image_lists, config)
        if writer is not None:
            import pprint
            formatted_cfg = pprint.pformat(result_dict)
            writer.add_text("evaluation", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n'), global_step=epoch_num)

class JsonEvaluator(KITTI360Evaluator):
    def __init__(self, json_data_path, result_path, sample_over=1):
        self.imdb = json.load(open(json_data_path, 'r'))
        self.gt_image_lists = [obj for obj in self.imdb['labels']]
        self.pred_lists = []
        self.result_path = result_path
