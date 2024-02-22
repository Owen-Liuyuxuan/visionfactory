import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
from typing import Sized, Optional, Dict
from cityscapesscripts.evaluation.evalObjectDetection3d import evaluate3dObjectDetection, EvaluationParameters
from tqdm import tqdm
import os
import shutil
import json
from vision_base.utils.builder import build
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.pipeline_hooks.evaluation_hooks.base_evaluation_hooks import BaseEvaluationHook
from vision_base.pipeline_hooks.train_val_hooks.base_validation_hooks import BaseValidationHook

from mono3d.evaluation.cityscapes.data_frame import CityScapeFrame

class CityScapesObjEvaluationHook(BaseEvaluationHook):
    """
        Cityscapes Object Detection Evaluation Hook
    """
    def __init__(self,
                test_run_hook_cfg:EasyDict,
                dataset_json_path:str,
                result_path:str,
                result_path_split:str='validation', # determine in train/test code, should not in config
                **kwargs):
        self.test_hook:BaseValidationHook = build(**test_run_hook_cfg)
        self.result_path_split = result_path_split
        self.result_path = result_path
        self.dataset_eval_json:Dict = json.load(open(dataset_json_path, 'r'))
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if os.path.isdir(result_path):
            shutil.rmtree(result_path, ignore_errors=True)
            print("clean up the recorder directory of {}".format(result_path))
        os.mkdir(result_path)
        print("rebuild {}".format(result_path))

    def infer_cs_gt_file_from_image_path(self, image_path):
        base_dir = self.dataset_eval_json.get("base_directory", "")
        full_image_path = os.path.join(base_dir, image_path)

        ## full path: base_dir/<modal>/<split>/<city>/<file_name>
        ## file_name: <city>_number1_number2_<modal>.postfix 
        image_file_name:str = os.path.basename(full_image_path)
        image_dir = os.path.dirname(full_image_path)
        city_name = os.path.basename(image_dir)
        split_dir = os.path.dirname(image_dir)
        split_name = os.path.basename(split_dir)
        modal_dir = os.path.dirname(split_dir)
        modal_name = os.path.basename(modal_dir)
        self.cs_base_dir = os.path.dirname(modal_dir)

        gt_modal_name = "gtBbox3d"
        gt_bbox_file = os.path.join(
            self.cs_base_dir, gt_modal_name, split_name, 
            city_name, image_file_name.replace(".png", ".json").replace(modal_name, gt_modal_name)
        )
        return gt_bbox_file

    
    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        
        for index in tqdm(range(len(dataset_val)), dynamic_ncols=True):
            image_file = self.dataset_eval_json['images'][index]
            gt_bbox_file = self.infer_cs_gt_file_from_image_path(image_file)
            original_cs_frame = CityScapeFrame(json.load(open(gt_bbox_file, 'r')))

            data = dataset_val[index]
            collated_data:Dict = collate_fn([data])

            output_dict = self.test_hook(collated_data, meta_arch, global_step, epoch_num)

            original_bboxes = output_dict['original_bboxes']
            N = original_bboxes.shape[0]
            pitch_roll_alphas = torch.zeros_like(original_bboxes[:, 0:3])
            pitch_roll_alphas[:, 2] = original_bboxes[:, 10]

            original_cs_frame.set_objects(original_bboxes[:, 0:4].cpu().numpy(),
                                original_bboxes[:, 7:10].cpu().numpy(),
                                original_bboxes[:, 4:7].cpu().numpy(),
                                pitch_roll_alphas.cpu().numpy(),
                                output_dict['scores'].cpu().numpy(),
                                output_dict['cls_names'])
            
            name = os.path.basename(image_file).split('.')[0]
            
            file_path = os.path.join(self.result_path, name + '.json')
            original_cs_frame.to_json(file_path)

        if not self.result_path_split == 'test':
            eval_params = EvaluationParameters(
                labels_to_evaluate=["car", "truck", "bus", "train", "motorcycle", "bicycle"], #default ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
                min_iou_to_match=0.7, #default 0.7
                max_depth=100, #default 100
                step_size=5, #default 5
                matching_method=int(False), #default False
                cw=-1 #default -1
            )
            if self.result_path_split.startswith('val'):
                gt_folder = os.path.join(self.cs_base_dir, 'gtBbox3d', 'val')
            else:
                gt_folder = os.path.join(self.cs_base_dir, 'gtBbox3d', 'train')

            import coloredlogs
            import logging
            coloredlogs.install(logging.INFO)
            evaluate3dObjectDetection(
                gt_folder=gt_folder,
                pred_folder=self.result_path,
                result_folder=self.result_path,
                eval_params=eval_params,
                plot=False
            )
            coloredlogs.install(logging.CRITICAL)
