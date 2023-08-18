import torch
import tqdm
import torch.nn as nn
from typing import Optional, Dict
from torch.utils.tensorboard.writer import SummaryWriter
from mono3d.evaluation.kitti.evaluate import evaluate as evaluate_kitti
from mono3d.evaluation.kitti.result_writer import KittiResultWriter
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.pipeline_hooks.evaluation_hooks.base_evaluation_hooks import BaseEvaluationHook
from vision_base.utils.builder import build

class KittiObjEvaluateHook(BaseEvaluationHook):
    def __init__(self, test_run_hook_cfg,
                 temp_file_dir, label_file_dir, label_split_file, 
                 obj_mapping={'car': 'Car', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist', 'truck': 'Truck'}, result_path_split='test'):
        self.test_hook = build(**test_run_hook_cfg)
        self.temp_file_dir = temp_file_dir
        self.label_file_dir = label_file_dir
        self.label_split_file = label_split_file
        self.current_classes = set([obj_mapping[key] for key in obj_mapping])
        self.result_writer = KittiResultWriter(temp_file_dir, obj_mapping=obj_mapping)
        self.result_path_split = result_path_split

    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        
        for index in tqdm.tqdm(range(len(dataset_val)), dynamic_ncols=True):
            data = dataset_val[index]
            collated_data:Dict = collate_fn([data])

            output_dict = self.test_hook(collated_data, meta_arch, global_step, epoch_num)

            self.result_writer.write(index, output_dict['scores'].cpu().numpy(), 
                                    output_dict['original_bboxes'][:, 0:4].cpu().numpy(), output_dict['cls_names'],
                                    output_dict['original_bboxes'][:, 4:].cpu().numpy())

        if not self.result_path_split == 'test':
            result_texts = evaluate_kitti(self.label_file_dir, self.temp_file_dir, self.label_split_file, current_classes=self.current_classes)
            for class_index, result_text in enumerate(result_texts):
                if writer is not None:
                    writer.add_text("Evaluation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
                print(result_text)