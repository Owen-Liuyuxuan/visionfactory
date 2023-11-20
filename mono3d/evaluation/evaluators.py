import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch.nn as nn
import json
import numpy as np
from typing import Optional, Dict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.tensorboard.writer import SummaryWriter
from mono3d.evaluation.kitti.evaluate import evaluate as evaluate_kitti
from mono3d.evaluation.kitti.eval_general import general_eval_3d, get_mAP_v2, print_str, CLASS_NAMES
from mono3d.evaluation.kitti.result_writer import KittiResultWriter
from mono3d.data.utils import Json2annotation, JsonToCoCo
from mono3d.data.dataset import select_by_split_file
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
                    writer.add_text("validation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
                print(result_text)

class Object2DEvaluateHook(BaseEvaluationHook):
    def __init__(self, test_run_hook_cfg, gt_json_file, result_path_split='test'):
        self.test_hook = build(**test_run_hook_cfg)
        self.gt_json_file = gt_json_file
        self._preprocess()
        self.result_path_split = result_path_split
            

    def _preprocess(self):
        print("Loading Annotations into COCO format")
        with open(self.gt_json_file, 'r') as f:
            self.gt_file_obj = json.load(f)
        self.labeled_classes = self.gt_file_obj['labeled_objects']
        self.coco = COCO()
        coco_dataset = JsonToCoCo(self.gt_file_obj, False, 0, 0)
        self.coco.dataset = coco_dataset
        self.coco.createIndex()
        print("Done")

    def _formatting_cocolog(self, coco_stats):
        """
            coco_stats:np.NdArray : 12
        """
        maxDets=[100, 100, 100, 100, 100, 100,
                    1, 10, 100, 100, 100, 100]
        areas=['all', 'all', 'all', 'small', 'medium',' large',
               'all', 'all', 'all', 'small', 'medium', 'large']
        titleStrs=['Average Precision' for _ in range(6)] + \
                    ['Average Recall' for _ in range(6)]
        typeStrs = ['(AP)' for _ in range(6)] + \
                    ['(AR)' for _ in range(6)]
        iouStrs = [
            f"IoU={iou}" for iou in ([
                "0.50:0.95", "0.50", "0.75"
            ]  + ["0.50:0.95" for _ in range(9)] )
        ]
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        coco_string_result = ""
        for i in range(12):
            coco_string_result += iStr.format(titleStrs[i], typeStrs[i], iouStrs[i], areas[i], maxDets[i], coco_stats[i])
            coco_string_result += '\n'
        return coco_string_result

    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        
        predictions = []

        for index in tqdm.tqdm(range(len(dataset_val)), dynamic_ncols=True):
            data = dataset_val[index]
            collated_data:Dict = collate_fn([data])

            output_dict = self.test_hook(collated_data, meta_arch, global_step, epoch_num)

            scores = output_dict['scores'].cpu().numpy() # Array [N]
            bbox2d = output_dict['original_bboxes'][:, 0:4].cpu().numpy() # Array [N, 4] [x1, y1, x2, y2]
            cls_names = output_dict['cls_names'] #List[N]

            N = len(scores)
            for i in range(N):
                if cls_names[i] in self.labeled_classes: 
                    x1 = bbox2d[i][0]
                    y1 = bbox2d[i][1]
                    w = bbox2d[i][2] - x1
                    h = bbox2d[i][3] - y1
                    score = scores[i]
                    class_id = self.labeled_classes.index(cls_names[i])
                    predictions.append(np.array([index, x1, y1, w, h, score, class_id]))

        if not self.result_path_split == 'test':
            coco_dt = self.coco.loadRes(np.stack(predictions, axis=0)) # [N, 7]
            # Evaluate Each Class
            for i in range(len(self.labeled_classes)):
                print(f"Evaluating {self.labeled_classes[i]}")
                coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
                coco_eval.params.catIds = [i] #person id : 1
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                result_text = self._formatting_cocolog(coco_eval.stats)
                if writer is not None:
                    writer.add_text("validation result {}".format(self.labeled_classes[i]), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
            # Evaluate Mean:
            coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            result_text = self._formatting_cocolog(coco_eval.stats)
            if writer is not None:
                writer.add_text("validation result {}".format('mean'), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
           
            return coco_eval.stats

class Object3DKittiMetricEvaluateHook(BaseEvaluationHook):
    """
        This evaluation hook aims to evaluate detection results directly on the standard json format on KITTI 3D Metrics.
        The basic idea is to read JSON file and convert it the memory format for kitti evaluator. 
            Also we need to define iou threshold for each class.
    """
    def __init__(self, test_run_hook_cfg,
                gt_json_file,
                iou_2d_threshold={'car':0.7, 'pedestrian':0.5, 'truck':0.7, 'bicycle':0.5, 'bus':0.7, 'motorcycle':0.5, 'trailer':0.7},
                iou_bev_threshold={'car':0.7, 'pedestrian':0.5, 'truck':0.7, 'bicycle':0.5, 'bus':0.7, 'motorcycle':0.5, 'trailer':0.7},
                iou_3d_threshold={'car':0.7, 'pedestrian':0.5, 'truck':0.7, 'bicycle':0.5, 'bus':0.7, 'motorcycle':0.5, 'trailer':0.7},
                split_file=None,
                result_path_split='test'):
        self.test_hook = build(**test_run_hook_cfg)
        self.gt_json_file = gt_json_file
        self.result_path_split = result_path_split
        with open(gt_json_file, 'r') as f:
            gt_file_obj = json.load(f)
        if split_file is not None:
            gt_file_obj = select_by_split_file(gt_file_obj, split_file)
        self.kitti_eval_annotations = Json2annotation(gt_file_obj)

        cls_with_threshold = set(iou_2d_threshold.keys()).intersection(
                    set(iou_bev_threshold.keys()), set(iou_3d_threshold.keys()))
        
        self.classes_to_test = [cls_ for cls_ in gt_file_obj['labeled_objects'] if cls_ in cls_with_threshold]
        self.min_overlaps = np.array(
            [
                [iou_2d_threshold[cls_], iou_bev_threshold[cls_], iou_3d_threshold[cls_]] 
                    for cls_ in self.classes_to_test
            ]
        ).T[None] #[1, 3, classes]

    def _evaluate(self, prediction_anno):
        current_classes = self.classes_to_test
        cls_index = [CLASS_NAMES.index(cls_) for cls_ in current_classes]
        metrics = general_eval_3d(
            self.kitti_eval_annotations,
            prediction_anno,
            current_classes=cls_index,
            min_overlaps=self.min_overlaps,
            compute_aos=True,
            difficultys=[0, 1, 2],
            z_axis=1,
            z_center=1.0,
        )
        result = []
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            cls_result = ''
            for i in range(self.min_overlaps.shape[0]):
                mAPbbox = get_mAP_v2(metrics["bbox"]["precision"][j, :, i])
                mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox)
                mAPbev = get_mAP_v2(metrics["bev"]["precision"][j, :, i])
                mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
                mAP3d = get_mAP_v2(metrics["3d"]["precision"][j, :, i])
                mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
                cls_result += print_str(
                    (f"{curcls} "
                    "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*self.min_overlaps[i, :, j])))
                cls_result += print_str(f"bbox AP:{mAPbbox}")
                cls_result += print_str(f"bev  AP:{mAPbev}")
                cls_result += print_str(f"3d   AP:{mAP3d}")
                mAPaos = get_mAP_v2(metrics["bbox"]["orientation"][j, :, i])
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                cls_result += print_str(f"aos  AP:{mAPaos}")
            result.append(cls_result)
        return result

    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        
        predictions = dict()
        predictions['annotations'] = []

        for index in tqdm.tqdm(range(len(dataset_val)), dynamic_ncols=True):
            data = dataset_val[index]
            collated_data:Dict = collate_fn([data])

            output_dict = self.test_hook(collated_data, meta_arch, global_step, epoch_num)
            original_bboxes = output_dict['original_bboxes'].cpu().numpy().astype(np.float64) # [N, 11] #xyxy, x3d, y3d, z, w, h, l, alpha, theta

            scores = output_dict['scores'].cpu().numpy().astype(np.float64) # Array [N]
            bbox2d = original_bboxes[:, 0:4] # Array [N, 4] [x1, y1, x2, y2]
            xyz = original_bboxes[:, 4:7] # Array [N, 3] [x, y, z]
            whl = original_bboxes[:, 7:10] # Array [N, 3] [w, h, l]
            alpha = original_bboxes[:, 10] # Array [N] [ry]
            theta = original_bboxes[:, 11] # Array [N] [ry]
            cls_names = output_dict['cls_names'] #List[N]

            frame_annotations = []
            for i in range(len(scores)):
                if cls_names[i] not in self.classes_to_test:
                    continue
                frame_annotations.append(dict(
                    whl = whl[i], xyz = xyz[i], alpha = alpha[i], theta = theta[i],
                    image_id = index, bbox2d = bbox2d[i], visibility_level = 0,
                    category_name = cls_names[i], score = scores[i]
                ))
            predictions['annotations'].append(frame_annotations)
        
        if self.result_path_split == 'test':
            predictions['images'] = []
            with open('test.json', 'w') as f:
                for i in range(len(predictions['annotations'])):
                    for j in range(len(predictions['annotations'][i])):
                        predictions['annotations'][i][j]['whl'] = list(predictions['annotations'][i][j]['whl'])
                        predictions['annotations'][i][j]['xyz'] = list(predictions['annotations'][i][j]['xyz'])
                        predictions['annotations'][i][j]['bbox2d'] = list(predictions['annotations'][i][j]['bbox2d'])
                        predictions['annotations'][i][j]['bbox2d'] = list(predictions['annotations'][i][j]['bbox2d'])
                    predictions['images'].append(f"{i:06d}.png")
                json.dump(predictions, f)
            return
        else:
            kitti_pred_anno = Json2annotation(predictions)
            result_texts = self._evaluate(kitti_pred_anno)            
            for class_index, result_text in enumerate(result_texts):
                if writer is not None:
                    writer.add_text("validation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
                print(result_text)
