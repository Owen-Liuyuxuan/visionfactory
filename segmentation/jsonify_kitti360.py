import json
import sys
import os
sys.path.append(os.getcwd())
from segmentation.evaluation.labels import labels as kitti360_labels

def read_split_file(file_path, sample_over=1):
    imdb = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % sample_over == 0:
                splitted = line.strip().split(' ')
                obj = dict()
                obj['image_path'] = (splitted[0])
                obj['gt_path'] = (splitted[1])
                imdb.append(obj)
    return imdb

def read_through_kitti360(base_path="/data/KITTI-360", split='train'):
    json_path = f"seg_dataset_meta/jsonified_kitti360_seg_{split}.json"
    json_obj = dict()
    meta_file = os.path.join(base_path, f"data_2d_semantics/train/2013_05_28_drive_{split}_frames.txt")
    images = []
    labels = []
    remapped_labels = []
    label_text_set = [label.name for label in kitti360_labels[:-1]]
    json_obj['label_sets'] = label_text_set
    imdb = read_split_file(meta_file)
    for obj in imdb:
        images.append(os.path.join(base_path, obj['image_path']))
        labels.append(os.path.join(base_path, obj['gt_path']))
        remapped_labels.append(os.path.join(base_path, obj['gt_path']))
    
    json_obj['images'] = images
    json_obj['labels'] = labels
    json_obj['remapped_labels'] = remapped_labels
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=4)

if __name__ == "__main__":
    from fire import Fire
    Fire(read_through_kitti360)