import json
import sys
import os
sys.path.append(os.getcwd())
from segmentation.data_utils.cityscape_labels import labels as cityscape_labels

def read_split_file(file_path, sample_over=1):
    file_name = os.path.basename(file_path)
    if 'train' in file_name:
        split = 'train'
    elif 'val' in file_name:
        split = 'val'
    elif 'test' in file_name:
        split = 'test'
    else:
        raise ValueError("Unknown split file name {}".format(file_name))
    imdb = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % sample_over == 0:
                line = line.strip()
                obj = dict()
                obj['image_path'] = os.path.join('leftImg8bit', split, f"{line}_leftImg8bit.png")
                obj['gt_path'] = os.path.join('gtFine', split, f"{line}_gtFine_labelIds.png")
                imdb.append(obj)
    return imdb

def read_through_cityscape(base_path="/data/cityscapes", split='train'):
    json_path = f"seg_dataset_meta/jsonified_cityscapes_seg_{split}.json"
    json_obj = dict()
    meta_file = os.path.join(base_path, f"{split}.txt")
    imdb = read_split_file(meta_file)
    images = []
    labels = []
    remapped_labels = []
    label_text_set = [label.name for label in cityscape_labels[:-1]]
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
    Fire(read_through_cityscape)