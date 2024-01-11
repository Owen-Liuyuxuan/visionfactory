"""
The core idea is to produce a unified json data description file for objects365 dataset.

1. Unify classes annotations. We know there are categories in nuScenes not labeled in KITTI/objects365 and there are many other classes in objects365. We need to know that whether we are labeling each category in each image. If a category is not labeled in this image, we should not supress the prediction/evaluation of this category during training.
2. We need to unify the coordination, rotation.
3. We need to include camera calibration information.
4. We allow images with 2D labels.
5. Suggested data augmentation methods in training: RandomWarpAffine.
6. In objects365, we only use the 2D label.
7. Objects365 takes the COCO dataset format and comes in almost random order (not sorted by image ID). We need to correctly implement the image ID to index mapping.

Suggested unified 3D Types:

['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

Other Types are shown in below

"""
import os
import json
import yaml
import numpy as np
import tqdm

LABELED_3D_OBJECTS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
LABEL_MAPPING = {'Person': 'pedestrian', 'SUV': 'car', 'Bicyle': 'bicycle', 'Van': 'truck', 'Motorcycle': 'motorcycle', 
                 'Truck': 'truck', 'Traffic cone': 'traffic_cone', 'Sports Car': 'car', 'Tricycle': 'motorcycle',
                 'Fire Truck': 'truck', 'Ambulance': 'truck', 'Heavy Truck': 'truck', 'Pickup Truck': 'truck',
                   'Bus': 'bus', 'Machinery Vehicle': 'construction_vehicle'}

obtained_labeled_objects = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',   # These are 3D labeled
                            'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
                            'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes',
                            'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'Wine Glass',
                            'Belt', 'Monitor/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 
                            'Stool', 'Barrel/bucket', 'Couch', 'Sandals', 'Basket', 'Drum', 'Pen/Pencil', 'Wild Bird', 'High Heels', 'Guitar', 'Carpet', 'Cell Phone', 
                            'Bread', 'Camera', 'Canned', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 
                            'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Fork', 'Traffic Sign', 'Balloon', 'Tripod', 
                            'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine', 
                            'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 
                            'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 
                            'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 
                            'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 
                            'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 
                            'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Billiards', 'Converter', 
                            'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette', 'Paint Brush', 'Pear', 'Hamburger', 'Extractor', 'Extension Cord', 'Tong', 
                            'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 
                            'Projector', 'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hot-air balloon', 'Cello', 
                            'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 
                            'Trumpet', 'Pineapple', 'Golf Ball', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 
                            'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 
                            'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 
                            'Mushroom', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measure/Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 
                            'Steak', 'Crosswalk Sign', 'Stapler', 'Camel', 'Formula 1', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 
                            'Papaya', 'Antelope', 'Parrot', 'Seal', 'Butterfly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 
                            'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 
                            'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle', 'Cosmetics Brush/Eyeliner Pencil', 
                            'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis']

def main(objects365_json_path, split='train', json_path="objects365_train.json"):
    objects365_yaml = yaml.load(
        open(os.path.join("./mono3d", 'data', 'objects365_base_class.yaml'), 'r'), Loader=yaml.FullLoader
        )
    objects365_names = objects365_yaml['names']
    labeled_objects = []
    index_mapping = {}
    for name in LABELED_3D_OBJECTS:
        labeled_objects.append(name)
    for i in objects365_names:
        base_name = objects365_names[i]
        if base_name in LABEL_MAPPING:
            index_mapping[i] = labeled_objects.index(LABEL_MAPPING[base_name])
        else:
            labeled_objects.append(base_name)
            index_mapping[i] = len(labeled_objects) - 1

    
    print("Start Loading Objects365 Json, this may take a while.")
    objects365_coco_json = json.load(open(objects365_json_path, 'r')) # this will be in COCO format
    print("Finished Loading Objects365 Json.")

    # Define the mapping of COCO categories to Unified types
    category_mapping = {category['id']: labeled_objects[index_mapping[category['id'] - 1]]  # the category id starts from 1, so we need to minus 1
                        for category in objects365_coco_json['categories']} 
    # Create a dictionary mapping from image ID to its index
    image_id_to_index = {image['id']: index for index, image in enumerate(objects365_coco_json['images'])}

    unified_json = {
        "labeled_objects": labeled_objects,  # Unique list of object names
        "images": [img['file_name'] for img in objects365_coco_json['images']],  # List of image paths
        "is_labeled_3d": False,  # Assuming 2D labels for COCO
        "total_frames": len(objects365_coco_json['images']),
        "calibrations": [{'P' : [1 for __ in range(12)]} for _ in objects365_coco_json['images']],  # Placeholder, as COCO doesn't provide calibration matrices
        "annotations": [[] for _ in objects365_coco_json['images']],  # Initialize empty lists for each image
        "base_directory": os.path.join(objects365_yaml['path'], objects365_yaml[split])
    }

    for annotation in tqdm.tqdm(objects365_coco_json['annotations']):
        image_index = image_id_to_index.get(annotation['image_id'])
        if image_index is not None:
            unified_annotation = {
                "bbox2d": [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]],
                "visibility_level": 0 if annotation['iscrowd'] == 0 else 3,  # Simplified mapping
                "category_name": category_mapping[annotation['category_id']],
                "image_id": image_index
            }
            unified_json['annotations'][image_index].append(unified_annotation)

    json.dump(unified_json, open(json_path, 'w'))


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
