import os
import argparse
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser(
    description='split coco data using stratified group kfold.'
)
parser.add_argument(
    '-d',
    '--data_path',
    dest='data_path',
    default='../../data/',
    help='data directory',
)
parser.add_argument(
    '-n',
    '--n_splits',
    dest='n_splits',
    default=4,
    type=int,
    help='number of splits for cross validation',
)
parser.add_argument(
    '-r',
    '--random_state',
    dest='random_state',
    default=41,
    type=int,
    help='random state',
)
args = parser.parse_args()

COCO_PATH = os.path.join(args.data_path, 'coco/')

TRAIN_PREFIX = 'train_'
VAL_PREFIX = 'val_'

sgkf = StratifiedGroupKFold(
    n_splits=args.n_splits,
    shuffle=True,
    random_state=args.random_state
)

with open(os.path.join(COCO_PATH, 'train.json'), 'r') as coco:
    coco = json.load(coco)

    images = np.array(coco['images'])
    annotations = np.array(coco['annotations'])
    categories = coco['categories']

    image_ids = np.array([ann['image_id'] for ann in annotations])
    category_ids = [ann['category_id'] for ann in annotations]

    for i, (train_ann_id, val_ann_id) in enumerate(sgkf.split(image_ids, category_ids, image_ids)):
        train_image_id = np.unique(image_ids[train_ann_id])
        val_image_id = np.unique(image_ids[val_ann_id])

        train_images = images[train_image_id]
        train_anns = annotations[train_ann_id]

        coco_train = {
            'categories' : categories,
            'images' : list(train_images),
            'annotations' : list(train_anns),
        }

        with open(os.path.join(COCO_PATH, f'{TRAIN_PREFIX}{i}.json'), 'w') as coco_train_json:
            json.dump(
                coco_train, 
                coco_train_json,
                indent=2,
            )

        val_images = images[val_image_id]
        val_anns = annotations[val_ann_id]

        coco_val = {
            'categories' : categories,
            'images' : list(val_images),
            'annotations' : list(val_anns),
        }

        with open(os.path.join(COCO_PATH, f'{VAL_PREFIX}{i}.json'), 'w') as coco_val_json:
            json.dump(
                coco_val, 
                coco_val_json,
                indent=2
            )
print('split done.')