import os
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold

def save_coco(
    categories,
    images,
    annotations,
    image_ids,
    ann_ids,
    file_path
):
    image_ids = np.unique(image_ids[ann_ids])
    images = images[image_ids]
    annotations = annotations[ann_ids]

    coco = {
        'categories' : categories,
        'images' : list(images),
        'annotations' : list(annotations),
    }

    with open(file_path, 'w') as coco_file:
        json.dump(
            coco,
            coco_file,
            indent=2,
        )


def split_coco_sgkf(
    data_root,
    n_splits,
    seed
):
    TRAIN_PREFIX = 'train_'
    VAL_PREFIX = 'val_'

    coco_path = os.path.join(data_root, 'coco/')

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    with open(os.path.join(coco_path, 'train.json'), 'r') as coco:
        coco = json.load(coco)

        images = np.array(coco['images'])
        annotations = np.array(coco['annotations'])
        categories = coco['categories']

        image_ids = np.array([ann['image_id'] for ann in annotations])
        category_ids = [ann['category_id'] for ann in annotations]

        for i, (train_ann_ids, val_ann_ids) in enumerate(sgkf.split(image_ids, category_ids, image_ids)):
            train_file_path = os.path.join(coco_path, f'{TRAIN_PREFIX}{i}.json')
            
            save_coco(
                categories,
                images,
                annotations,
                image_ids,
                train_ann_ids,
                train_file_path
            )

            val_file_path = os.path.join(coco_path, f'{VAL_PREFIX}{i}.json')
            
            save_coco(
                categories,
                images,
                annotations,
                image_ids,
                val_ann_ids,
                val_file_path
            )

    print('split done.')

if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('..')

    from utils import rename_folder
    from labelmetxt_converter.labelmetxt2coco import convert2coco

    parser = argparse.ArgumentParser(
        description='split coco data using stratified group kfold.'
    )
    parser.add_argument(
        '-d',
        '--data_root',
        dest='data_root',
        default='../../data/',
        help='data root directory',
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
        '-s',
        '--seed',
        dest='seed',
        default=41,
        type=int,
        help='seed',
    )
    args = parser.parse_args()

    data_root = args.data_root

    # data 폴더 내에 images 폴더가 존재하지 않는다면, train 폴더명을 images로 변경
    if 'images' not in os.listdir(data_root):
        rename_folder.train2images(data_root)
    
    # coco format의 train, test 데이터가 존재하지 않는다면 변환하기
    if 'coco' not in os.listdir(data_root):
        convert2coco(data_root)

    split_coco_sgkf(
        data_root,
        args.n_splits,
        args.seed
    )