import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold

from config import CFG

def get_train_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.ToGray(p=1),
        A.CLAHE(p=1),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.Normalize(),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.ToGray(p=1),
        A.CLAHE(p=1),
        A.Normalize(),
        ToTensorV2(),
    ])

class CustomDataset(Dataset):
    def __init__(
        self, 
        coco, 
        data_dir, 
        image_ids, 
        transforms=None
    ):
        self.coco = coco
        self.data_dir = data_dir
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)

        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)

        if ann_ids:
            anns = self.coco.loadAnns(ann_ids)

            # boxes : (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
            boxes = np.array(ann['bbox'] for ann in anns)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            labels = np.array([ann['category_id'] + 1 for ann in anns]) # Background = 0

    def __len__(self):
        return (len(self.image_ids))

def stratified_group_split(path):
    data_dir = os.path.join(path, 'data/')

    sgkf = StratifiedGroupKFold(
    n_splits=CFG['N_SPLITS'],
    shuffle=True,
    random_state=CFG['SEED']
    )

    train_image_indices = []
    val_image_indices = []

    with open(os.path.join(data_dir, 'coco_train.json'), 'r') as coco_train:
        coco_train = json.load(coco_train)

        annotations = coco_train['annotations']

        image_ids = np.array([ann['image_id'] for ann in annotations])
        category_ids = [ann['category_id'] for ann in annotations]

        for train_id, val_id in sgkf.split(image_ids, category_ids, image_ids):
            train_image_id = np.unique(image_ids[train_id])
            val_image_id = np.unique(image_ids[val_id])

            train_image_indices.append(train_image_id)
            val_image_indices.append(val_image_id)

        return train_image_indices[CFG['FOLD_NUM']], val_image_indices[CFG['FOLD_NUM']]
    
# def load_data(imgs, anns=None, train_mode=False, shuffle=False):
#     if train_mode:
#         dataset = CustomDataset(
#             imgs,
#             anns,
#             get_train_transforms,
#         )
#     else:
#         dataset = CustomDataset(
#             imgs,
#             anns,
#             get_test_transforms,
#         )
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=CFG['BATCH_SIZE'],
#         shuffle=shuffle,
#         num_workers=0,
#     )

#     return dataloader

# def split_load_train_val(path):
#     train_imgs, train_anns, val_imgs, val_anns = stratified_group_split(path)

#     train_loader = load_data(
#         train_imgs,
#         train_anns,
#         train_mode=True,
#         shuffle=True,
#     )

#     val_loader = load_data(
#         val_imgs,
#         val_anns,
#     )

#     return train_loader, val_loader

# def load_test_data(path):
#     test_dir = os.path.join(path, 'data/test/')
#     test_imgs = sorted(glob.glob(os.path.join(test_dir, '*.png')))
#     test_loader = load_data(test_imgs)

#     return test_loader