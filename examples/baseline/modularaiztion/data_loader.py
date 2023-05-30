import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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

def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets

class CustomDataset(Dataset):
    def __init__(
        self, 
        coco, 
        data_dir, 
        transforms=None
    ):
        self.coco = coco
        self.data_dir = data_dir
        self.transforms = transforms

    def __getitem__(self, index):
        image_ids = self.coco.getImgIds()
        image_id = image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)

        if ann_ids:
            anns = self.coco.loadAnns(ann_ids)

            # boxes : (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
            boxes = np.array([ann['bbox'] for ann in anns])
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            labels = np.array([ann['category_id'] + 1 for ann in anns]) # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]

            return image, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        else:
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed["image"]

            return image

    def __len__(self):
        return len(self.coco.getImgIds())