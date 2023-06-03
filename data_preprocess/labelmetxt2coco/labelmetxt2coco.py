import os
import glob
import argparse
import json
import cv2
import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='convert labelme txt format to COCO json format.'
)
parser.add_argument(
    '-d',
    '--data_path',
    dest='data_path',
    default='../../data/',
    help='data directory',
)

args = parser.parse_args()

TRAIN_PATH = os.path.join(args.data_path, 'train/')
TEST_PATH = os.path.join(args.data_path, 'test/')
DEST_PATH = os.path.join(args.data_path, 'coco/')

os.makedirs(DEST_PATH, exist_ok=True)

# categories
categories_list = []
cls_id = 0

with open(os.path.join(args.data_path, 'classes.txt'), 'r') as cls_file:
    lines = cls_file.readlines()

    for line in lines:
        line = line.strip()

        # category_name
        category_name = line.split(',')[-1]

        # categories
        categories_list.append({
            'id': cls_id,
            'name' : category_name,
        })

        cls_id += 1

### Convert train data

# 이미지, annotaion 경로
train_imgs = sorted(glob.glob(os.path.join(TRAIN_PATH, '*.png')))
anns = sorted(glob.glob(os.path.join(TRAIN_PATH, '*.txt')))

ISCROWD = 0

train_images_list = []
annotations_list = []
ann_id = 0

print('converting train data to COCO format...')
for img_id, (img, ann) in enumerate(zip(tqdm(train_imgs), anns)):
    # date_captured
    date_captured = os.path.getmtime(img)
    date_captured = datetime.datetime.fromtimestamp(date_captured).strftime('%Y-%m-%d %H:%M:%S')

    # file_name
    file_name = '/'.join(img.replace('\\', '/').split('/')[-2:])

    # width & height
    img = cv2.imread(img)
    img_h, img_w, _ = img.shape

    # images
    train_images_list.append({
        'id' : img_id,
        'width' : img_w,
        'height' : img_h,
        'file_name' : file_name,
        'date_captured' : date_captured
    })

    with open(ann, 'r') as ann_file:
        lines = ann_file.readlines()

        for line in lines:
            line = line.strip()

            # category_id
            category_id = int(float(line.split(' ')[0]))

            # bbox
            bbox = list(map(lambda x : float(x), line.split(' ')[1:]))

            xs = [bbox[i] for i in range(0, 8, 2)]
            ys = [bbox[i] for i in range(1, 8, 2)]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            width = x_max - x_min 
            height = y_max - y_min

            bbox = [x_min, y_min, width, height]

            # area
            area = width * height

            # annotations
            annotations_list.append({
                'id' : ann_id,
                'image_id' : img_id,
                'category_id' : category_id,
                'area' : area,
                'bbox' : bbox,
                'iscrowd' : ISCROWD,
            })

            ann_id += 1
    
coco_train = {
    'categories' : categories_list,
    'images' : train_images_list,
    'annotations' : annotations_list,
}

with open(os.path.join(DEST_PATH, 'train.json'), 'w') as file:
    json.dump(coco_train, file, indent=2)

### Convert test data

# test 이미지 경로
test_imgs = sorted(glob.glob(os.path.join(TEST_PATH, '*.png')))

test_images_list = []

print('converting test data to COCO format...')
for img_id, img in enumerate(tqdm(test_imgs)):
    # date_captured
    date_captured = os.path.getmtime(img)
    date_captured = datetime.datetime.fromtimestamp(date_captured).strftime('%Y-%m-%d %H:%M:%S')

    # file_name
    file_name = '/'.join(img.replace('\\', '/').split('/')[-2:])

    # width & height
    img = cv2.imread(img)
    img_h, img_w, _ = img.shape

    # images
    test_images_list.append({
        'id' : img_id,
        'width' : img_w,
        'height' : img_h,
        'file_name' : file_name,
        'date_captured' : date_captured
    })

coco_test = {
    'categories' : categories_list,
    'images' : test_images_list,
    'annotations' : [],
}

with open(os.path.join(DEST_PATH, 'test.json'), 'w') as file:
    json.dump(coco_test, file, indent=2)