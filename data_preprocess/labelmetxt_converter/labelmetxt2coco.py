import os
import glob
import json
import datetime
import cv2
from tqdm.auto import tqdm

def categories2coco(data_root):
    categories = []
    cat_id = 0

    with open(os.path.join(data_root, 'classes.txt')) as cls_file:
        lines = cls_file.readlines()

        for line in lines:
            line = line.strip()

            # category name
            cat_name = line.split(',')[-1]

            # coco categories format
            categories.append({
                'id' : cat_id,
                'name' : cat_name,
            })

            cat_id += 1

    return categories

def get_coco_images(img_id, img_path):
    # date captured
    date_captured = os.path.getmtime(img_path)
    date_captured = datetime.datetime.fromtimestamp(date_captured).strftime('%Y-%m-%d %H:%M:%S')

    # file name
    file_name = '/'.join(img_path.replace('\\', '/').split('/')[-2:])

    # images
    # width & height
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    coco_image = {
        'id' : img_id,
        'width' : img_w,
        'height' : img_h,
        'file_name' : file_name,
        'date_captured' : date_captured
    }

    return coco_image

def get_coco_bbox(labelme_bbox):
    xs = [labelme_bbox[i] for i in range(0, 8, 2)]
    ys = [labelme_bbox[i] for i in range(1, 8, 2)]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    w = x_max - x_min 
    h = y_max - y_min

    coco_bbox = [x_min, y_min, w, h]

    return coco_bbox

def train2coco(train_path):
    ISCROWD = 0

    images_path = sorted(glob.glob(os.path.join(train_path, '*.png')))
    annotations_path = sorted(glob.glob(os.path.join(train_path, '*.txt')))

    coco_images = []
    coco_annotations = []
    ann_id = 0

    print('converting train data to COCO format...')
    for img_id, (img_path, ann_path) in enumerate(zip(tqdm(images_path), annotations_path)):
        # coco images format
        coco_image = get_coco_images(img_id, img_path)
        coco_images.append(coco_image)

        # annotations
        with open(ann_path, 'r') as ann_file:
            lines = ann_file.readlines()

            for line in lines:
                line = line.strip()

                # category_id
                category_id = int(float(line.split(' ')[0]))

                # bbox
                labelme_bbox = list(map(lambda x : float(x), line.split(' ')[1:]))
                coco_bbox = get_coco_bbox(labelme_bbox)

                # area
                bbox_w = coco_bbox[2]
                bbox_h = coco_bbox[3]
                area = bbox_w * bbox_h

                # coco annotations format
                coco_annotations.append({
                    'id' : ann_id,
                    'image_id' : img_id,
                    'category_id' : category_id,
                    'area' : area,
                    'bbox' : coco_bbox,
                    'iscrowd' : ISCROWD,
                })

                ann_id += 1

    return coco_images, coco_annotations

def test2coco(test_path):
    images_path = sorted(glob.glob(os.path.join(test_path, '*.png')))

    coco_images = []

    print('converting test data to COCO format...')
    for img_id, img_path in enumerate(tqdm(images_path)):
        # coco images format
        coco_image = get_coco_images(img_id, img_path)
        coco_images.append(coco_image)

    return coco_images

def save_coco(
    coco_path,
    categories,
    images,
    annotations=[]
):
    coco = {
        'categories' : categories,
        'images' : images,
        'annotations' : annotations
    }

    if annotations:
        file_name = 'train.json'
    else:
        file_name = 'test.json'

    print(f"saving '{file_name}'...")
    with open(os.path.join(coco_path, file_name), 'w') as coco_file_name:
        json.dump(coco, coco_file_name, indent=2)

def convert2coco(data_root):
    coco_path = os.path.join(data_root, 'coco/')
    train_path = os.path.join(data_root, 'images/')
    test_path = os.path.join(data_root, 'test/')

    # coco 데이터가 저장될 directory
    os.makedirs(coco_path, exist_ok=True)

    # coco categories format
    categories = categories2coco(data_root)

    # convert train data to coco format & save
    train_images, train_annotations = train2coco(train_path)
    save_coco(
        coco_path,
        categories,
        train_images,
        train_annotations
    )
    # convert test data to coco format & save
    test_images = test2coco(test_path)
    save_coco(
        coco_path,
        categories,
        test_images
    )

if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('..')

    from utils import rename_folder

    parser = argparse.ArgumentParser(
        description='convert labelme txt format to COCO json format.'
    )
    parser.add_argument(
        '-d',
        '--data_root',
        dest='data_root',
        default='../../data/',
        help='input data root directory.',
    )
    args = parser.parse_args()

    data_root = args.data_root

    # data 폴더 내에 images 폴더가 존재하지 않는다면, train 폴더명을 images로 변경
    if 'images' not in os.listdir(data_root):
        rename_folder.train2images(data_root)

    convert2coco(data_root)