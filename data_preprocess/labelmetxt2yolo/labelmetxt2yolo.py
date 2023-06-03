import os
import glob
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import yaml
import cv2
from sklearn.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser(
    description='convert labelme txt format to YOLO format.'
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

TRAIN_PATH = os.path.join(args.data_path, 'train/')
TEST_PATH = os.path.join(args.data_path, 'test/')
LABEL_PATH = os.path.join(args.data_path, 'labels/')
YOLO_PATH = os.path.join(args.data_path, 'yolo/')

os.makedirs(LABEL_PATH, exist_ok=True)
os.makedirs(YOLO_PATH, exist_ok=True)

def make_yolo_dataset(image_paths, txt_paths):
    for image_path, txt_path in zip(tqdm(image_paths), txt_paths):
        img = cv2.imread(image_path)

        with open(txt_path, 'r') as txt_file:
            yolo_labels = []

            img_h, img_w, _ = img.shape
            lines = txt_file.readlines()

            for line in lines:
                line = list(map(float, line.strip().split(' ')))

                label = int(line[0])

                bbox = line[1:]

                xs = [bbox[i] for i in range(0, 8, 2)]
                ys = [bbox[i] for i in range(1, 8, 2)]

                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)

                box_w = x_max - x_min 
                box_h = y_max - y_min

                normalized_x = (x_min + x_max) / (2 * img_w)
                normalized_y = (y_min + y_max) / (2 * img_h)

                normalized_box_w = box_w / img_w
                normalized_box_h = box_h / img_h

                yolo_labels.append(f'{label} {normalized_x} {normalized_y} {normalized_box_w} {normalized_box_h}')

        dest_label_path = os.path.join(LABEL_PATH, os.path.basename(txt_path))

        with open(dest_label_path, 'w') as label_file:
            for yolo_label in yolo_labels:
                    label_file.write(f'{yolo_label}\n')

train_imgs = sorted(glob.glob(os.path.join(TRAIN_PATH, '*.png')))
train_txts = sorted(glob.glob(os.path.join(TRAIN_PATH, '*.txt')))

print('making YOLO label txt files...')
make_yolo_dataset(train_imgs, train_txts)

img_names = []
labels = []

for img, txt in zip(train_imgs, train_txts):
    img_name = img.replace('\\', '/').split('/')[-1]
    txt_name = txt.replace('\\', '/').split('/')[-1]

    with open(txt, 'r') as t:
        lines = t.readlines()

        for line in lines:
            line = line.strip()
            label = int(float(line.split(' ')[0]))
            bbox = ' '.join(line.split(' ')[1:])

            img_names.append(img_name)
            labels.append(label)

sgkf = StratifiedGroupKFold(
    n_splits=args.n_splits,
    shuffle=True,
    random_state=args.random_state,
)

img_names = np.array(img_names)

print('splitting data using stratified group kfold...')
for i, (train_idx, val_idx) in enumerate(sgkf.split(img_names, labels, img_names)):

    train_imgs = np.unique(img_names[train_idx])
    train_imgs = list(map(lambda x : os.path.abspath(x).replace('\\', '/'), train_imgs))

    val_imgs = np.unique(img_names[val_idx])
    val_imgs = list(map(lambda x : os.path.abspath(x).replace('\\', '/'), val_imgs))

    with open(os.path.join(YOLO_PATH, f'train_{i}.txt'), 'w') as path_file:
        for img in train_imgs:
            path_file.write(f'{img}\n')

    with open(os.path.join(YOLO_PATH, f'val_{i}.txt'), 'w') as path_file:
        for img in val_imgs:
            path_file.write(f'{img}\n')

test_imgs = sorted(glob.glob(os.path.join(TEST_PATH, '*.png')))
test_imgs = list(map(lambda x : os.path.abspath(x).replace('\\', '/'), test_imgs))

print('saving test info...')
with open(os.path.join(YOLO_PATH, 'test.txt'), 'w') as path_file:
    for img in test_imgs:
        path_file.write(f'{img}\n')

category_names = []

with open(os.path.join(args.data_path, 'classes.txt'), 'r') as cls_file:
    lines = cls_file.readlines()

    for line in lines:
        line = line.strip()

        # category_name
        category_name = line.split(',')[-1]

        category_names.append(category_name)

categories = dict()

for i, name in enumerate(category_names):
    categories[i] = name

print('saving YAML configuration...')
for i in range(args.n_splits):
    yaml_config = {
    'path' : os.path.abspath(args.data_path).replace('\\', '/') + '/',
    'train' : os.path.join(YOLO_PATH, f'train_{i}.txt').replace(args.data_path, ''),
    'val' : os.path.join(YOLO_PATH, f'val_{i}.txt').replace(args.data_path, ''),
    'test' : os.path.join(YOLO_PATH, f'test.txt').replace(args.data_path, ''),
    'nc' : len(category_names),
    'names' : categories,
    }

    with open(os.path.join(YOLO_PATH, f'fold_{i}.yaml'), 'w') as file:
        yaml.dump(yaml_config, file, sort_keys=False)