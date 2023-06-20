import os
import glob
from tqdm.auto import tqdm
import yaml
import numpy as np
import cv2
from sklearn.model_selection import StratifiedGroupKFold

def normalize_labelme_bbox(
    labelme_bbox,
    image_width,
    image_height
):
    xs = [labelme_bbox[i] for i in range(0, 8, 2)]
    ys = [labelme_bbox[i] for i in range(1, 8, 2)]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    box_w = x_max - x_min 
    box_h = y_max - y_min

    normalized_x = (x_min + x_max) / (2 * image_width)
    normalized_y = (y_min + y_max) / (2 * image_height)

    normalized_box_w = box_w / image_width
    normalized_box_h = box_h / image_height

    return normalized_x, normalized_y, normalized_box_w, normalized_box_h

def label2yolo(labelme_path, yolo_label_path):
    image_paths = sorted(glob.glob(os.path.join(labelme_path, '*.png')))
    labelme_label_paths = sorted(glob.glob(os.path.join(labelme_path, '*.txt')))

    print('making yolo labels...')
    for image_path, labelme_label_path in zip(tqdm(image_paths), labelme_label_paths):
        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape

        with open(labelme_label_path, 'r') as labelme_txt:
            yolo_labels = []

            lines = labelme_txt.readlines()

            for line in lines:
                line = list(map(float, line.strip().split(' ')))
                label = int(line[0])
                labelme_bbox = line[1:]
                normalized_x, normalized_y, normalized_box_w, normalized_box_h = normalize_labelme_bbox(labelme_bbox, img_w, img_h)
                yolo_labels.append(f'{label} {normalized_x} {normalized_y} {normalized_box_w} {normalized_box_h}')

        yolo_label_file_path = os.path.join(yolo_label_path, os.path.basename(labelme_label_path))

        with open(yolo_label_file_path, 'w') as yolo_label_file:
            for yolo_label in yolo_labels:
                yolo_label_file.write(f'{yolo_label}\n')

def save_yolo_image_path( 
    image_paths,
    dest_path
):
    image_paths = list(map(lambda x : os.path.abspath(x).replace('\\', '/'), image_paths))

    print(f'saving {os.path.basename(dest_path)}...')
    with open(dest_path, 'w') as image_path_file:
        for image_path in image_paths:
            image_path_file.write(f'{image_path}\n')

def split_yolo_sgkf(
    labelme_path,
    yolo_path,
    n_splits,
    seed
):
    image_paths = np.array(sorted(glob.glob(os.path.join(labelme_path, '*.png'))))
    labelme_label_paths = sorted(glob.glob(os.path.join(labelme_path, '*.txt')))

    imgs = []
    labels = []

    for img, txt in zip(image_paths, labelme_label_paths):
        with open(txt, 'r') as t:
            lines = t.readlines()

            for line in lines:
                line = line.strip()
                label = int(float(line.split(' ')[0]))

                imgs.append(img)
                labels.append(label)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    imgs = np.array(imgs)

    print('splitting data using stratified group kfold...')
    for i, (train_idx, val_idx) in enumerate(sgkf.split(imgs, labels, imgs)):
        train_imgs = np.unique(imgs[train_idx])
        train_file_path = os.path.join(yolo_path, f'train_{i}.txt')
        save_yolo_image_path(train_imgs, train_file_path)

        val_imgs = np.unique(imgs[val_idx])
        val_file_path = os.path.join(yolo_path, f'val_{i}.txt')
        save_yolo_image_path(val_imgs, val_file_path)

def make_data_yaml(data_root, n_splits):
    yolo_path = os.path.join(data_root, 'yolo/')

    category_names = []

    with open(os.path.join(data_root, 'classes.txt'), 'r') as cls_file:
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

    train_yaml_config = {
        'train' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), f'train.txt'),
        'val' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), f'val_sample.txt'),
        'test' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), 'test.txt'),
        'nc' : len(category_names),
        'names' : categories,
    }

    with open(os.path.join(yolo_path, 'all.yaml'), 'w') as file:
        yaml.dump(train_yaml_config, file, sort_keys=False)

    for i in range(n_splits):
        # yaml_config = {
        #     'path' : os.path.abspath(yolo_path).replace('\\', '/') + '/',
        #     'train' : f'train_{i}.txt',
        #     'val' : f'val_{i}.txt',
        #     'test' : f'test.txt',
        #     'nc' : len(category_names),
        #     'names' : categories,
        # }
        yaml_config = {
            'train' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), f'train_{i}.txt'),
            'val' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), f'val_{i}.txt'),
            'test' : os.path.join(os.path.abspath(yolo_path).replace('\\', '/'), 'test.txt'),
            'nc' : len(category_names),
            'names' : categories,
        }

        with open(os.path.join(yolo_path, f'fold_{i}.yaml'), 'w') as file:
            yaml.dump(yaml_config, file, sort_keys=False)

def convert2yolo(
    data_root,
    n_splits,
    seed
):
    labelme_path = os.path.join(data_root, 'images/')
    yolo_label_path = os.path.join(data_root, 'labels/') # yolo label txt를 저장할 directory
    yolo_path = os.path.join(data_root, 'yolo/') # yolo 이미지 경로 및 yaml config 저장할 directory
    os.makedirs(yolo_label_path, exist_ok=True)
    os.makedirs(yolo_path, exist_ok=True)

    label2yolo(labelme_path, yolo_label_path)

    split_yolo_sgkf(
        labelme_path,
        yolo_path,
        n_splits,
        seed
    )

    train_image_paths = sorted(glob.glob(os.path.join(data_root, 'images/', '*.png')))
    train_dest_path = os.path.join(yolo_path, 'train.txt')
    save_yolo_image_path(train_image_paths, train_dest_path)

    sample_image_path = [train_image_paths[0]]
    sample_dest_path = os.path.join(yolo_path, 'val_sample.txt')
    save_yolo_image_path(sample_image_path, sample_dest_path)

    test_image_paths = sorted(glob.glob(os.path.join(data_root, 'test/', '*.png')))
    test_dest_path = os.path.join(yolo_path, 'test.txt')
    save_yolo_image_path(test_image_paths, test_dest_path)

    make_data_yaml(data_root, n_splits)
    
if __name__ == '__main__':
    import argparse

    import sys
    sys.path.append('..')
    from utils import rename_folder

    parser = argparse.ArgumentParser(
        description='convert labelme txt format to YOLO format.'
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

    convert2yolo(
        data_root,
        args.n_splits,
        args.seed
    )