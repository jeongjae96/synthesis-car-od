import os
import argparse

from utils import rename_folder
from labelmetxt_converter.labelmetxt2yolo import convert2yolo
from labelmetxt_converter.labelmetxt2coco import convert2coco
from split_data.split_coco import split_coco_sgkf

parser = argparse.ArgumentParser(
    description='convert label txt format to COCO & YOLO and split data using stratified group kfold.'
)
parser.add_argument(
    '-d',
    '--data_root',
    dest='data_root',
    default='../data/',
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
n_splits = args.n_splits
seed = args.seed

# data 폴더 내에 images 폴더가 존재하지 않는다면, train 폴더명을 images로 변경
if 'images' not in os.listdir(data_root):
    rename_folder.train2images(data_root)

convert2yolo(
    data_root,
    n_splits,
    seed
)
convert2coco(data_root)
split_coco_sgkf(
    data_root,
    n_splits,
    seed
)