import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import os
import glob
import argparse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='inference MMDetection models.')
parser.add_argument(
    '-c',
    '--config_path',
    dest='config_path',
    help='config path'
)
parser.add_argument(
    '-r',
    '--root_dir',
    dest='root_dir',
    default='../../',
    help='root directory'
)
parser.add_argument(
    '-w',
    '--weights',
    dest='weights',
    help='trained weights'
)

args = parser.parse_args()

classes = (
    "chevrolet_malibu_sedan_2012_2016",
    "chevrolet_malibu_sedan_2017_2019",
    "chevrolet_spark_hatchback_2016_2021",
    "chevrolet_trailblazer_suv_2021_",
    "chevrolet_trax_suv_2017_2019",
    "genesis_g80_sedan_2016_2020",
    "genesis_g80_sedan_2021_",
    "genesis_gv80_suv_2020_",
    "hyundai_avante_sedan_2011_2015",
    "hyundai_avante_sedan_2020_",
    "hyundai_grandeur_sedan_2011_2016",
    "hyundai_grandstarex_van_2018_2020",
    "hyundai_ioniq_hatchback_2016_2019",
    "hyundai_sonata_sedan_2004_2009",
    "hyundai_sonata_sedan_2010_2014",
    "hyundai_sonata_sedan_2019_2020",
    "kia_carnival_van_2015_2020",
    "kia_carnival_van_2021_",
    "kia_k5_sedan_2010_2015",
    "kia_k5_sedan_2020_",
    "kia_k7_sedan_2016_2020",
    "kia_mohave_suv_2020_",
    "kia_morning_hatchback_2004_2010",
    "kia_morning_hatchback_2011_2016",
    "kia_ray_hatchback_2012_2017",
    "kia_sorrento_suv_2015_2019",
    "kia_sorrento_suv_2020_",
    "kia_soul_suv_2014_2018",
    "kia_sportage_suv_2016_2020",
    "kia_stonic_suv_2017_2019",
    "renault_sm3_sedan_2015_2018",
    "renault_xm3_suv_2020_",
    "ssangyong_korando_suv_2019_2020",
    "ssangyong_tivoli_suv_2016_2020",
)

root = os.path.join(args.root_dir, 'data/')

# config file 가져오기
config_file = glob.glob(os.path.join(args.config_path, '*.py'))[0]
cfg = Config.fromfile(config_file)

# data config 수정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = os.path.join(root, 'coco/test.json')
cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
cfg.data.test.test_mode = True
cfg.data.samples_per_gpu = 4
cfg.gpu_ids = [1]
cfg.work_dir = args.config_path

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    # samples_per_gpu=1,
    samples_per_gpu=8,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False
)

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, args.weights)
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load
model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader, show_score_thr=0.00) # output 계산

# submission 양식에 맞게 output 후처리
coco = COCO(cfg.data.test.ann_file)
class_num = len(classes)
results = pd.read_csv(os.path.join(root, 'sample_submission.csv'))

file_names = []
class_ids = []
confidences = []
x_mins = []
y_mins = []
x_maxes = []
y_maxes = []

for i, out in enumerate(tqdm(output)):
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    file_name = os.path.basename(image_info['file_name'])

    for j in range(class_num):
        for x_min, y_min, x_max, y_max, score in out[j]:
            # file_name : test 파일 이름
            # class_id : 검출한 객체 id
            # confidence : 검출한 객체의 정확도(0~1)
            # point1_x : 검출한 객체 좌상단 x좌표 == x_min
            # point1_y : 검출한 객체 좌상단 y좌표 == y_min
            # point2_x : 검출한 객체 우상단 x좌표 == x_max
            # point2_y : 검출한 객체 우상단 y좌표 == y_min
            # point3_x : 검출한 객체 우하단 x좌표 == x_max
            # point3_y : 검출한 객체 우하단 y좌표 == y_max
            # point4_x : 검출한 객체 좌하단 x좌표 == x_min
            # point4_y : 검출한 객체 좌하단 y좌표 == y_max

            file_names.append(file_name)
            class_ids.append(j)
            confidences.append(score)
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxes.append(x_max)
            y_maxes.append(y_max)

results['file_name'] = file_names
results['class_id'] = class_ids
results['confidence'] = confidences
results['point1_x'] = x_mins
results['point1_y'] = y_mins
results['point2_x'] = x_maxes
results['point2_y'] = y_mins
results['point3_x'] = x_maxes
results['point3_y'] = y_maxes
results['point4_x'] = x_mins
results['point4_y'] = y_maxes

results.sort_values(by=['confidence'], ascending=False, inplace=True)
results.to_csv(os.path.join(f'{args.root_dir}', 'submissions/', f'{os.path.basename(config_file).split(".")[0]}.csv'), index=False)