import os
import random
import numpy as np

import torch

# working directory 경로 설정
def get_working_dir(cur_path, working_dir='synthesis-car-od'):
    path_len = len(working_dir)

    cur_path = cur_path.replace('\\', '/')
    index = cur_path.find(working_dir)

    return cur_path[:index + path_len + 1]

# 동일한 실험 환경을 위한 시드 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# device 설정
def set_device():
    if torch.cuda.is_available(): # NVIDIA CUDA
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available(): # MAC
        device = torch.device('mps')
    else: # GPU 없다면 CPU
        device = torch.device('cpu')

    return device