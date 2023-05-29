import os

from config import CFG
from utils import get_working_dir, seed_everything, set_device
from data_loader import split_load_train_val, load_test_data
from model import build_model

# -- settings
seed_everything(CFG['SEED'])
working_dir = get_working_dir(os.getcwd())
device = set_device()

print(f'working directory: {working_dir}')
print(f'device: {device}')

# # -- data_load
# train_loader, val_loader = split_load_train_val(working_dir)
# test_loader = load_test_data(working_dir)

# # -- model
# model = build_model()