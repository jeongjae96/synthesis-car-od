import os
from pycocotools.coco import COCO

from torch.utils.data import DataLoader

from config import CFG
from utils import get_working_dir, seed_everything, set_device
from data_loader import CustomDataset, get_train_transforms, get_test_transforms, collate_fn
from model import build_model

print(CFG)

# -- settings
seed_everything(CFG['SEED'])
working_dir = get_working_dir(os.getcwd())
data_dir = os.path.join(working_dir, 'data/')
device = set_device()

print(f'working directory: {working_dir}')
print(f'device: {device}')

# -- data_load
# load coco
train_coco = COCO(os.path.join(data_dir, 'coco_sgkf/', f'train_{CFG["FOLD_NUM"]}.json'))
val_coco = COCO(os.path.join(data_dir, 'coco_sgkf/', f'val_{CFG["FOLD_NUM"]}.json'))
test_coco = COCO(os.path.join(data_dir, 'coco_test.json'))

train_dataset = CustomDataset(
    train_coco,
    data_dir,
    transforms=get_train_transforms()
)
val_dataset = CustomDataset(
    val_coco,
    data_dir,
    transforms=get_test_transforms()
)
test_dataset = CustomDataset(
    test_coco,
    data_dir,
    transforms=get_test_transforms
)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# -- model
model = build_model()

# -- train
