# [MMDetection](https://mmdetection.readthedocs.io/en/latest/)

- MMDetection works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.6+.

## Prerequisites

1.  Create a conda environment and activate it.

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)

## Installation

1. Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

2. Install MMDetection.

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Verify the Installation

1. Download config and checkpoint files.

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

2. Verify the inference demo.

```
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```