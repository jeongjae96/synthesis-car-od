# MMDetection

## [v2.25.2](https://mmdetection.readthedocs.io/en/v2.2.0/install.html)

- Windows is not officially supported

### Installation

1. Create a conda environment and activate it.

```
conda create -n openmmlab_v2 python=3.8 -y
conda activate openmmlab_v2
```

2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)

```
# example
# CUDA 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
```

3. Clone the mmdetection v2.25.2 repository.

```
# pwd: ~/models/mmdetection/
git clone -b v2.25.2 https://github.com/open-mmlab/mmdetection.git
mv mmdetection mmdetection_v2
cd mmdetection_v2
```

4. Install build requirements and then install mmdetection.

```
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.0
pip install -r requirements/albu.txt
pip install tqdm
```

On macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

### Train

- Train ```cascade_rcnn_swinB```

```
python tools/train.py ../config/cascade_rcnn_swinB.py --seed 41 --deterministic --no-validate
```

- Train ```cascade_rcnn_swinL```

```
python tools/train.py ../config/cascade_rcnn_swinL.py --seed 41 --deterministic --no-validate
```

### Inference

- Move to ```mmdetection``` folder

```
cd ..
```

- Inference ```cascade_rcnn_swinB```

```
python inference.py -c mmdetection_v2/work_dirs/cascade_rcnn_swinB/ -w epoch_50.pth
```

- Inference ```cascade_rcnn_swinL```

```
python inference.py -c mmdetection_v2/work_dirs/cascade_rcnn_swinL/ -w epoch_40.pth
```

## [v3.0.0](https://mmdetection.readthedocs.io/en/latest/)

- MMDetection works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.6+.

### Prerequisites

1.  Create a conda environment and activate it.

```
conda create --name openmmlab_v3 python=3.8 -y
conda activate openmmlab_v3
```

2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)

### Installation

1. Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

2. Install MMDetection.

```
git clone https://github.com/open-mmlab/mmdetection.git
mv mmdetection mmdetection_v3
cd mmdetection_v3
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### Verify the Installation

1. Download config and checkpoint files.

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

2. Verify the inference demo.

```
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```