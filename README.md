# [합성데이터 기반 객체 탐지 AI 경진대회](https://dacon.io/competitions/official/236107/overview/description)

## File Structure

```
.
├── README.md
├── data
│   ├── images
│   │   ├── syn_*.txt
│   │   └── syn_*.png
│   ├── labels
│   │   └── syn_*.txt
│   ├── coco
│   │   ├── train.json
│   │   ├── test.json
│   │   ├── train_*.json
│   │   └── val_*.json
│   ├── yolo
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── val_*.txt
│   │   ├── train_*.txt
│   │   ├── fold_*.yaml
│   │   └── all.yaml
│   ├── test
│   │    └── *.png
│   ├── classes.txt
│   └── sample_submission.csv
├── models
│    ├── mmdetection
│    │   └── ...
│    └── yolo
│        ├── yolov7
│        │   └── ...
│        └── yolov8
│            └── ...
├── requirements
│    └── *.txt
├── data_preprocess
│    └── ...
├── submissions
│    └── ...
└── examples
     └── ...
```

## Process

### prerequisites

- ```data``` 폴더 생성 후, [데이터셋](https://dacon.io/competitions/official/236107/data)을 [File Structure](https://github.com/jeongjae96/synthesis-car-od#file-structure)에 맞게 이동

```
.
└── data
    ├── train
    │   ├── syn_*.txt
    │   └── syn_*.png
    ├── test
    │    └── *.png
    ├── classes.txt
    └── sample_submission.csv
```

- ```submissions``` 폴더 생성

```
# submissions csv를 저장할 폴더
mkdir submissions
```

- MMdetection, Yolo 모델 input 데이터를 위한 포맷 변환

```
# 가상환경 구축
conda create -n data_preprocess python=3.9 -y
conda activate data_preprocess
```

```
# 필요 라이브러리 설치
pip install -r requirements/data_preprocess.txt   
```

```
# 데이터 포맷 변환
cd data_preprocess
python main.py
```

### Train & Inference

#### MMdetection Models

- 경로 이동

```
cd ../models/mmdetection/
```

- Prerequisites & Installation (추후 링크 연결 예정)

- Train & Inference (추후 링크 연결 예정)

#### Yolo Models