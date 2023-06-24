# [합성데이터 기반 객체 탐지 AI 경진대회](https://dacon.io/competitions/official/236107/overview/description)

[![support](https://img.shields.io/badge/Support-Linux-blue)](#running-locally)

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

- Clone the repository

```
git clone https://github.com/jeongjae96/synthesis-car-od.git
```

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

- 경로 이동(synthesis-car-od/models/mmdetection/)

```
cd ../models/mmdetection/
```

- [Prerequisites & Installation](https://github.com/jeongjae96/synthesis-car-od/tree/main/models/mmdetection#v2252)

- [Train & Inference](https://github.com/jeongjae96/synthesis-car-od/tree/main/models/mmdetection#train)

#### Yolo Models

- 경로 이동(synthesis-car-od/models/yolo/)

```
cd ../yolo/
```

- [Prerequisites & Installation](https://github.com/jeongjae96/synthesis-car-od/blob/main/models/yolo/readme.md#anaconda%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%B4-python-38-%EB%B2%84%EC%A0%84%EC%9D%98-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%83%9D%EC%84%B1)

- [Train & Inference](https://github.com/jeongjae96/synthesis-car-od/blob/main/models/yolo/readme.md#train)

### Bbox Ensemble

```
# 경로 이동(synthesis-car-od/)
cd ../../../
```

```
# 가상환경 구축
conda create -n ensemble python=3.9 -y
conda activate ensemble
```

```
# 필요 라이브러리 설치
pip install -r requirements/ensemble.txt   
```

- ```ensemble/WBF_ensemble.ipynb``` 파일에서 ```ensemble``` 커널 설정 후, 실행

- 최종 제출 파일: ```submissions/submission.csv```