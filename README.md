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
├── data_preprocess
│    └── ...
└── examples
     └── ...
```