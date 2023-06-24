## YOLO pipeline
---

<br>

### Anaconda를 이용해 Python 3.8 버전의 가상환경 생성
```bash
conda create -n yolo python=3.8 -y
```
### 가상환경 활성화
```bash
conda activate yolo
```

### torch install
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

### requirements.txt를 이용해 필요한 패키지 설치
```bash
pip install -r requirements/yolo.txt
```

### labelmetxt to yolo label
```bash
cd data_preprocess/labelmetxt_converter/
```
```bash
python labelmetxt2yolo.py
```

---

# YOLOv7

## [YOLOv7](https://github.com/WongKinYiu/yolov7)

### Prerequisites

```bash
# start yolov7
cd yolov7
```

```bash
# download yolov7 weights
bash get_yolov7_weights.sh
```

### Train

```bash
python train_aux.py --workers 8 --device 0 --batch-size 8 --data ../../../data/yolo/all.yaml --img 1024 1024 --cfg cfg/training/yolov7-e6e.yaml --weights weights/yolov7-e6e.pt --name yolov7-e6e --hyp data/hyp.scratch.p6.yaml --epochs 200 --save_period 50 --cache-images
```

### Inference
```bash
python test.py --data ../../../data/yolo/all.yaml --img-size 1024 --batch-size 32 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights runs/train/yolov7-e6e/weights/last.pt --name yolov7-e6e --task test --verbose --save-conf --save-txt
```

### YOLO Label to Submission
```bash
python label2submission.py --data ../../../data/ --target yolov7-e6e --result ../../../submissions/yolov7.csv
```


---
# YOLOv8


### run yolov8/yolo_pipeline.ipynb