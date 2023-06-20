## YOLO pipeline
---

<br>

### Anaconda를 이용해 Python 3.8 버전의 가상환경 생성
```bash
conda create -n yolo_test python=3.8 -y
```
### 가상환경 활성화
```bash
conda activate yolo_test
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