# Data Preprocess

- ```main.py```: Labelme TXT format을 YOLO & COCO format으로 변환

```
python main.py -d {data_root} -n {n_splits} -s {seed}
```

## labelmetxt_converter

- ```labelmetxt2yolo.py```: Labelme TXT format을 YOLO format으로 변환 및 stratified group kfold를 이용해 train/validation split

```
python labelmetxt2yolo.py -d {data_root} -n {n_splits} -s {seed}
```

- ```labelmetxt2coco.py```: COCO train/test data로 변환

```
python labelmetxt2coco.py -d {data_root}
```

## split_data

- ```split_coco.py```: COCO train 데이터를 stratified group kfold를 이용해 train/validation split

```
python split_coco.py -d {data_root} -n {n_splits} -s {seed}
```

# Data Format

## [Our Data Format](https://dacon.io/competitions/official/236107/data)

- 이미지는 png 파일로 주어지고, 동일한 파일명으로 매핑되는 txt 파일로 annotation 정보가 제공됩니다.
- txt 파일 내에 한 객체당 한 줄로 ```class_id x1 y1 x2 y2 x3 y3 x4 y4``` 형식으로 annotaion 정보가 제공됩니다.
- LabelMe 형식의 Bounding Box 좌표
    - x1: 객체 좌상단 x좌표
    - y1: 객체 좌상단 y좌표
    - x2: 객체 우상단 x좌표
    - y2: 객체 우상단 y좌표
    - x3: 객체 우하단 x좌표
    - y3: 객체 우하단 y좌표
    - x4: 객체 좌하단 x좌표
    - y4: 객체 좌하단 y좌표
- ```classes.txt``` 파일에 ```class_id, class_name```이 존재합니다.

## [COCO Data Format](https://cocodataset.org/#format-data)

### Basic Data Structure using JSON

```
{
    "info": info, # 필수 아님
    "images": [image], # 필수
    "annotations": [annotation], # 필수
    "licenses": [license], # 필수 아님
}

# 필수 아님
info{
    "year": int, 
    "version": str, 
    "description": str, 
    "contributor": str, 
    "url": str, 
    "date_created": datetime,
}

image{
    "id": int, # 필수
    "width": int, # 필수
    "height": int, # 필수
    "file_name": str, # 필수
    "license": int, # 필수 아님
    "flickr_url": str, # 필수 아님
    "coco_url": str, # 필수 아님
    "date_captured": datetime, # 필수
}

# 필수 아님
license{
    "id": int, 
    "name": str, 
    "url": str,
}
```

### Object Detection Annotation Structure

```
annotation{
    "id": int, # 필수 아님
    "image_id": int,  # 필수
    "category_id": int, # 필수
    "segmentation": RLE or [polygon], # 필수 아님
    "area": float, # 필수 아님
    "bbox": [x,y,width,height], # 필수 
    "iscrowd": 0 or 1, # 0: polygon, 1: RLE # 필수 아님
}

categories[{
    "id": int, # 필수
    "name": str, # 필수
    "supercategory": str, # 필수 아님
}]
```