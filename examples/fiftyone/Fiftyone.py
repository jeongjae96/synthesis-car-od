#!/usr/bin/env python
# coding: utf-8

# In[3]:


import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

def fiftyone_hy(data_path, labels_path, sample_count=10, shuffle=True, seed=42):
    '''
    data_path    : 데이터 파일의 경로. data/train이 존재하는 경로를 지정해주면 됨.
    labels_path  : COCO형식으로 만들어진 labeling json 파일의 경로. 해당 파일의 이름까지 적어주면 됨.
    sample_count : default = 10. 정해진 sample수 만큼 이미지를 가져온다.
    shuffle      : default = True. dataset의 이미지를 shuffle한 후 가져옴.
    seed         : default = 42. seed값을 지정함.
    '''
    
    data_path = data_path
    labels_path = labels_path
    name = 'test'

    # Import dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        # name=name,
        max_samples=sample_count, 
        shuffle=shuffle, 
        seed=seed
    )

    view = dataset.view()
    print(view)
    
    # launch fiftyone app
    return fo.launch_app(dataset)

