from glob import glob
from tqdm import tqdm
import cv2
import pandas as pd

def yolo_to_labelme(line, image_width, image_height, txt_file_name):    
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]
    
    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)
    
    return file_name, int(class_id), confidence, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max


def convert_yolo2labelme(data_path, target, csv_path):
  infer_txts = glob(f"runs/test/{target}/labels/*.txt")

  results = []
  for infer_txt in tqdm(infer_txts):
      base_file_name = infer_txt.split("/")[-1].split(".")[0]
      imgage_height, imgage_width = cv2.imread(f"{data_path}test/{base_file_name}.png").shape[:2]        
      with open(infer_txt, "r") as reader:        
          lines = reader.readlines()        
          for line in lines:
              results.append(yolo_to_labelme(line, imgage_width, imgage_height, infer_txt))

  df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
  df_submission.sort_values(["confidence"], ascending=False, inplace=True)
  df_submission.to_csv(f"{csv_path}", index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='convert YOLO format to labelme txt format.'
    )
    parser.add_argument(
        '-d',
        '--data',
        dest='data',
        default='../../data/',
        help='image data path',
    )
    parser.add_argument(
        '-t',
        '--target',
        dest='target',
        default='yolov7-e6e16',
        help='yolo inference result target',
    )
    parser.add_argument(
        '-r',
        '--result',
        dest='result',
        default='../../../submissions/yolov7.csv',
        help='yolo inference result save csv file path',
    )

    args = parser.parse_args()

    convert_yolo2labelme(
        args.data,
        args.target,
        args.result
    )