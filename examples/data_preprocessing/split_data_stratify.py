import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


# Assume that we have a function that can read labels from a .txt file
def read_label_from_txt(txt_file):
    with open(txt_file, "r") as f:
        # For simplicity, we assume that the first line of each .txt file contains the main label
        label = f.readline().strip()
    return label


# Define path to images and annotations
path_to_data = "data/train"  # Your path to the data

data_list = []

# Get list of all .png files in the directory
image_files = [f for f in os.listdir(path_to_data) if f.endswith(".png")]

for idx, image_file in enumerate(image_files):
    txt_file = os.path.join(path_to_data, image_file.replace(".png", ".txt"))

    if os.path.isfile(txt_file):
        label = read_label_from_txt(txt_file)
        label = label.split()[0]
        # print(idx, label)
        data_list.append({"file": image_file, "label": label})

# Create a DataFrame from the list
df = pd.DataFrame(data_list)

# Split the data into train and test sets
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# Saving to csv for further usage
train_df.to_csv("train_set.csv", index=False)
val_df.to_csv("val_set.csv", index=False)


# Define the source and destination directories
source_dir = "data/train"  # Your path to the original data
dest_dir = "data/split"
train_dir = os.path.join(dest_dir, "train_set")  # Your path to the train set directory
val_dir = os.path.join(dest_dir, "val_set")  # Your path to the validation set directory

# Create the directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


# Function to copy both image and txt files
def copy_files(df, source_dir, destination_dir, data_type):
    for file_name in tqdm(df["file"], desc=f"Copying {data_type} files"):
        # Copy image file
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        if os.path.isfile(source_file) and not os.path.isfile(destination_file):
            shutil.copy2(source_file, destination_file)

        # Copy corresponding txt file
        txt_file_name = file_name.replace(".png", ".txt")
        source_file_txt = os.path.join(source_dir, txt_file_name)
        destination_file_txt = os.path.join(destination_dir, txt_file_name)
        if os.path.isfile(source_file_txt) and not os.path.isfile(destination_file_txt):
            shutil.copy2(source_file_txt, destination_file_txt)


# Copy the train files
copy_files(train_df, source_dir, train_dir, data_type="train")

# Copy the validation files
copy_files(val_df, source_dir, val_dir, data_type="val")
