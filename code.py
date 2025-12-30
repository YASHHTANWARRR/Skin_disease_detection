import os
import shutil
import numpy as np 
import pandas as pd
import torch 
import matplotlib.pyplot as plt 
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

#directories
basic_dir = "C:\\Users\\Hp\\Documents\\dataset folders\\skin disease dataset\\6GB HAM10000 DATA\\hmnist_28_28_RGB.csv"
image1_dir = "C:\\Users\\Hp\\Documents\\dataset folders\\skin disease dataset\\6GB HAM10000 DATA\\HAM10000_images_part_1"
image2_dir = "C:\\Users\\Hp\\Documents\\dataset folders\\skin disease dataset\\6GB HAM10000 DATA\\HAM10000_images_part_2"
csv_dir = "C:\\Users\\Hp\\Documents\\dataset folders\\skin disease dataset\\6GB HAM10000 DATA\\HAM10000_metadata.csv"
output_dir = "C:\\Users\\Hp\\Documents\\dataset folders\\skin disease dataset\\6GB HAM10000 DATA\\output_folder"

#output folders
split = ['train', 'val', 'test']
classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

df= pd.read_csv(csv_dir)


#train validation and testing split

train_df,temp_df = train_test_split(
    df ,test_size=0.30,stratify=df['dx'],random_state=42
    )

val_df,test_df=train_test_split(
    temp_df,test_size=0.50,stratify=temp_df['dx'],random_state=42
    )

def copy_images(dataframes,split_name):
    for _, row in tqdm (dataframes.iterrows(),total=len(dataframes)):
        img_name=row["image_id"]+".jpg"
        