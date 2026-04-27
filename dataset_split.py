import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# directories
image1_dir = "/home/hornet/dataset_folders/skin_diseases/archive(1)/HAM10000_images_part_1"
image2_dir = "/home/hornet/dataset_folders/skin_diseases/archive(1)/HAM10000_images_part_2"
csv_dir = "/home/hornet/dataset_folders/skin_diseases/archive(1)/HAM10000_metadata.csv"
output_dir = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder"

# splits and classes
splits = ['train', 'val', 'test']
classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

# create folders
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# load dataframe (CPU - correct for this task)
df = pd.read_csv(csv_dir)

# train / val / test split (correct stratified split)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['dx'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['dx'], random_state=42
)

def copy_images(df, split_name):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['image_id'] + ".jpg"

        path1 = os.path.join(image1_dir, image_id)
        path2 = os.path.join(image2_dir, image_id)

        if os.path.exists(path1):
            src_path = path1
        elif os.path.exists(path2):
            src_path = path2
        else:
            print(f"Missing file: {image_id}")
            continue

        dest_path = os.path.join(output_dir, split_name, row['dx'], image_id)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        shutil.copyfile(src_path, dest_path)

# run copying
copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("✅ Output folder created successfully.")