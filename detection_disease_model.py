import os
from matplotlib import transforms
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch 
import tensorflow as tf 
import sklearn
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report,f1_score,confusion_matrix,accuracy_score,precision_score,recall_score



#device configuration
device= torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(f'using device:',device)

#dataset directories 
test_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\test'
train_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\train'
val_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\val'

#transforming the training images for cnn
train_transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                    std=[0.5,0.5,0.5])
])

#transforming testing images for cnn 
test_transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                        std=[0.5,0.5,0.5])
])

#transforming validation images for cnn 
val_transform= transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                    std=[0.5,0.5,0.5])
])



