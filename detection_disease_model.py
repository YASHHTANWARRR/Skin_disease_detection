import os
from matplotlib import transforms
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch 
import tensorflow as tf 
import sklearn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import classification_report,f1_score,confusion_matrix,accuracy_score,precision_score,recall_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from torchvision import transforms,models

#device configuration
device= torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(f'using device:',device)

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

#dataset directories 
test_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\test'
train_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\train'
val_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\val'

#image folders and dataloaders
train_data=ImageFolder(root=train_split,transform=train_transform)
test_data=ImageFolder(root=test_split,transform=test_transform)
val_data=ImageFolder(root=val_split,transform=val_transform)

train_loader=DataLoader(dataset=train_data,
                        batch_size=32,
                        shuffle=True)

test_loader=DataLoader(dataset=test_data,
                    batch_size=32,
                    shuffle=False)

val_loader=DataLoader(dataset=val_data,
                    batch_size=32,
                    shuffle=False)

#model
class SkinLesionCNN(nn.Module):
    def __init__(self):
        super(SkinLesionCNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)# zeroth layer
        self.bn1=nn.BatchNorm2d(16)#first layer
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)#second layer
        self.pool2=nn.MaxPool2d(2,2)
        self.bn2=nn.BatchNorm2d(32)#third layer
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)#fourth layer
        self.pool3=nn.MaxPool2d(2,2)#fifth layer
        self.bn3=nn.BatchNorm2d(64)#sixth layer
        self.conv4=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)#seventh layer
        self.pool4=nn.MaxPool2d(2,2)
        self.bn4=nn.BatchNorm2d(128)#eighth layer
        self.conv5=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)#ninth layer

        #fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)
        
        def forward(self, x):
        x = F.relu(self.conv1(x))   # (28,28,16)
        x = self.pool(x)            # (14,14,16)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)            # (7,7,64)
        x = self.bn2(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))

        return F.softmax(self.out(x), dim=1)


