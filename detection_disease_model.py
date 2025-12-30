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
from tensorflow.keras.models import Sequential
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
        super(SkinLesionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn2(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))

        return self.out(x)  # ❗ NO softmax here

model=SkinLesionCNN().to(device)
print(model)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameter(),lr=0.01)

#code ends for disease detection model

# Training
epochs = 10
print("\nStarting training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total


#validation
model.eval()
val_loss= 0.0
val_correct=0
val_total=0

with torch.no_grad():
    for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        
        val_loss+= loss.item() * images.size(0)
        _, predicted = torch.max(outputs,1)
        val_total += labels.size(0)
        val_correct+= (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(val_data)
    val_epoch_acc = 100*val_correct/val_total
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    print(f"  Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


# Evaluation on test set
print("\nEvaluating on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Save the model
model_path = 'Skin_disease_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")