import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import cupy as cp
from cuml.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transforms
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                        saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# dataset paths (FIX THESE)
train_split = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder/train"
test_split  = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder/test"

# datasets and loaders
train_data = ImageFolder(train_split, transform=train_transform)
test_data  = ImageFolder(test_split, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=8, shuffle=False)

# model (YOUR ORIGINAL)
class SkinLesionCNN(nn.Module):
    def __init__(self):
        super(SkinLesionCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))

        return self.out(x)

# initialization
model = SkinLesionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# training (CNN)
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# feature extraction (IMPORTANT PART FOR RAPIDS)
model.eval()

features = []
labels_list = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)

        # pass through conv layers ONLY
        x = model.pool(F.relu(model.conv1(images)))
        x = model.bn1(x)

        x = F.relu(model.conv2(x))
        x = model.pool(F.relu(model.conv3(x)))
        x = model.bn2(x)

        x = F.relu(model.conv4(x))
        x = F.relu(model.conv5(x))

        x = x.view(x.size(0), -1)

        features.append(x.cpu())
        labels_list.append(labels)

# convert to numpy
features = torch.cat(features).numpy()
labels = torch.cat(labels_list).numpy()

# move to GPU (RAPIDS)
X_gpu = cp.asarray(features)
y_gpu = cp.asarray(labels)

# RAPIDS model
rf_model = RandomForestClassifier()
rf_model.fit(X_gpu, y_gpu)

print("RAPIDS model trained successfully")

# testing
test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        x = model.pool(F.relu(model.conv1(images)))
        x = model.bn1(x)

        x = F.relu(model.conv2(x))
        x = model.pool(F.relu(model.conv3(x)))
        x = model.bn2(x)

        x = F.relu(model.conv4(x))
        x = F.relu(model.conv5(x))

        x = x.view(x.size(0), -1)

        test_features.append(x.cpu())
        test_labels.append(labels)

# convert test data
test_features = torch.cat(test_features).numpy()
test_labels = torch.cat(test_labels).numpy()

X_test_gpu = cp.asarray(test_features)

# predictions
preds = rf_model.predict(X_test_gpu)
preds_cpu = cp.asnumpy(preds)

# evaluation metrics
accuracy = accuracy_score(test_labels, preds_cpu)
precision = precision_score(test_labels, preds_cpu, average='weighted')
recall = recall_score(test_labels, preds_cpu, average='weighted')
f1 = f1_score(test_labels, preds_cpu, average='weighted')
conf_matrix = confusion_matrix(test_labels, preds_cpu)

print("\n===== RAPIDS MODEL EVALUATION =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n")
print(classification_report(
    test_labels,
    preds_cpu,
    target_names=train_data.classes
))