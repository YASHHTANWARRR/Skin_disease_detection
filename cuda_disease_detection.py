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
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm


# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# transforms
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# dataset paths
train_split = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder/train"
val_split   = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder/val"
test_split  = "/home/hornet/dataset_folders/skin_diseases/archive(1)/output_folder/test"


# datasets and loaders (optimized)
train_data = ImageFolder(train_split, transform=train_transform)
val_data   = ImageFolder(val_split, transform=val_transform)
test_data  = ImageFolder(test_split, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)

val_loader   = DataLoader(val_data, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)

test_loader  = DataLoader(test_data, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)


# model
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


# training and validation
epochs = 20

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}")


# save model
torch.save(model.state_dict(), "cnn_model.pth")


# test CNN
model.eval()
cnn_preds = []
cnn_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        cnn_preds.extend(preds.cpu().numpy())
        cnn_labels.extend(labels.numpy())

cnn_acc = accuracy_score(cnn_labels, cnn_preds)


# RAPIDS feature extraction (optimized: no unnecessary conversions)
features = []
labels_list = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)

        x = model.pool(F.relu(model.conv1(images)))
        x = model.bn1(x)
        x = F.relu(model.conv2(x))
        x = model.pool(F.relu(model.conv3(x)))
        x = model.bn2(x)
        x = F.relu(model.conv4(x))
        x = F.relu(model.conv5(x))
        x = x.view(x.size(0), -1)

        features.append(x.detach())
        labels_list.append(labels.to(device))

# direct GPU conversion (better)
X_gpu = cp.asarray(torch.cat(features).cpu().numpy())
y_gpu = cp.asarray(torch.cat(labels_list).cpu().numpy())

rf_model = RandomForestClassifier()
rf_model.fit(X_gpu, y_gpu)


# RAPIDS testing
test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)

        x = model.pool(F.relu(model.conv1(images)))
        x = model.bn1(x)
        x = F.relu(model.conv2(x))
        x = model.pool(F.relu(model.conv3(x)))
        x = model.bn2(x)
        x = F.relu(model.conv4(x))
        x = F.relu(model.conv5(x))
        x = x.view(x.size(0), -1)

        test_features.append(x.detach())
        test_labels.append(labels)

X_test = cp.asarray(torch.cat(test_features).cpu().numpy())
y_test = torch.cat(test_labels).numpy()

preds = cp.asnumpy(rf_model.predict(X_test))


# evaluation
conf_matrix = confusion_matrix(y_test, preds)

print("\n===== FINAL RESULTS =====")
print("CNN Accuracy    :", cnn_acc)
print("RAPIDS Accuracy :", accuracy_score(y_test, preds))

print(classification_report(y_test, preds, target_names=train_data.classes))


# confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# prediction function
def predict_image(img_path):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    print("Predicted class:", train_data.classes[pred.item()])
