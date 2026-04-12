import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.utils.prune as prune

from cuml.ensemble import RandomForestClassifier
import cupy as cp

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
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

test_transform = val_transform

# dataset paths
train_split = r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\train'
val_split   = r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\val'
test_split  = r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\test'

# datasets and loaders
train_data = ImageFolder(train_split,
                        transform=train_transform)
val_data   = ImageFolder(val_split,
                        transform=val_transform)
test_data  = ImageFolder(test_split,
                        transform=test_transform)

train_loader = DataLoader(train_data,
                        batch_size=32,
                        shuffle=True)
val_loader   = DataLoader(val_data,
                        batch_size=32,
                        shuffle=False)
test_loader  = DataLoader(test_data, 
                        batch_size=32,
                        shuffle=False)

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

print(model)

# training
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
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

    train_loss = running_loss / len(train_data)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_data)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%\n")


#pruning and quantization added 
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)

quantized_model = torch.quantization.quantize_dynamic(
    model,                      
    {torch.nn.Linear},          
    dtype=torch.qint8
)

print("Quantization applied successfully")

print("Pruning applied")
# testing
model.eval()

start = time.time()

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)

end = time.time()
print("Original Model Time:", end - start)

start = time.time()

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = quantized_model(images)

end = time.time()
print("Quantized Model Time:", end - start)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = quantized_model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds,
                            target_names=train_data.classes))

# save model
torch.save(model.state_dict(), "Skin_disease_model.pth")
print("Model saved successfully.")

# RAPIDS RANDOM FOREST

def flatten_loader(loader):
    X = []
    y = []
    
    for images, labels in loader:
        images = images.view(images.size(0), -1)
        X.append(images.numpy())
        y.append(labels.numpy())
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    return X, y

print("\nPreparing data for RAPIDS...")

X_train, y_train = flatten_loader(train_loader)
X_test, y_test = flatten_loader(test_loader)

print("Training RAPIDS Random Forest...")

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

print("RAPIDS Random Forest Accuracy:",
        accuracy_score(y_test, rf_preds))