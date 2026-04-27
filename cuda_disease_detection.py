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
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pandas as pd

# output directory + auto run versioning
base_output = "output_results"
os.makedirs(base_output, exist_ok=True)

# auto increment run folder
existing_runs = [d for d in os.listdir(base_output) if d.startswith("run_")]
run_id = len(existing_runs) + 1

run_dir = os.path.join(base_output, f"run_{run_id}")
os.makedirs(run_dir, exist_ok=True)

print(f"Saving results to: {run_dir}")


# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),# was (28,28) but increased to (128,128) for better performance. Adjust as needed.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
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

# class imbalance handling
class_counts = np.bincount(train_data.targets)
class_weights = len(train_data.targets) / (len(class_counts) * class_counts)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

weights = 1. / class_counts
sample_weights = weights[train_data.targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


train_loader = DataLoader(train_data, batch_size=64, sampler=sampler,
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
        self.dropout = nn.Dropout(0.5)# added dropout Adjust rate from 0.3 to 0.5 after run 1.

        self.fc1 = nn.Linear(256 * 32 * 32, 256)# was 256 * 7 * 7, but changed to 256 * 32 * 32 due to increased input size. Adjust as needed.
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
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)#lr was 0.001, but reduced to 0.0003 for better convergence. Adjust as needed.


# training and validation
epochs = 40 # first iteration was 20, but increased to 40 for better performance. Adjust as needed.

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
torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))


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

torch.cuda.empty_cache()  #  FIX: clear GPU memory before feature extraction

# RAPIDS feature extraction
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

        features.append(x.detach().cpu())   # FIX: move features to CPU
        labels_list.append(labels.cpu())    #  FIX: keep labels on CPU

        del x                                #  FIX: free tensor
        torch.cuda.empty_cache()             #  FIX: prevent GPU memory buildup
        

X_gpu = cp.asarray(torch.cat(features).cpu().numpy())
y_gpu = cp.asarray(torch.cat(labels_list).cpu().numpy())

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    max_features='sqrt'
)
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

        test_features.append(x.detach().cpu())   # FIX: move to CPU
        test_labels.append(labels) 
        
        del x                                    # FIX
        torch.cuda.empty_cache()                 # FIX

X_test = cp.asarray(torch.cat(test_features).cpu().numpy())
y_test = torch.cat(test_labels).numpy()

preds = cp.asnumpy(rf_model.predict(X_test))


# evaluation
conf_matrix = confusion_matrix(y_test, preds)

print("\n===== FINAL RESULTS =====")
print("CNN Accuracy    :", cnn_acc)
rapids_acc = accuracy_score(y_test, preds)
print("RAPIDS Accuracy :", rapids_acc)


report = classification_report(
    y_test, preds,
    target_names=train_data.classes,
    zero_division=0
)
print(report)


# save metrics
with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
    f.write(report)


# confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
plt.close()

# AUTO COMPARISON CSV
summary_path = os.path.join(base_output, "summary.csv")

new_row = {
    "run": run_id,
    "cnn_acc": cnn_acc,
    "rapids_acc": rapids_acc
}

if os.path.exists(summary_path):
    df = pd.read_csv(summary_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])

df.to_csv(summary_path, index=False)

print("\nSaved comparison to summary.csv")


# prediction function
def predict_image(img_path):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128,128)),# was (28,28) but increased to (128,128) for better performance. Adjust as needed.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    print("Predicted class:", train_data.classes[pred.item()])