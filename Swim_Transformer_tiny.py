# ================= SWIN TRANSFORMER =================

!pip install timm -q

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ===== CONFIG =====
DATASET_PATH = "/Dataset_path/input/cataract-grading/data"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "val")

NUM_CLASSES = 4
IMG_SIZE = 224   # Swin default
BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-5
MODEL_PATH = "swin_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORMS =====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== MODEL =====
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, factor=0.5
)

best_acc = 0.0

# ===== TRAINING LOOP =====
for epoch in range(EPOCHS):
    model.train()
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_dataset)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print("Best Validation Accuracy:", best_acc)

# ===== LOAD BEST MODEL =====
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===== CONFUSION MATRIX =====
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds,
                            target_names=train_dataset.classes))
