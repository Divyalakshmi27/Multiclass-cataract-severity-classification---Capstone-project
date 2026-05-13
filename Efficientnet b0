import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_dir = "Dataset_path/train"  
val_dir   = "Dataset_path/val"    

# ======================
# Hyperparameters
# ======================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
PATIENCE = 5
NUM_CLASSES = 4

# ======================
# Transforms
# ======================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Dataset & Loaders
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Classes:", train_dataset.classes)

# ======================
# Model - EfficientNet B0
# ======================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Scheduler (compatible version)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.3,
    patience=3
)

# ======================
# Training Loop
# ======================
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # ---- TRAIN ----
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total

    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_efficientnet_b0.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # Early stopping
    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# ======================
# Load Best Model
# ======================
model.load_state_dict(best_model_wts)

print("\nBest Validation Accuracy:", best_acc)

# ======================
# Evaluation
# ======================
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
