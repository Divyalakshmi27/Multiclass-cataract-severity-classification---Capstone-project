import os
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "/Dataset_path"
VAL_DIR      = os.path.join(DATASET_PATH, "val")

IMG_SIZE   = 224
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = val_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# MODELS
# ============================================================

efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1280, NUM_CLASSES)
)

swin = timm.create_model("swin_tiny_patch4_window7_224",
                         pretrained=False,
                         num_classes=NUM_CLASSES)

convnext = timm.create_model("convnext_tiny",
                             pretrained=False,
                             num_classes=NUM_CLASSES)

# Load weights (update paths if needed)
efficientnet.load_state_dict(torch.load("/Model_path1", map_location=DEVICE))
swin.load_state_dict(torch.load("/Model_path2", map_location=DEVICE))
convnext.load_state_dict(torch.load("/Model_path1", map_location=DEVICE))

efficientnet.to(DEVICE).eval()
swin.to(DEVICE).eval()
convnext.to(DEVICE).eval()

# ============================================================
# INFERENCE
# ============================================================

all_labels = []
preds_eff, preds_swin, preds_convnext = [], [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        out_eff  = efficientnet(images)
        out_swin = swin(images)
        out_conv = convnext(images)

        preds_eff.extend(out_eff.argmax(1).cpu().numpy())
        preds_swin.extend(out_swin.argmax(1).cpu().numpy())
        preds_convnext.extend(out_conv.argmax(1).cpu().numpy())

        all_labels.extend(labels.numpy())

preds_eff      = np.array(preds_eff)
preds_swin     = np.array(preds_swin)
preds_convnext = np.array(preds_convnext)
all_labels     = np.array(all_labels)

# ============================================================
# MAJORITY VOTING (NO SCIPY — FASTER)
# ============================================================

stack = np.stack([preds_eff, preds_swin, preds_convnext], axis=1)

preds_vote = []
for row in stack:
    counts = np.bincount(row)
    preds_vote.append(np.argmax(counts))

preds_vote = np.array(preds_vote)
acc_vote   = (preds_vote == all_labels).mean()

print(f"\nMajority Voting Accuracy: {acc_vote*100:.2f}%")

# ============================================================
# REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(all_labels, preds_vote, target_names=CLASS_NAMES))

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
accs = [
    (preds_eff == all_labels).mean(),
    (preds_swin == all_labels).mean(),
    (preds_convnext == all_labels).mean(),
    acc_vote
]

names = ['EffNet', 'Swin', 'ConvNeXt', 'Majority Vote']
colors = ['blue', 'blue', 'blue', 'green']

axes[0].bar(names, [a*100 for a in accs], color=colors)
axes[0].set_title("Accuracy Comparison")

# Confusion Matrix
cm = confusion_matrix(all_labels, preds_vote)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[1])

axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()

# ============================================================
# F1 SCORES
# ============================================================

f1_vote = f1_score(all_labels, preds_vote, average=None)

plt.figure(figsize=(8, 4))
plt.bar(CLASS_NAMES, f1_vote)
plt.title("F1 Score (Majority Voting)")
plt.ylim(0.7, 1.05)
plt.show()
