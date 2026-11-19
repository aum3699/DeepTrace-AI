
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ==========================================
# DEEPTRACE - EFFICIENTNET-B0 CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "real_fake_dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001
PATIENCE = 4
IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

print("\n====================================")
print("       DEEPTRACE MODEL TRAINING")
print("====================================")
print("Device:", DEVICE)
print("Architecture: EfficientNet-B0")
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# ==========================================
# IMAGE TRANSFORMATIONS
# ==========================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transforms
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

if train_dataset.classes != ["fake", "real"]:
    raise ValueError("Unexpected training class names.")

if val_dataset.classes != train_dataset.classes:
    raise ValueError("Validation classes do not match training.")

if test_dataset.classes != train_dataset.classes:
    raise ValueError("Test classes do not match training.")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ==========================================

print("\nLoading pretrained EfficientNet-B0...")

try:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
except Exception as error:
    raise RuntimeError(
        "Could not load pretrained EfficientNet-B0 weights. "
        "Check your internet connection and try again. "
        "Original error: " + str(error)
    )

# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace final classification layer
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    2
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ==========================================
# TRAINING AND VALIDATION
# ==========================================

best_val_accuracy = 0.0
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_model_path = MODEL_DIR / "deeptrace_efficientnet_b0.pth"


for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    # ---------------- TRAIN ----------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = correct / total


    # ---------------- VALIDATION ----------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save best validation checkpoint
    if val_accuracy > best_val_accuracy or (
        val_accuracy == best_val_accuracy
        and val_loss < best_val_loss
    ):

        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "image_size": IMAGE_SIZE,
            "architecture": "efficientnet_b0",
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy
        }, best_model_path)

        print("Best model checkpoint saved!")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ==========================================
# SAVE TRAINING HISTORY
# ==========================================

history_path = MODEL_DIR / "training_history.json"

with open(history_path, "w") as file:
    json.dump(history, file, indent=4)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n====================================")
print("       TRAINING COMPLETED")
print("====================================")

print("Best epoch:", best_epoch)
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print("Model saved:", best_model_path)
print("History saved:", history_path)

print("\nTesting dataset has NOT been used for training.")
print("Next step: Independent test evaluation.")
