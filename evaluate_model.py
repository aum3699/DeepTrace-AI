
from pathlib import Path
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "real_fake_dataset"
MODEL_PATH = BASE_DIR / "models" / "deeptrace_efficientnet_b0.pth"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8

print("\n====================================")
print("    DEEPTRACE MODEL EVALUATION")
print("====================================")


# ==========================================
# CHECK MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# This is a locally created checkpoint from
# your own training script. Only load trusted files.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

print("Model:", checkpoint["architecture"])
print("Classes:", classes)
print("Best epoch:", checkpoint["best_epoch"])


# ==========================================
# LOAD TEST DATA
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=test_transform
)

if test_dataset.classes != classes:
    raise ValueError("Test classes do not match model classes.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# ==========================================
# PREDICTIONS
# ==========================================

y_true = []
y_pred = []

print("\nEvaluating test images...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.cpu().tolist())


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes)))
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("       TEST EVALUATION RESULTS")
print("====================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# ==========================================
# SAVE RESULTS
# ==========================================

results = {
    "model": "EfficientNet-B0",
    "test_images": len(test_dataset),
    "classes": classes,
    "accuracy": float(accuracy),
    "precision_macro": float(precision),
    "recall_macro": float(recall),
    "f1_macro": float(f1),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
}

report_path = REPORT_DIR / "evaluation_report.json"

with open(report_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation report saved:", report_path)

print("\n====================================")
print("    EVALUATION COMPLETED")
print("====================================")
