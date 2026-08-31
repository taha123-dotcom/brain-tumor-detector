"""
Unified Pipeline — Brain Tumor Classification System
=====================================================
End-to-end pipeline: Data Loading → Preprocessing → Feature Extraction → Modeling → Evaluation

Usage:
    python pipeline.py                    # Run full pipeline
    python pipeline.py --quick            # Quick mode (smaller dataset, fewer epochs)
    python pipeline.py --eval-only        # Evaluate existing model only
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "archive (1)" / "Brain_Cancer_4class"
OUTPUT_DIR = BASE_DIR / "pipeline_output"
IMG_SIZE = 224
CLASSES = ["brain_glioma", "brain_menin", "brain_tumor", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Normalization constants (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(dataset_dir, max_per_class=None):
    """Load images and labels from the dataset directory."""
    print("\n" + "=" * 60)
    print("STAGE 1: DATA LOADING")
    print("=" * 60)

    images = []
    labels = []
    class_counts = {}

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"  [WARN] Class directory not found: {class_dir}")
            continue

        files = sorted([f for f in class_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        if max_per_class:
            files = files[:max_per_class]

        class_counts[class_name] = len(files)
        for img_path in files:
            img = Image.open(img_path).convert("RGB")
            images.append(np.array(img))
            labels.append(CLASS_TO_IDX[class_name])

    print(f"  Loaded {len(images)} images across {len(class_counts)} classes:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    return images, labels, class_counts


def train_test_split(images, labels, test_ratio=0.2, seed=42):
    """Stratified train/test split."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels[indices] == cls]
        n_test = max(1, int(len(cls_idx) * test_ratio))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")
    return (images[train_idx], labels[train_idx],
            images[test_idx], labels[test_idx])


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(img, apply_denoise=True, apply_clahe=True):
    """Apply preprocessing pipeline to a single image."""
    result = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    if apply_denoise:
        result = cv2.GaussianBlur(result, (3, 3), 0)

    if apply_clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


def normalize_image(img):
    """Normalize to [0,1] then apply ImageNet normalization."""
    arr = img.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr


def augment_image(img):
    """Random augmentation: horizontal flip and rotation."""
    if np.random.random() > 0.5:
        img = np.flip(img, axis=1).copy()
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def preprocess_batch(images, train=True, apply_denoise=True, apply_clahe=True):
    """Preprocess an entire batch of images."""
    print("\n" + "=" * 60)
    print("STAGE 2: IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"  Techniques: {'denoise + CLAHE + augment' if train else 'denoise + CLAHE'}")

    processed = []
    for i, img in enumerate(images):
        result = preprocess_image(img, apply_denoise, apply_clahe)
        if train:
            result = augment_image(result)
        processed.append(result)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")

    processed = np.array(processed, dtype=np.uint8)
    print(f"  Preprocessing complete: {processed.shape}")
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def extract_hog_features(images):
    """Extract Histogram of Oriented Gradients features."""
    print("  Extracting HOG features...")
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor(
            _winSize=(IMG_SIZE // 4, IMG_SIZE // 4),
            _blockSize=(IMG_SIZE // 4, IMG_SIZE // 4),
            _blockStride=(IMG_SIZE // 8, IMG_SIZE // 8),
            _cellSize=(IMG_SIZE // 8, IMG_SIZE // 8),
            _nbins=9
        )
        descriptor = hog.compute(gray)
        hog_features.append(descriptor.flatten() if descriptor is not None else np.zeros(324))
    return np.array(hog_features)


def extract_color_histograms(images, bins=64):
    """Extract color histogram features per channel."""
    print("  Extracting color histograms...")
    hist_features = []
    for img in images:
        hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()
        hist = np.concatenate([hist_r, hist_g, hist_b])
        hist = hist / (hist.sum() + 1e-7)
        hist_features.append(hist)
    return np.array(hist_features)


def extract_edge_features(images):
    """Extract edge-based features (Canny edge density + contour count)."""
    print("  Extracting edge features...")
    edge_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges > 0) / (IMG_SIZE * IMG_SIZE)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_contours = min(len(contours), 100) / 100.0
        moments = cv2.moments(edges)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        edge_features.append(np.concatenate([[density, n_contours], hu]))
    return np.array(edge_features)


def extract_features(images, use_cnn=True):
    """Extract combined features from images."""
    print("\n" + "=" * 60)
    print("STAGE 3: FEATURE EXTRACTION")
    print("=" * 60)

    hog_feats = extract_hog_features(images)
    hist_feats = extract_color_histograms(images)
    edge_feats = extract_edge_features(images)

    print(f"  HOG features:        {hog_feats.shape[1]} dims")
    print(f"  Color hist features: {hist_feats.shape[1]} dims")
    print(f"  Edge features:       {edge_feats.shape[1]} dims")

    cnn_feats = None
    if use_cnn:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            print("  Extracting CNN features (ResNet18)...")
            resnet = models.resnet18(pretrained=True)
            resnet.fc = torch.nn.Identity()
            resnet.eval()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            cnn_feats = []
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                tensors = torch.stack([transform(img) for img in batch])
                with torch.no_grad():
                    features = resnet(tensors)
                cnn_feats.append(features.numpy())
                if (i + batch_size) % 512 == 0:
                    print(f"    CNN features: {i + batch_size}/{len(images)}...")

            cnn_feats = np.concatenate(cnn_feats, axis=0)
            print(f"  CNN features:        {cnn_feats.shape[1]} dims")
        except ImportError:
            print("  [WARN] PyTorch not available, skipping CNN features")
            use_cnn = False

    if use_cnn and cnn_feats is not None:
        combined = np.concatenate([cnn_feats, hog_feats, hist_feats, edge_feats], axis=1)
    else:
        combined = np.concatenate([hog_feats, hist_feats, edge_feats], axis=1)

    print(f"  Combined features:   {combined.shape[1]} dims")
    return combined, cnn_feats, hog_feats, hist_feats, edge_feats


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train_cnn_model(train_images, train_labels, test_images, test_labels, epochs=15, quick=False):
    """Train a CNN model (BrainTumorCNN)."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [WARN] PyTorch not available, skipping CNN training")
        return None, None

    print("  Training CNN (BrainTumorCNN)...")

    class BrainTumorCNN(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    def to_tensor_batch(images):
        arr = images.astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = arr.transpose(0, 3, 1, 2)
        return torch.from_numpy(arr)

    X_train = to_tensor_batch(train_images)
    y_train = torch.from_numpy(train_labels)
    X_test = to_tensor_batch(test_images)
    y_test = torch.from_numpy(test_labels)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = BrainTumorCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0.0
    actual_epochs = epochs
    if quick:
        actual_epochs = min(epochs, 5)
        print(f"    Quick mode: {actual_epochs} epochs")

    for epoch in range(actual_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        avg_loss = running_loss / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(batch_y).sum().item()
                test_total += batch_y.size(0)
        test_acc = test_correct / test_total

        scheduler.step(1 - test_acc)
        print(f"    Epoch {epoch + 1}/{actual_epochs}: "
              f"loss={avg_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"  CNN best test accuracy: {best_acc:.4f}")
    return model, best_acc


def train_sklearn_models(X_train, y_train, X_test, y_test):
    """Train SVM and Random Forest on extracted features."""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n  Training SVM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    t0 = time.time()
    svm.fit(X_train_scaled, y_train)
    svm_time = time.time() - t0
    svm_acc = svm.score(X_test_scaled, y_test)
    print(f"    SVM accuracy: {svm_acc:.4f} (trained in {svm_time:.1f}s)")

    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_acc = rf.score(X_test, y_test)
    print(f"    RF accuracy:  {rf_acc:.4f} (trained in {rf_time:.1f}s)")

    return {
        "svm": {"model": svm, "accuracy": svm_acc, "scaler": scaler},
        "rf": {"model": rf, "accuracy": rf_acc},
    }


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5: EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, class_names, model_name):
    """Compute and display comprehensive evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, jaccard_score
    )

    print(f"\n  --- {model_name} ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    IoU:       {iou:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "iou": float(iou),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Generate and save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel="True Label", xlabel="Predicted Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = max(max(row) for row in cm) / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i][j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_acc"], label="Train", linewidth=2)
    ax1.plot(history["test_acc"], label="Test", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"], label="Train Loss", linewidth=2, color="red")
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing model accuracies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(models, accs, color=colors[:len(models)], edgecolor="black", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Brain Tumor Classification")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6: EXPORT & SAVE
# ──────────────────────────────────────────────────────────────────────────────
def save_results(results, output_dir):
    """Save all results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for model_name, metrics in results.items():
        serializable[model_name] = {
            k: v for k, v in metrics.items()
            if k != "confusion_matrix" or isinstance(v, list)
        }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {output_dir / 'evaluation_results.json'}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer samples and epochs")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing model only")
    parser.add_argument("--epochs", type=int, default=15, help="CNN training epochs")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max images per class")
    args = parser.parse_args()

    if args.quick:
        args.max_per_class = args.max_per_class or 200
        args.epochs = min(args.epochs, 5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BRAIN TUMOR CLASSIFICATION — UNIFIED PIPELINE")
    print("=" * 60)
    t_start = time.time()

    # ── Stage 1: Data Loading ──
    images, labels, class_counts = load_dataset(DATASET_DIR, args.max_per_class)
    if len(images) == 0:
        print("  [ERROR] No images found. Check dataset path.")
        sys.exit(1)

    train_imgs, train_labels, test_imgs, test_labels = train_test_split(images, labels)

    # ── Stage 2: Preprocessing ──
    train_processed = preprocess_batch(train_imgs, train=True)
    test_processed = preprocess_batch(test_imgs, train=False)

    # ── Stage 3: Feature Extraction ──
    train_feats, train_cnn, train_hog, train_hist, train_edge = extract_features(
        train_processed, use_cnn=True
    )
    test_feats, test_cnn, test_hog, test_hist, test_edge = extract_features(
        test_processed, use_cnn=True
    )

    all_results = {}

    # ── Stage 4a: CNN Training ──
    print("\n" + "=" * 60)
    print("STAGE 4: MODEL TRAINING")
    print("=" * 60)

    cnn_model, cnn_acc = train_cnn_model(
        train_processed, train_labels,
        test_processed, test_labels,
        epochs=args.epochs, quick=args.quick,
    )

    if cnn_model is not None:
        import torch
        cnn_model.eval()
        X_test_tensor = torch.from_numpy(
            test_processed.astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        X_test_tensor = (X_test_tensor - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)

        with torch.no_grad():
            cnn_logits = cnn_model(X_test_tensor)
            cnn_preds = cnn_logits.argmax(dim=1).numpy()

        cnn_results = evaluate_model(test_labels, cnn_preds, CLASSES, "CNN (BrainTumorCNN)")
        all_results["CNN"] = cnn_results

        # Save CNN model
        torch.save(cnn_model.state_dict(), OUTPUT_DIR / "brain_tumor_cnn.pth")
        print(f"  Saved: {OUTPUT_DIR / 'brain_tumor_cnn.pth'}")

    # ── Stage 4b: SVM & Random Forest ──
    if train_cnn is not None:
        sklearn_results = train_sklearn_models(train_cnn, train_labels, test_cnn, test_labels)
    else:
        sklearn_results = train_sklearn_models(train_feats, train_labels, test_feats, test_labels)

    for name, info in sklearn_results.items():
        model = info["model"]
        scaler = info.get("scaler")

        if scaler:
            X_test_input = scaler.transform(test_feats)
        else:
            X_test_input = test_feats

        preds = model.predict(X_test_input)
        results = evaluate_model(test_labels, preds, CLASSES, name.upper())
        all_results[name.upper()] = results

    # ── Stage 5: Visualization ──
    print("\n" + "=" * 60)
    print("STAGE 5: GENERATING VISUALIZATIONS")
    print("=" * 60)

    for model_name, metrics in all_results.items():
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, CLASSES, f"Confusion Matrix — {model_name}",
                              OUTPUT_DIR / f"confusion_matrix_{model_name.lower()}.png")

    plot_model_comparison(all_results, OUTPUT_DIR / "model_comparison.png")

    # ── Stage 6: Save ──
    print("\n" + "=" * 60)
    print("STAGE 6: SAVING RESULTS")
    print("=" * 60)

    save_results(all_results, OUTPUT_DIR)

    summary = {
        "dataset": {cls: cnt for cls, cnt in class_counts.items()},
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "models": {name: {"accuracy": m["accuracy"], "f1_score": m["f1_score"]}
                   for name, m in all_results.items()},
        "execution_time_sec": round(time.time() - t_start, 1),
    }
    with open(OUTPUT_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'pipeline_summary.json'}")

    # ── Final Summary ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()
    print("  Model Comparison:")
    print("  " + "-" * 45)
    for name, m in all_results.items():
        print(f"  {name:20s} acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}  iou={m['iou']:.4f}")
    print("  " + "-" * 45)


if __name__ == "__main__":
    main()
