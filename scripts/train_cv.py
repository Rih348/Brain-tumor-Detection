#!/usr/bin/env python3
"""Train a stroke-vs-normal image classifier with stratified k-fold CV."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {"normal": 0, "stroke": 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stroke-vs-normal image classifier using stratified k-fold cross-validation."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root containing normal/ and stroke/.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to save models and metrics.")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate while backbone is frozen.")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="Learning rate after unfreezing backbone.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Number of initial epochs with frozen backbone.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples(data_dir: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class directory '{class_dir}' to exist. "
                "Create both data/normal and data/stroke."
            )
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((file_path, label))

    if not samples:
        raise RuntimeError(f"No images found under {data_dir}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")

    return samples


class StrokeDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[Path, int]], transform: transforms.Compose) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.records[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(model_name: str) -> Tuple[nn.Module, List[nn.Parameter], List[nn.Parameter]]:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        backbone_params = [param for name, param in model.named_parameters() if not name.startswith("fc.")]
        head_params = list(model.fc.parameters())
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        backbone_params = [param for name, param in model.named_parameters() if not name.startswith("classifier.")]
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, backbone_params, head_params


def set_requires_grad(params: Iterable[nn.Parameter], flag: bool) -> None:
    for param in params:
        param.requires_grad = flag


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        iterator = tqdm(loader, leave=False)
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images).squeeze(1)
                loss = criterion(logits, targets)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.detach().item() * batch_size
            total_count += batch_size
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    y_true = np.concatenate(all_targets)
    y_prob = sigmoid_numpy(np.concatenate(all_logits))
    metrics = compute_metrics(y_true, y_prob)
    return avg_loss, metrics


def save_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_one_fold(
    fold_idx: int,
    train_records: Sequence[Tuple[Path, int]],
    val_records: Sequence[Tuple[Path, int]],
    args: argparse.Namespace,
    device: torch.device,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Dict[str, float]:
    fold_dir = args.output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = StrokeDataset(train_records, train_transform)
    val_dataset = StrokeDataset(val_records, eval_transform)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model, backbone_params, head_params = build_model(args.model_name)
    model = model.to(device)
    set_requires_grad(backbone_params, False)
    set_requires_grad(head_params, True)

    optimizer = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    train_labels = np.array([label for _, label in train_records], dtype=np.int64)
    pos_count = int(train_labels.sum())
    neg_count = int(len(train_labels) - pos_count)
    pos_weight = float(neg_count / max(pos_count, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = -float("inf")
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    backbone_unfrozen = args.freeze_epochs <= 0

    for epoch in range(1, args.epochs + 1):
        if not backbone_unfrozen and epoch > args.freeze_epochs:
            set_requires_grad(backbone_params, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            backbone_unfrozen = True

        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=amp_enabled,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=amp_enabled,
        )

        score = val_metrics["auc"]
        if np.isnan(score):
            score = val_metrics["f1"]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auc": train_metrics["auc"],
            "val_auc": val_metrics["auc"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_recall_sensitivity": val_metrics["recall_sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[Fold {fold_idx}][Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_recall={val_metrics['recall_sensitivity']:.4f} "
            f"val_specificity={val_metrics['specificity']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": args.model_name,
                "image_size": args.image_size,
                "state_dict": model.state_dict(),
                "class_to_label": CLASS_TO_LABEL,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_val_metrics": best_metrics,
            }
            torch.save(checkpoint, fold_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[Fold {fold_idx}] Early stopping at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / "history.csv", index=False)

    best_metrics = dict(best_metrics)
    best_metrics.update(
        {
            "fold": float(fold_idx),
            "best_epoch": float(best_epoch),
            "train_samples": float(len(train_records)),
            "val_samples": float(len(val_records)),
            "pos_weight": float(pos_weight),
        }
    )
    save_json(best_metrics, fold_dir / "best_metrics.json")
    return best_metrics


def summarize_results(results_df: pd.DataFrame) -> Dict[str, object]:
    metric_columns = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
    ]
    summary: Dict[str, object] = {"num_folds": int(len(results_df))}
    for col in metric_columns:
        values = pd.to_numeric(results_df[col], errors="coerce").dropna()
        summary[col] = {
            "mean": float(values.mean()) if not values.empty else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = collect_samples(args.data_dir)
    labels = np.array([label for _, label in samples], dtype=np.int64)
    normal_count = int((labels == 0).sum())
    stroke_count = int((labels == 1).sum())
    print(f"Found {len(samples)} images -> normal: {normal_count}, stroke: {stroke_count}")

    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2")
    if min(normal_count, stroke_count) < args.num_folds:
        raise ValueError(
            f"Not enough samples per class for {args.num_folds} folds. "
            "Each class must have at least num_folds images."
        )

    train_transform, eval_transform = build_transforms(args.image_size)
    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    fold_results: List[Dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(samples)), labels), start=1):
        train_records = [samples[i] for i in train_idx]
        val_records = [samples[i] for i in val_idx]
        print(f"\n========== Fold {fold_idx}/{args.num_folds} ==========")
        fold_metrics = train_one_fold(
            fold_idx=fold_idx,
            train_records=train_records,
            val_records=val_records,
            args=args,
            device=device,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        fold_results.append(fold_metrics)

    results_df = pd.DataFrame(fold_results)
    results_path = args.output_dir / "cv_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = summarize_results(results_df)
    summary_path = args.output_dir / "cv_summary.json"
    save_json(summary, summary_path)

    print("\n===== Cross-validation completed =====")
    print(results_df[["fold", "auc", "recall_sensitivity", "specificity", "f1"]].to_string(index=False))
    print(f"\nSaved per-fold metrics to: {results_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()
