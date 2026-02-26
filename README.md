# Stroke Image Classification (PyTorch Template)

Ready-to-run starter template for binary image classification:

- `normal` (label `0`)
- `stroke` (label `1`)

This project is built for small datasets (like ~500 images) using:

- transfer learning (`resnet18` or `efficientnet_b0`)
- light fine-tuning
- stratified K-fold cross-validation
- key medical-style metrics (AUC, sensitivity, specificity, F1)

---

## 1) Folder structure

Use this exact layout:

```text
/workspace
├── data
│   ├── normal
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   └── stroke
│       ├── stroke_001.jpg
│       ├── stroke_002.jpg
│       └── ...
├── outputs
├── requirements.txt
└── scripts
    └── train_cv.py
```

Supported image types: `.jpg .jpeg .png .bmp .tif .tiff .webp`

---

## 2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Run training (5-fold CV, recommended)

```bash
python scripts/train_cv.py \
  --data-dir data \
  --output-dir outputs \
  --model-name resnet18 \
  --image-size 224 \
  --batch-size 16 \
  --epochs 30 \
  --num-folds 5 \
  --freeze-epochs 3 \
  --lr-head 1e-3 \
  --lr-finetune 1e-4 \
  --amp
```

If you do not have GPU, add:

```bash
--cpu
```

---

## 4) Quick smoke test (fast check)

Run a shorter job first:

```bash
python scripts/train_cv.py \
  --data-dir data \
  --output-dir outputs_smoke \
  --epochs 2 \
  --num-folds 2 \
  --batch-size 8 \
  --cpu
```

---

## 5) Outputs

After training:

- Per-fold model checkpoints:
  - `outputs/fold_1/best_model.pt`
  - ...
- Per-fold training history:
  - `outputs/fold_1/history.csv`
- Per-fold best metrics:
  - `outputs/fold_1/best_metrics.json`
- CV aggregate table:
  - `outputs/cv_results.csv`
- CV summary (mean/std):
  - `outputs/cv_summary.json`

Metrics include:

- `auc`
- `recall_sensitivity` (important for missed stroke risk)
- `specificity`
- `f1`
- `accuracy`, `precision`

---

## 6) Notes for better medical performance

1. Keep classes balanced if possible.
2. Ensure labels are clean (label noise hurts more than model choice).
3. Use patient-level splits when multiple images come from same patient.
4. Review false negatives first.
5. Increase data over time; 500 images is a good prototype start, not final clinical quality.