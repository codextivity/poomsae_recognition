# Poomsae Recognition Training Pipeline

This document is a living guide for training and evaluating the LSTM models in this repository.  
Update it whenever the pipeline, configs, or data layout changes.

Last updated: 2026-03-10

## 1. Scope

This pipeline covers:
1. Dataset preparation
2. Window dataset creation
3. Main LSTM training (22 classes)
4. Optional short-movement model training
5. Evaluation

## 2. Project Paths

Core paths are managed in `configs/paths.py`.

Important folders:
- `data/raw/videos`: raw input videos
- `data/raw/JSON`: annotation JSON files
- `data/processed/keypoints`: extracted keypoints
- `data/processed/windows`: training windows (`.npz`)
- `checkpoints`: model checkpoints
- `results`: evaluation outputs

## 3. Data Requirements

Each training sample needs:
1. A video file in `data/raw/videos`
2. A matching annotation file in `data/raw/JSON`

Expected annotation naming:
- `P052.mp4` -> `P052_annotations.json`

Expected annotation fields:
- `annotations`: list of movement entries
- Each entry includes at least `movement` and `startTime`

## 4. End-to-End Commands (Main 22-Class Model)

Run commands from project root:

```bash
python preprocessing/extract_keypoints.py
python preprocessing/create_windows.py
python training/train_main_22class_weighted.py
python evaluation/evaluate.py
```

What each step does:
1. `extract_keypoints.py`
- Uses RTMPose (Halpe26) to extract `(26, 3)` keypoints per frame
- Applies hip-centered and height-scaled normalization
- Saves `*_keypoints.pkl` into `data/processed/keypoints`

2. `create_windows.py`
- Builds sliding windows from keypoints + annotation timeline
- Uses class mapping for 22 classes (including split movements like `14_1/14_2`, `16_1/16_2`)
- Saves `*_windows.npz` into `data/processed/windows`

3. `train_main_22class_weighted.py`
- Loads windows via `training/dataset.py`
- Trains BiLSTM model from `models/lstm_classifier.py`
- Applies class weighting for imbalance
- Saves checkpoints to `checkpoints/lstm_taegeuk1_*.pth`

4. `evaluate.py`
- Loads best checkpoint
- Evaluates on test split
- Saves metrics and plots in `results`

## 5. Optional: Short-Movement Model

Use this when you want a specialized model for fast movements (`6_1`, `12_1`, `14_1`, `16_1`).

```bash
python preprocessing/create_windows_short.py
python training/train_short.py
```

Outputs:
- `data/processed/windows/short_movements_windows.npz`
- `checkpoints/lstm_short_best.pth`

## 6. Optional Step: `segment_movements.py`

`preprocessing/segment_movements.py` is optional for training.

Use it only if you need per-movement segmented artifacts for debugging or analysis.  
Main training does not require it because `create_windows.py` uses full keypoints + annotations directly.

## 7. Expected Outputs

Main model:
- `checkpoints/lstm_taegeuk1_best.pth`
- `checkpoints/training_history_best.json`
- `results/evaluation_results.json`
- `results/confusion_matrix.png`
- `results/per_movement_accuracy.png`

Short model:
- `checkpoints/lstm_short_best.pth`

## 8. Common Issues

1. `No window files found in data/processed/windows`
- Confirm `create_windows.py` finished successfully
- Confirm `configs/paths.py` points to the intended processed directory

2. RTMPose import error
- Install `rtmlib` and runtime dependencies

3. CUDA not available
- Scripts fall back to CPU, but training/inference will be slower

## 9. Files Involved

Config:
- `configs/paths.py`
- `configs/lstm_config.py`
- `configs/lstm_config_short.py`
- `configs/training_config.py`

Preprocessing:
- `preprocessing/extract_keypoints.py`
- `preprocessing/create_windows.py`
- `preprocessing/create_windows_short.py`
- `preprocessing/segment_movements.py` (optional)

Training:
- `training/dataset.py`
- `training/train_main_22class_weighted.py`
- `training/train_short.py`
- `models/lstm_classifier.py`

Evaluation:
- `evaluation/evaluate.py`

## 10. Update Policy

When changing pipeline behavior, update this README in the same commit:
1. Command order changes
2. Input/output path changes
3. Class mapping or label policy changes
4. Training/evaluation script changes
