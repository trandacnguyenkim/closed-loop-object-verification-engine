# Data Preparation — experiments/data_preparation

This folder contains scripts to prepare generated synthetic data for YOLO training.

Combine multiple synthetic dataset generation outputs (keep only successful, YOLO-compatible images and labels):

## Combine Multiple Synthetic Dataset Generations
```bash
python combine_synthetic_datasets.py --sources <list of directories> --output_dir <output directory> --val_split 0 --seed 42
```

What to expect:
- Input: one or more synthetic run directories containing scenario folders with `images/`, `labels/`, and `manifests/`.
- Selection: only scenarios with `success=true` in the manifest are kept.
- Per scenario: the latest image is selected, matching label files are merged, and labels are written in YOLO format.
- Output: a YOLO-ready dataset folder with `images/`, `labels/`, `dataset.yaml`, and `dataset_report.json`.
- Split behavior: with `--val_split 0`, all accepted samples go to `train/`; otherwise a reproducible train/val split is created.

## Build a Full Dataset for Training
```bash
python build_combined_dataset.py --coco_train /path/to/train --coco_val /path/to/val --synthetic /path/to/synth --n_train 9000 --n_val 5000 --n_synthetic 1000 --seed 42 --output ./combined_dataset
```

This command combines sampled COCO data with synthetic data into one training dataset output folder.

What to expect:
- Input: COCO train/val image folders with annotation JSONs, plus a synthetic YOLO dataset (`images/train` + `labels/train`).
- Sampling: randomly selects `n_train` and `n_val` from COCO, and up to `n_synthetic` from synthetic (all if omitted).
- Output: `images/train`, `images/val`, `labels/train`, `labels/val`, and `dataset.yaml` under `--output`.
- Split behavior: synthetic images are added to train only; val contains sampled COCO val images.
- Reproducibility: same `--seed` gives the same sampled subset.

Notes:
- `--sources` accepts one or more directories containing run outputs (each with scenario subfolders).
- `--val_split 0` places all accepted images into `images/train/` and `labels/train/`.
- The script writes `dataset.yaml` and `dataset_report.json` into the output directory.

Example:
```bash
python combine_synthetic_datasets.py --sources /scratch/user/clove_run1 /scratch/user/clove_run2 \
  --output_dir ./combined_synthetic --val_split 0 --seed 42
```

Example:
```bash
python build_combined_dataset.py --coco_train /datasets/coco/train2017 --coco_val /datasets/coco/val2017 --synthetic /scratch/qmt1/clove/combined_synthetic_dataset_042726 --n_train 9000 --n_val 5000 --n_synthetic 1000 --seed 42 --output ./combined_dataset
```

See `combine_synthetic_datasets.py` and `build_combined_dataset.py` for more options and behaviour.
