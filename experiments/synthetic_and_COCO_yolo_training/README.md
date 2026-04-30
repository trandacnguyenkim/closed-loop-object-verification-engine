# YOLO Training (COCO + Synthetic)

This folder contains:

- `train_yolo.py`: Python entry point for YOLO training.
- `train_yolo_*.slurm`: SLURM job scripts for predefined dataset mixes.

## 1) Run `train_yolo.py` directly

From the repo root:

```bash
python experiments/synthetic_and_COCO_yolo_training/train_yolo.py \
	--model yolov8s.yaml \
	--scratch \
	--data /scratch/$USER/clove/combined_dataset_8kCOCO_2ksynth/dataset.yaml \
	--epochs 100 \
	--imgsz 640 \
	--batch 32 \
	--workers 8 \
	--device 0 \
	--project runs/yolo_experiments \
	--name coco8000_syn2k \
	--seed 42 \
	--patience 15 \
	--validate \
	--comparison_file runs/yolo_experiments/experiment_comparison.json
```

Useful flags:

- `--scratch`: train from architecture YAML (random init).
- `--model`: YAML for scratch (`yolov8s.yaml`) or checkpoint for fine-tuning (`.pt`).
- `--validate`: run validation and store mAP metrics.
- `--comparison_file`: append run metrics into a shared JSON.

## 2) Run with SLURM scripts

Submit one of the predefined jobs:

```bash
sbatch experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_10k.slurm
sbatch experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_10kbaseline.slurm
sbatch experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_8k_synthetic_2k.slurm
sbatch experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_7-5k_synthetic_2-5k.slurm
sbatch "experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_9k_synthetic_1k copy 2.slurm"
```

Quick SLURM checks:

```bash
squeue -u $USER
tail -f logs/yolo_coco8k_2ksyn_<jobid>.out
tail -f logs/yolo_coco8k_2ksyn_<jobid>.err
```

Notes:

- Launch from repo root so relative paths resolve correctly.
- Each script defines dataset path, run name, and output log filenames.
- Training outputs are written under `/scratch/$USER/clove/runs/yolo_experiments/`.
