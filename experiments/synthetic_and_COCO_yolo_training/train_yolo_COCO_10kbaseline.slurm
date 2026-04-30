#!/bin/bash
#SBATCH --job-name=yolo-coco10k-baseline
#SBATCH --partition=debug
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/yolo_coco10k_baseline_%j.out
#SBATCH --error=logs/yolo_coco10k_baseline_%j.err

# ======================================================================
# CLOVE -- YOLO Training: 10,000 COCO baseline (coco10k_baseline)
#
# Dataset  : /scratch/<USER>/training_datasets/coco_10k/
# Val set  : 100% real COCO val images (inside dataset.yaml)
# Model    : YOLOv8n from scratch (yolov8n.yaml)
# Output   : runs/yolo_experiments/coco10k_baseline/
#
# Submit:
#   sbatch experiments/synthetic_and_COCO_yolo_training/train_yolo_COCO_10kbaseline.slurm
# ======================================================================

echo "=============================================="
echo " CLOVE YOLO Training -- coco10k_baseline"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo "=============================================="
nvidia-smi

module purge

# ── 1. Paths ───────────────────────────────────────────────────────────
USER="${USER:-qmt1}"
VENV_DIR="/scratch/$USER/venvs/clove2"

# Dataset / COCO paths (adjust if your COCO files live elsewhere)
COCO_ROOT="/scratch/$USER/datasets/coco"
COCO_TRAIN="$COCO_ROOT/train2017"
COCO_VAL="$COCO_ROOT/val2017"

DATASET_DIR="/scratch/$USER/training_datasets/coco_10k"
DATASET_YAML="$DATASET_DIR/dataset.yaml"

RUNS_DIR="runs/yolo_experiments"
COMPARISON_FILE="$RUNS_DIR/experiment_comparison.json"

# ── 2. Cache dirs ──────────────────────────────────────────────────────
export CLOVE_CACHE_DIR="/scratch/$USER/cache/clove"
export XDG_CACHE_HOME="$CLOVE_CACHE_DIR/xdg"
export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CLOVE_CACHE_DIR/torch"
export TMPDIR="$CLOVE_CACHE_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR" \
         "$RUNS_DIR" "$CLOVE_DIR/logs"

# ── 3. Activate venv ──────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    echo "       Run the generation job first to build the environment."
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "Python : $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# Make sure ultralytics is present (no-op if already installed)
python -c "import ultralytics" 2>/dev/null || {
    echo "[Setup] Installing ultralytics..."
    pip install --quiet ultralytics
}

# ── 4. Pre-flight checks ───────────────────────────────────────────────
if [ ! -f "$DATASET_YAML" ]; then
    echo "[Prep] Dataset YAML not found: $DATASET_YAML"
    echo "[Prep] Building COCO 10k combined dataset (0 synthetic)..."
    mkdir -p "$DATASET_DIR"

    python experiments/data_preparation/build_combined_dataset.py \
        --coco_train "$COCO_TRAIN" \
        --coco_val   "$COCO_VAL" \
        --synthetic  "$RUNS_DIR/no_synthetic" \
        --n_train    10000 \
        --n_val      5000 \
        --output     "$DATASET_DIR" \
        --n_synthetic 0 || {
        echo "ERROR: build_combined_dataset.py failed"
        exit 1
    }
fi

echo ""
echo "Dataset YAML: $DATASET_YAML"
echo "--- YAML contents ---"
cat "$DATASET_YAML"
echo "---------------------"

# Count images in train and val splits as listed in the YAML
TRAIN_TXT=$(grep "^train:" "$DATASET_YAML" | awk '{print $2}')
VAL_TXT=$(grep "^val:" "$DATASET_YAML" | awk '{print $2}')

if [ -f "$TRAIN_TXT" ]; then
    N_TRAIN=$(wc -l < "$TRAIN_TXT")
    echo "Train images : $N_TRAIN  (from $TRAIN_TXT)"
elif [ -d "$TRAIN_TXT" ]; then
    N_TRAIN=$(find "$TRAIN_TXT" -name "*.jpg" -o -name "*.png" | wc -l)
    echo "Train images : $N_TRAIN  (from $TRAIN_TXT)"
fi

if [ -f "$VAL_TXT" ]; then
    N_VAL=$(wc -l < "$VAL_TXT")
    echo "Val images   : $N_VAL  (from $VAL_TXT)"
elif [ -d "$VAL_TXT" ]; then
    N_VAL=$(find "$VAL_TXT" -name "*.jpg" -o -name "*.png" | wc -l)
    echo "Val images   : $N_VAL  (from $VAL_TXT)"
fi

# ── 5. Training hyperparameters ────────────────────────────────────────
MODEL="yolov8s.yaml"
EPOCHS=5
IMGSZ=640
BATCH=32
WORKERS=8
SEED=42
PATIENCE=15
RUN_NAME="coco10k_baseline"

echo ""
echo "=============================="
echo " Training hyperparameters"
echo "=============================="
echo "  Model    : $MODEL"
echo "  Epochs   : $EPOCHS"
echo "  Img size : $IMGSZ"
echo "  Batch    : $BATCH"
echo "  Workers  : $WORKERS"
echo "  Seed     : $SEED"
echo "  Patience : $PATIENCE"
echo "  Run name : $RUN_NAME"
echo ""

# ── 6. Train ───────────────────────────────────────────────────────────

echo "=============================="
echo " Starting training"
echo "=============================="

srun python experiments/synthetic_and_COCO_yolo_training/train_yolo.py \
    --model           "$MODEL"           \
    --scratch                            \
    --data            "$DATASET_YAML"    \
    --epochs          $EPOCHS            \
    --imgsz           $IMGSZ             \
    --batch           $BATCH             \
    --workers         $WORKERS           \
    --device          0                  \
    --project         "$RUNS_DIR"        \
    --name            "$RUN_NAME"        \
    --seed            $SEED              \
    --patience        $PATIENCE          \
    --validate                           \
    --comparison_file "$COMPARISON_FILE"

TRAIN_EXIT=$?

# ── 7. Report ──────────────────────────────────────────────────────────
echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=============================="
    echo " Training completed successfully"
    echo "=============================="
    echo "Results saved in : $RUNS_DIR/$RUN_NAME"
    echo "Comparison file  : $COMPARISON_FILE"
    if [ -f "$COMPARISON_FILE" ]; then
        echo ""
        echo "--- mAP summary ---"
        python -c "
import json
with open('$COMPARISON_FILE') as f:
    data = json.load(f)
for run, info in data.items():
    metrics = info.get('metrics', {})
    map50   = metrics.get('mAP50',    metrics.get('metrics/mAP50(B)',    'N/A'))
    map5095 = metrics.get('mAP50-95', metrics.get('metrics/mAP50-95(B)', 'N/A'))
    print(f'  {run:<25}  mAP50={map50}  mAP50-95={map5095}')
"
    fi
else
    echo "=============================="
    echo " ERROR: Training failed (exit code $TRAIN_EXIT)"
    echo "=============================="
    echo "Check logs: logs/yolo_coco10k_baseline_${SLURM_JOB_ID}.err"
    exit $TRAIN_EXIT
fi

echo ""
echo "Job finished at $(date)"
