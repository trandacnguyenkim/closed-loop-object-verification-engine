#!/bin/bash
#SBATCH --job-name=train-yolo-coco_10k
#SBATCH --partition=debug
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=00:45:00

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

# ── 1. Cache dirs ─────────────────────────────────────────────────────
export CLOVE_CACHE_DIR="/scratch/kt68/cache/clove"
export XDG_CACHE_HOME="$CLOVE_CACHE_DIR/xdg"
export HF_HOME="/scratch/kt68/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CLOVE_CACHE_DIR/torch"
export TMPDIR="$CLOVE_CACHE_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="/scratch/kt68/cache/pip"

# ── 3. Venv: create once, reuse on every subsequent job ───────────────
VENV_DIR="/scratch/kt68/venvs/clove2"
MIGC_DIR="/home/kt68/clove/MIGC"
CLOVE_DIR="/home/kt68/clove"

if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    echo "Setting up virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
    "$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$CLOVE_DIR/requirements.txt"

    touch "$VENV_DIR/.setup_complete"
    echo "Venv setup complete."
else
    echo "Reusing existing venv at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# ── 5. Sanity checks ──────────────────────────────────────────────────
echo "Python:         $(which python)"
echo "PyTorch:        $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── 6. Run ────────────────────────────────────────────────────────────
cd "$CLOVE_DIR"
srun python train_yolo.py \
      --data /scratch/kt68/training_datasets/coco_10k/dataset.yaml \
      --name coco_10k \
      --project /scratch/kt68/runs/yolo_baseline \
      --epochs 100 --batch 32 --validate \
      --workers 8 \

echo "Job finished at $(date)"