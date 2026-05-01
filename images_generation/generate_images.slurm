#!/bin/bash
#SBATCH --job-name=clove-loop
#SBATCH --partition=debug
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=00:30:00

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

# ── 0. Scratch root ──────────────────────────────────────────────────
# Use provided SCRATCH_USER or fall back to the current user
SCRATCH_USER="${SCRATCH_USER:-${USER:-$(whoami)}}"
SCRATCH_ROOT="/scratch/$SCRATCH_USER"

# ── 1. Cache dirs ─────────────────────────────────────────────────────
export CLOVE_CACHE_DIR="$SCRATCH_ROOT/cache/clove"
export XDG_CACHE_HOME="$CLOVE_CACHE_DIR/xdg"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CLOVE_CACHE_DIR/torch"
export TMPDIR="$CLOVE_CACHE_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="$SCRATCH_ROOT/cache/pip"

mkdir -p "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# ── 2. Model paths ────────────────────────────────────────────────────
export MIGC_CKPT_PATH="$SCRATCH_ROOT/pretrained_weights/MIGC_SD14.ckpt"
export MIGC_SAFETENSOR="$SCRATCH_ROOT/pretrained_weights/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
export MIGC_YAML="$SCRATCH_ROOT/pretrained_weights/v1-inference.yaml"

# ── 3. Venv: create once, reuse on every subsequent job ───────────────
VENV_DIR="$SCRATCH_ROOT/venvs/clove2"
CLOVE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MIGC_DIR="${MIGC_DIR:-$CLOVE_DIR/MIGC}"

if [ ! -d "$MIGC_DIR" ]; then
    echo "ERROR: MIGC repository not found at '$MIGC_DIR'."
    echo "Set MIGC_DIR to a valid checkout before submitting this job."
    exit 1
fi

if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    echo "Setting up virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir -r "$CLOVE_DIR/requirements.txt"
    pip install --no-cache-dir -e "$MIGC_DIR"

    cd "$MIGC_DIR" && python setup.py develop && cd "$CLOVE_DIR"

    pip install --no-cache-dir gdown   # install gdown into venv here

    python -c "from migc.migc_pipeline import StableDiffusionMIGCPipeline; print('[Setup] MIGC OK')"
    if [ $? -ne 0 ]; then
        echo "[Setup] MIGC import failed. Aborting."
        exit 1
    fi

    touch "$VENV_DIR/.setup_complete"
    echo "Venv setup complete."
else
    echo "Reusing existing venv at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# ── 4. Download weights (once, uses venv's gdown) ─────────────────────
WEIGHTS_DIR="$SCRATCH_ROOT/pretrained_weights"
WEIGHTS_READY="$WEIGHTS_DIR/.weights_complete"

if [ ! -f "$WEIGHTS_READY" ]; then
    echo "Downloading model weights..."
    mkdir -p "$WEIGHTS_DIR/clip/text_encoder"

    gdown --id 1v5ik-94qlfKuCx-Cv1EfEkxNBygtsz0T \
          -O "$WEIGHTS_DIR/MIGC_SD14.ckpt"

    python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id   = 'SG161222/Realistic_Vision_V5.1_noVAE',
    filename  = 'Realistic_Vision_V5.1_fp16-no-ema.safetensors',
    local_dir = '$WEIGHTS_DIR',
    token     = os.getenv('HF_TOKEN'),
)
print('Realistic Vision downloaded.')
"

    gdown --id 1Z_BFepTXMbe-cib7Lla5A224XXE1mBcS \
          -O "$WEIGHTS_DIR/clip/text_encoder/pytorch_model.bin"

    wget -q "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml" \
         -O "$WEIGHTS_DIR/v1-inference.yaml"

    if [ -f "$WEIGHTS_DIR/MIGC_SD14.ckpt" ] && \
       [ -f "$WEIGHTS_DIR/Realistic_Vision_V5.1_fp16-no-ema.safetensors" ] && \
       [ -f "$WEIGHTS_DIR/clip/text_encoder/pytorch_model.bin" ] && \
       [ -f "$WEIGHTS_DIR/v1-inference.yaml" ]; then
        touch "$WEIGHTS_READY"
        echo "Weights downloaded and verified."
    else
        echo "ERROR: One or more weights failed to download. Aborting."
        exit 1
    fi
else
    echo "Weights already present — skipping download."
fi

# ── 5. Sanity checks ──────────────────────────────────────────────────
echo "Python:         $(which python)"
echo "PyTorch:        $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "MIGC:           $(python -c 'from migc.migc_pipeline import StableDiffusionMIGCPipeline; print("OK")')"
echo "MIGC_CKPT:      $MIGC_CKPT_PATH"
echo "MIGC_SAFETENSOR:$MIGC_SAFETENSOR"

# ── 6. Run ────────────────────────────────────────────────────────────
cd "$CLOVE_DIR"
srun python "$CLOVE_DIR/images_generation/generate_images.py" --smoke

echo "Job finished at $(date)"