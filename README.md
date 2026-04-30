# CLOVE: Closed-Loop Object Verification Engine

A synthetic data generation pipeline that uses closed-loop refinement to create high-quality, annotated images aligned with COCO object detection standards. The system iteratively improves generated images using an LLM-based planner, a diffusion model generator, and a zero-shot object critic validator.

## Overview

**CLOVE** generates synthetic images for 250 diverse scenarios covering all 80 COCO object categories. The core loop refines images through multiple iterations:

1. **LLM Planner** → generates detailed image layout instructions
2. **Image Generator** → creates image from layout using MIGC (diffusion model)
3. **Critic Validator** → verifies objects are present and correctly placed
4. **Refinement Loop** → repeats until layout is accepted or max iterations reached

Each scenario can be run in two modes:
- **Standard mode**: standard prompts
- **Hard-negative mode**: prompts designed to challenge the system

## Prerequisites

### System Requirements
- Python 3.9+
- CUDA 12.1 compatible GPU (highly recommended for diffusion inference)

### Clone MIGC Repository
The image generator depends on MIGC (Modified Image Generation Canvas). Clone and install it first:

```bash
git clone https://github.com/limuloo/MIGC
cd MIGC
pip install -e .
cd ..  # return to clove repo
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment (Optional)
Copy `.env.example` to `.env` and update as needed:

```bash
cp .env.example .env
```

Key variables:
- `LOCAL_LLM_MODEL`: Which Hugging Face model to use for the planner (default: `Qwen/Qwen2.5-14B-Instruct`)
- `HF_TOKEN`: Your Hugging Face token (required for gated models)

## Project Structure

```
.
├── src/                          # Core engine modules
│   ├── loop_manager.py           # Orchestrates the closed-loop refinement
│   ├── llm_planner.py            # LLM-based layout planner
│   ├── image_generator.py        # Diffusion-based image generation (MIGC wrapper)
│   ├── critic_validator.py       # Zero-shot object detection validator
│   ├── local_llm.py              # Local LLM interface (Qwen, etc.)
│   └── utils.py                  # Utility functions (COCO annotations, etc.)
│
├── images_generation/            # Image generation scripts & scenarios
│   ├── generate_images.py        # Main batch generation script
│   └── SCENARIOS.py              # 250 COCO-aligned test scenarios
│
├── experiments/                  # Experiment workflows
│   ├── data_preperation/         # Prepare synthetic data for training
│   └── synthetic_and_COCO_yolo_training/  # YOLO training on synthetic + COCO data
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── .env.example                  # Environment variable template
└── README.md                     # This file
```

## Image Generation Workflow

### Quick Start: Generate Images

Change the `USER` to your NetID to save the images in your scratch directory.

**Smoke test** (1 scenario, standard mode, 1 iteration):
```bash
cd images_generation
python generate_images.py --smoke
```

**Full batch** (all 250 scenarios, standard mode only):
```bash
python generate_images.py --standard-only
```

**Full batch with hard negatives** (all 250 scenarios × 2 modes = 500 runs):
```bash
python generate_images.py
```

To generate more image variations without editing `SCENARIOS.py`, run with a different seed each time:

```bash
python generate_images.py --seed 123
```

### Output Structure

Results are saved to `/scratch/{USER}/clove_output/{BATCH_TIMESTAMP}/`:

```
clove_output/
├── 042926-143022/              # batch_timestamp directory
│   ├── batch_summary.json      # Overall batch statistics & per-scenario results
│   ├── coco_annotations.json   # COCO-format annotations for all accepted images
│   └── {scenario}_{mode}/      # Per-scenario run directory
│       ├── manifest.json       # Full iteration log with layouts & critic feedback
│       ├── final_image.png     # Accepted image (if success=true)
│       └── iteration_0/
│           ├── layout.txt      # Detailed layout instructions
│           ├── generated.png   # Raw generated image
│           └── validation.json # Critic verdict & detected objects
```

### Understanding the Loop

For each scenario:
1. **Iteration 0**: Planner generates initial layout → Generator creates image → Critic validates
2. **Iteration N**: If objects missing or wrongly placed:
   - Planner refines layout based on critic feedback
   - Generator updates image
   - Critic re-validates
3. **Success**: When critic approves all required objects (success=true) or max_iterations reached

### Command-Line Options

```bash
python generate_images.py [OPTIONS]

OPTIONS:
  --smoke              Run smoke test: 1 scenario, standard mode, max_iterations=1
  --standard-only      Run all scenarios in standard mode only (no hard negatives)
   --seed INT           Initial random seed for generation (default: 91)
                       Default behavior: all scenarios × both modes = 500 runs
```

## Experiments Workflow

### Data Preparation
Prepare generated synthetic data for training:

See [experiments/data_preparation/README.md](experiments/data_preparation/README.md) for detailed instructions and runnable examples, including:
- combining multiple synthetic outputs (`combine_synthetic_datasets.py`)
- building a full COCO+synthetic training dataset (`build_combined_dataset.py`)

```bash
cd experiments/data_preparation/
# Follow instructions in that subfolder
```

This step typically:
- Filters accepted images from the batch
- Converts COCO annotations to training format
- Splits into train/val sets
- Applies any data augmentation

### YOLO Training

Train object detection models on synthetic data, optionally combined with COCO:

```bash
cd experiments/synthetic_and_COCO_yolo_training/
# Follow instructions in that subfolder
```

This step typically:
- Trains YOLOv8 or similar detector
- Evaluates on COCO validation set
- Compares synthetic-only vs. synthetic+COCO training

## Core Modules

### `loop_manager.py`
Orchestrates the closed-loop refinement. For each scenario:
- Maintains iteration count and manifests
- Calls planner → generator → critic in sequence
- Decides when to accept or refine

**Key function**: `run(user_prompt, img_name, hard_negative, run_timestamp, initial_seed)`

### `llm_planner.py`
Uses local LLM (Qwen by default) to generate detailed layout instructions.

**Input**: User prompt + critic feedback (if iteration > 0)  
**Output**: Structured layout with spatial instructions, object attributes, etc.

### `image_generator.py`
Wraps MIGC diffusion model to generate images from layout.

**Input**: Detailed layout instructions  
**Output**: PIL Image

### `critic_validator.py`
Zero-shot object detection (YOLO or similar) to verify:
- All required objects are present
- Objects are in reasonable spatial relationships
- No spurious objects

**Input**: Generated image  
**Output**: Dictionary with `approved` (bool), `detected_objects` (list), `feedback` (str for refinement)

### `utils.py`
Helper functions including COCO annotation saving.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LOCAL_LLM_MODEL` | HF model ID for planner | `Qwen/Qwen2.5-14B-Instruct` |
| `HF_TOKEN` | HF authentication (optional) | — |

To use an offline/local model:
```bash
export LOCAL_LLM_MODEL="/path/to/local/model"
```

## Troubleshooting

### Out of Memory
If CUDA runs out of memory during diffusion:
- Reduce batch size in `LoopManager` config
- Use a smaller LLM model
- Reduce `max_iterations`

### MIGC Import Errors
Ensure MIGC is installed in editable mode:
```bash
cd /path/to/MIGC
pip install -e .
```

### Missing SCENARIOS
If `from SCENARIOS import SCENARIOS` fails, run from the `images_generation/` folder:
```bash
cd images_generation
python generate_images.py
```

Or use a relative import:
```python
from .SCENARIOS import SCENARIOS
```

## Datasets
Our datasets are saved in `/scratch/qmt1/clove/combined_synthetic_dataset_042726`
