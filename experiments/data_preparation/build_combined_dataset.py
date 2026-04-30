"""
build_combined_dataset.py — Samples COCO images, converts to YOLO format,
and merges with a pre-formatted synthetic YOLO dataset.

Usage:
    python build_combined_dataset.py \
        --coco_train  /path/to/coco/train2017 \
        --coco_val    /path/to/coco/val2017 \
        --synthetic   /path/to/synthetic_yolo_dataset \
        --n_train     9000 \
        --n_val       1000 \
        --output      ./combined_dataset

Expected COCO layout:
    <coco_train>/           ← raw images (.jpg)
    <coco_root>/annotations/instances_train2017.json
    <coco_root>/annotations/instances_val2017.json

Expected synthetic layout (standard YOLO):
    <synthetic>/
      images/train/
      labels/train/

Output layout:
    <output>/
      images/train/   images/val/
      labels/train/   labels/val/
      dataset.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Optional


# COCO-80 class names in canonical Ultralytics order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_NAME_TO_IDX = {name: i for i, name in enumerate(COCO_CLASSES)}


# ── COCO → YOLO conversion ────────────────────────────────────────────────────

def _find_annotations(coco_dir: Path, split: str) -> Path:
    """Walk upward from coco_dir to find annotations/instances_{split}.json."""
    for parent in [coco_dir] + list(coco_dir.parents):
        candidate = parent / "annotations" / f"instances_{split}.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find annotations/instances_{split}.json near {coco_dir}"
    )


def _convert_and_copy(image_ids: set[int], coco_dir: Path, ann_data: dict,
                      out_img_dir: Path, out_lbl_dir: Path) -> None:
    """Copy sampled images and write their YOLO labels."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    cat_to_yolo = {
        c["id"]: COCO_NAME_TO_IDX[c["name"]]
        for c in ann_data["categories"]
        if c["name"] in COCO_NAME_TO_IDX
    }

    anns_by_image: dict[int, list] = {}
    for ann in ann_data["annotations"]:
        if not ann.get("iscrowd", 0):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in ann_data["images"]:
        if img["id"] not in image_ids:
            continue
        w, h = float(img["width"]), float(img["height"])
        stem = Path(img["file_name"]).stem

        lines = []
        for ann in anns_by_image.get(img["id"], []):
            if ann["category_id"] not in cat_to_yolo:
                continue
            x, y, bw, bh = ann["bbox"]
            bw, bh = float(bw), float(bh)
            if bw <= 0 or bh <= 0:
                continue
            cx = max(0.0, min(1.0, (float(x) + bw / 2) / w))
            cy = max(0.0, min(1.0, (float(y) + bh / 2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            lines.append(f"{cat_to_yolo[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        src = coco_dir / img["file_name"]
        if not src.is_file():
            src = coco_dir / Path(img["file_name"]).name  # flat fallback
        if src.is_file():
            shutil.copy2(src, out_img_dir / src.name)
            (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")


# ── Synthetic copy ────────────────────────────────────────────────────────────

def _copy_synthetic(synthetic_dir: Path, out_img_dir: Path, out_lbl_dir: Path,
                    n_synthetic: Optional[int], rng: random.Random) -> int:
    """Copy up to `n_synthetic` images (all if None) from synthetic train into outputs.

    Returns the number of synthetic images copied.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs_dir = synthetic_dir / "images" / "train"
    labels_dir = synthetic_dir / "labels" / "train"
    if not imgs_dir.is_dir():
        return 0

    imgs = [p for p in imgs_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not imgs:
        return 0

    if n_synthetic is None or n_synthetic >= len(imgs):
        chosen = imgs
    else:
        chosen = rng.sample(imgs, n_synthetic)

    for img in chosen:
        shutil.copy2(img, out_img_dir / img.name)
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.is_file():
            shutil.copy2(lbl, out_lbl_dir / lbl.name)

    return len(chosen)


# ── dataset.yaml writer ───────────────────────────────────────────────────────

def _write_yaml(output_dir: Path) -> None:
    lines = [
        f"train: {(output_dir / 'images' / 'train').resolve()}",
        f"val:   {(output_dir / 'images' / 'val').resolve()}",
        f"nc: {len(COCO_CLASSES)}",
        "names:",
        *[f"  - {name}" for name in COCO_CLASSES],
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build(coco_train: Path, coco_val: Path, synthetic: Path,
          n_train: int, n_val: int, output: Path, n_synthetic: Optional[int] = None,
          seed: int = 42) -> None:

    rng = random.Random(seed)

    # Load COCO annotation JSONs
    train_ann = json.loads(_find_annotations(coco_train, "train2017").read_text())
    val_ann   = json.loads(_find_annotations(coco_val,   "val2017").read_text())

    # Sample image IDs
    all_train_ids = [img["id"] for img in train_ann["images"]]
    all_val_ids   = [img["id"] for img in val_ann["images"]]

    sampled_train = set(rng.sample(all_train_ids, min(n_train, len(all_train_ids))))
    sampled_val   = set(rng.sample(all_val_ids,   min(n_val,   len(all_val_ids))))

    print(f"COCO train sample : {len(sampled_train):,} images")
    print(f"COCO val sample   : {len(sampled_val):,} images")

    # Convert and copy COCO subsets
    _convert_and_copy(sampled_train, coco_train, train_ann,
                      output / "images" / "train", output / "labels" / "train")
    _convert_and_copy(sampled_val, coco_val, val_ann,
                      output / "images" / "val", output / "labels" / "val")

    # Merge synthetic (train only)
    n_syn = _copy_synthetic(synthetic, output / "images" / "train",
                            output / "labels" / "train", n_synthetic, rng)
    print(f"Synthetic images  : {n_syn:,} images")
    print(f"Combined train    : {len(sampled_train) + n_syn:,} images")

    _write_yaml(output)
    print(f"Dataset ready     → {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined COCO + synthetic YOLO dataset.")
    parser.add_argument("--coco_train", required=True, type=Path, help="Path to COCO train images dir.")
    parser.add_argument("--coco_val",   required=True, type=Path, help="Path to COCO val images dir.")
    parser.add_argument("--synthetic",  required=True, type=Path, help="Path to synthetic YOLO dataset root.")
    parser.add_argument("--n_train",    required=True, type=int,  help="Number of COCO train images to sample.")
    parser.add_argument("--n_val",      required=True, type=int,  help="Number of COCO val images to sample.")
    parser.add_argument("--output",     default=Path("./combined_dataset"), type=Path, help="Output directory.")
    parser.add_argument("--n_synthetic", default=None, type=int,
                        help="Number of synthetic images to include (default: all).")
    parser.add_argument("--seed",       default=42, type=int, help="Random seed (default: 42).")
    args = parser.parse_args()

    build(args.coco_train, args.coco_val, args.synthetic,
        args.n_train, args.n_val, args.output, args.n_synthetic, args.seed)


if __name__ == "__main__":
    main()
