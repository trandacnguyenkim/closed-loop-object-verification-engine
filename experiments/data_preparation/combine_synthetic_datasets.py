"""
combine_synthetic_datasets.py — Merges multiple clove_output directories into
a single YOLO-ready dataset.

Expected input layout (each source directory):
    <source_dir>/
      <scenario_name>/
        images/    — rendered images (e.g. scene_iter1.jpg, scene_iter2.jpg, ...)
        labels/    — YOLO .txt label files, one per image, matching stem
        manifests/ — <scenario_name>_manifest.json

For every scenario found the script:
  1. Locates the scenario's manifest (<scenario>/manifests/<name>_manifest.json).
  2. Skips the scenario if ``success`` is False.
  3. Picks the lexicographically latest image from <scenario>/images/.
  4. Copies that image and its matching label into
     <output_dir>/images/train/ and <output_dir>/labels/train/.
  5. Writes a dataset.yaml consumable directly by Ultralytics train_yolo.py.
  6. Writes a dataset_report.json summarising acceptance / skip statistics.

Output layout (inside --output_dir):
    <output_dir>/
      images/
        train/  <latest_image>.jpg ...
        val/    <latest_image>.jpg ...  (only when --val_split > 0)
      labels/
        train/  <latest_image>.txt ...
        val/    <latest_image>.txt ...  (only when --val_split > 0)
      dataset.yaml
      dataset_report.json

Usage:
    python combine_synthetic_datasets.py \\
        --sources /scratch/user/clove_run1 /scratch/user/clove_run2 \\
        --output_dir ./combined_synthetic \\
        --val_split 0.1 \\
        --seed 42

    # Or pass source directories via a text file (one path per line):
    python combine_synthetic_datasets.py \\
        --sources_file ./my_run_dirs.txt \\
        --output_dir ./combined_synthetic
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import shutil
from pathlib import Path


# ── COCO-80 class names in canonical Ultralytics order (index 0–79) ──────────
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

COCO_CLASS_TO_ID = {name: idx for idx, name in enumerate(COCO_CLASSES)}


def _resolve_coco_class_id(class_token: str, scenario_name: str) -> int | None:
    """Map a class token (possibly descriptive text) to a COCO class id."""
    token = class_token.strip().lower().replace("_", " ")
    if token in COCO_CLASS_TO_ID:
        return COCO_CLASS_TO_ID[token]

    # Match class names as whole words inside descriptive phrases.
    for class_name in sorted(COCO_CLASSES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(class_name)}\b", token):
            return COCO_CLASS_TO_ID[class_name]

    # Fallback: infer class from scenario naming convention, e.g. "umbrella_*".
    scenario = scenario_name.strip().lower()
    for class_name in sorted(COCO_CLASSES, key=len, reverse=True):
        slug = class_name.replace(" ", "_")
        if scenario == slug or scenario.startswith(slug + "_"):
            return COCO_CLASS_TO_ID[class_name]

    return None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Per-scenario processing ───────────────────────────────────────────────────

def _process_scenario(
    scenario_dir: Path,
    images_out: Path,
    labels_out: Path,
    seen_stems: set[str],
) -> dict:
    """
    Process one scenario directory.  Returns a status dict:
        {
          "scenario":   str,         # absolute path to the scenario dir
          "manifest":   str | None,  # absolute path to the manifest used
          "status":     str,         # "accepted" | "skipped_failed" | "skipped_error"
          "reason":     str | None,  # why it was skipped (if applicable)
          "dest_image": str | None,  # final destination image path
        }
    """
    def skip(status: str, reason: str, manifest: Path | None = None) -> dict:
        return {
            "scenario":   str(scenario_dir.resolve()),
            "manifest":   str(manifest.resolve()) if manifest else None,
            "status":     status,
            "reason":     reason,
            "dest_image": None,
        }

    # 1. Find and load the manifest (use lexicographic last = most recent).
    manifest_files = sorted((scenario_dir / "manifests").glob("*_manifest.json"))
    if not manifest_files:
        return skip("skipped_error", "No *_manifest.json found in manifests/")

    manifest_path = manifest_files[-1]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return skip("skipped_error", f"Could not read manifest: {exc}", manifest_path)

    # 2. Check success flag.
    if not manifest.get("success", False):
        return skip("skipped_failed", "success=False in manifest", manifest_path)

    # 3. Pick the lexicographically latest image (relies on zero-padded names).
    images = sorted(
        p for p in (scenario_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        return skip("skipped_error", "No image files found in images/", manifest_path)

    src_image = images[-1]

    # 4. Locate all label files for this iteration.
    # Images are named …_iter<N>_final.jpg; labels are named …_iter<N>_0.txt,
    # …_iter<N>_1.txt, etc. Strip "_final" to recover the iteration stem.
    iter_stem = src_image.stem.removesuffix("_final")
    label_files = sorted((scenario_dir / "labels").glob(f"{iter_stem}_*.txt"))
    if not label_files:
        return skip("skipped_error", f"No label files found for iteration: {iter_stem}", manifest_path)

    # 5. Build a collision-safe destination stem.
    dest_stem, counter = src_image.stem, 1
    while dest_stem in seen_stems:
        dest_stem = f"{src_image.stem}_{counter}"
        counter += 1
    seen_stems.add(dest_stem)

    # 6. Merge per-object label lines and convert class names to numeric IDs.
    lines: list[str] = []
    for f in label_files:
        for raw_line in f.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                return skip("skipped_error", f"Malformed label line in {f.name}: {line}", manifest_path)

            # Support class names with spaces by treating the final 4 tokens as bbox.
            class_token = " ".join(parts[:-4]).lower().replace("_", " ")
            bbox_tokens = parts[-4:]

            class_id = _resolve_coco_class_id(class_token, scenario_dir.name)
            if class_id is not None:
                lines.append(f"{class_id} {' '.join(bbox_tokens)}")
                continue

            # Allow already-numeric YOLO labels to pass through.
            if len(parts) == 5:
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    return skip("skipped_error", f"Unknown class label '{parts[0]}' in {f.name}", manifest_path)

                if not (0 <= class_id < len(COCO_CLASSES)):
                    return skip("skipped_error", f"Class id out of range in {f.name}: {class_id}", manifest_path)

                lines.append(f"{class_id} {' '.join(parts[1:])}")
                continue

            return skip("skipped_error", f"Unknown class label '{class_token}' in {f.name}", manifest_path)

    dest_image = images_out / (dest_stem + src_image.suffix)
    shutil.copy2(src_image, dest_image)
    (labels_out / (dest_stem + ".txt")).write_text(("\n".join(lines) + "\n") if lines else "")

    return {
        "scenario":   str(scenario_dir.resolve()),
        "manifest":   str(manifest_path.resolve()),
        "status":     "accepted",
        "reason":     None,
        "dest_image": str(dest_image.resolve()),
    }


# ── dataset.yaml writer ───────────────────────────────────────────────────────

def _write_dataset_yaml(
    yaml_path: Path,
    train_images_dir: Path,
    val_images_dir: Path | None,
    names: list[str],
) -> None:
    """Write a minimal Ultralytics-compatible dataset YAML."""
    lines = [
        "# Combined synthetic CLOVE dataset — auto-generated by combine_synthetic_datasets.py",
        f"# Created: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"train: {train_images_dir.resolve()}",
    ]
    if val_images_dir:
        lines.append(f"val:   {val_images_dir.resolve()}")
    lines += ["", f"nc: {len(names)}", "names:"]
    lines += [f"  - {name}" for name in names]

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"[Combine] dataset.yaml written → {yaml_path}")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def combine_datasets(
    source_dirs: list[str],
    output_dir: str,
    val_split: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Scan all *source_dirs* for scenario sub-directories, keep only those
    whose manifest reports ``success=True``, and write a unified YOLO dataset
    under *output_dir*.

    Args:
        source_dirs: List of clove_output root directories to scan.
        output_dir:  Destination root for the combined dataset.
        val_split:   Fraction of accepted images to reserve for validation
                     (0.0 = no validation split, everything goes to train).
        seed:        RNG seed for reproducible train/val splitting.

    Returns:
        A summary dict with acceptance statistics (also written as JSON).
    """
    out = Path(output_dir)
    train_img_dir = out / "images" / "train"
    train_lbl_dir = out / "labels" / "train"
    val_img_dir   = out / "images" / "val"
    val_lbl_dir   = out / "labels" / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Discover scenario directories ────────────────────────
    print("\n[Combine] Discovering scenarios…")
    all_scenarios: list[tuple[str, Path]] = []
    for src in source_dirs:
        src_path = Path(src).resolve()
        if not src_path.is_dir():
            print(f"[Combine] WARNING: source not found, skipping: {src_path}")
            continue
        found = sorted(
            d for d in src_path.iterdir()
            if d.is_dir() and all((d / sub).is_dir() for sub in ("images", "labels", "manifests"))
        )
        print(f"[Combine]   {src_path}: {len(found)} scenario(s)")
        all_scenarios.extend((str(src_path), s) for s in found)

    total_scenarios = len(all_scenarios)
    print(f"[Combine] Total scenarios discovered: {total_scenarios}")
    if total_scenarios == 0:
        print("[Combine] WARNING: No scenarios found. Nothing to combine.")
        return {"total_scenarios": 0, "accepted": 0, "skipped_failed": 0, "skipped_error": 0}

    # ── Phase 2: Process every scenario ──────────────────────────────
    print("\n[Combine] Processing scenarios…")
    seen_stems: set[str] = set()
    accepted_entries: list[dict] = []
    per_source_stats: dict[str, dict] = {}
    n_skipped_failed = n_skipped_error = 0

    for src_dir, scenario in all_scenarios:
        entry = _process_scenario(scenario, train_img_dir, train_lbl_dir, seen_stems)
        stats = per_source_stats.setdefault(src_dir, {"total": 0, "accepted": 0, "skipped_failed": 0, "skipped_error": 0})
        stats["total"] += 1
        if entry["status"] == "accepted":
            accepted_entries.append(entry)
            stats["accepted"] += 1
        elif entry["status"] == "skipped_failed":
            n_skipped_failed += 1
            stats["skipped_failed"] += 1
        else:
            n_skipped_error += 1
            stats["skipped_error"] += 1
            print(f"[Combine]   WARN {scenario.name}: {entry['reason']}")

    n_accepted = len(accepted_entries)
    print(f"[Combine] Accepted: {n_accepted:,}  |  Skipped (failed): {n_skipped_failed:,}  |  Skipped (errors): {n_skipped_error:,}")

    # ── Phase 3: Optional train / val split ───────────────────────────
    val_stems: set[str] = set()
    val_entries:   list[dict] = []
    train_entries: list[dict] = accepted_entries

    if n_accepted > 0 and val_split > 0.0:
        n_val = min(max(1, round(n_accepted * val_split)), n_accepted - 1)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        shuffled = list(accepted_entries)
        random.Random(seed).shuffle(shuffled)
        val_entries, train_entries = shuffled[:n_val], shuffled[n_val:]
        val_stems = {Path(e["dest_image"]).stem for e in val_entries}

        for entry in val_entries:
            src_img = Path(entry["dest_image"])
            shutil.move(str(src_img), str(val_img_dir / src_img.name))
            entry["dest_image"] = str((val_img_dir / src_img.name).resolve())
            src_lbl = train_lbl_dir / (src_img.stem + ".txt")
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(val_lbl_dir / src_lbl.name))

        print(f"[Combine] Train: {len(train_entries):,}  |  Val: {len(val_entries):,}  (seed={seed})")
    else:
        print(f"[Combine] No val split — all {n_accepted:,} images in train/")

    # ── Phase 4: Write dataset.yaml ───────────────────────────────────
    _write_dataset_yaml(
        yaml_path        = out / "dataset.yaml",
        train_images_dir = train_img_dir,
        val_images_dir   = val_img_dir if val_entries else None,
        names            = COCO_CLASSES,
    )

    # ── Phase 5: Write dataset_report.json ───────────────────────────
    report = {
        "created_at":      datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed":            seed,
        "val_split":       val_split,
        "output_dir":      str(out.resolve()),
        "source_dirs":     [str(Path(s).resolve()) for s in source_dirs],
        "total_scenarios": total_scenarios,
        "accepted":        n_accepted,
        "train_images":    len(train_entries),
        "val_images":      len(val_entries),
        "skipped_failed":  n_skipped_failed,
        "skipped_error":   n_skipped_error,
        "per_source_stats": per_source_stats,
        "accepted_images": [
            {
                "scenario":   e["scenario"],
                "manifest":   e["manifest"],
                "dest_image": e["dest_image"],
                "split":      "val" if Path(e["dest_image"]).stem in val_stems else "train",
            }
            for e in accepted_entries
        ],
    }
    report_path = out / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[Combine] dataset_report.json written → {report_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n[Combine] Done.")
    print(f"[Combine] Output dir      : {out.resolve()}")
    print(f"[Combine] Total scenarios : {total_scenarios:,}")
    print(f"[Combine] Accepted        : {n_accepted:,}  (train: {len(train_entries):,}, val: {len(val_entries):,})")
    print(f"[Combine] Skipped         : failed={n_skipped_failed:,}, errors={n_skipped_error:,}")
    print(f"\n[Combine] To launch training:")
    print(f"  python train_yolo.py --data {out.resolve() / 'dataset.yaml'} --name combined_synthetic")
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple clove_output directories into a single "
            "YOLO-ready dataset, keeping only scenarios that passed generation validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--sources", nargs="+", metavar="DIR",
                           help="One or more clove_output root directories to scan.")
    src_group.add_argument("--sources_file", metavar="FILE",
                           help="Text file listing source directories, one per line. "
                                "Lines starting with '#' and blank lines are ignored.")
    parser.add_argument("--output_dir", default="./combined_synthetic",
                        help="Destination directory for the combined dataset (default: ./combined_synthetic).")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Fraction of accepted images to reserve for validation (default: 0.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/val split (default: 42).")
    return parser.parse_args()


def _load_sources_file(path: str) -> list[str]:
    """Read source directories from a text file, one per line."""
    return [
        line.strip() for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> None:
    args = _parse_args()

    if args.sources_file:
        if not Path(args.sources_file).is_file():
            raise SystemExit(f"[Combine] ERROR: --sources_file not found: {args.sources_file}")
        source_dirs = _load_sources_file(args.sources_file)
        if not source_dirs:
            raise SystemExit(f"[Combine] ERROR: No directories found in {args.sources_file}")
    else:
        source_dirs = args.sources

    if not 0.0 <= args.val_split < 1.0:
        raise SystemExit("[Combine] ERROR: --val_split must be in [0.0, 1.0).")

    print("[Combine] Source directories:")
    for d in source_dirs:
        print(f"  {d}")
    print(f"[Combine] Output directory : {args.output_dir}")
    print(f"[Combine] Val split        : {args.val_split:.0%}")
    print(f"[Combine] Seed             : {args.seed}")

    combine_datasets(
        source_dirs=source_dirs,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
