"""
Train a YOLO model on COCO (or a mixed COCO + synthetic dataset).

Three standard experiments are supported via --data:

  Experiment A — 100% COCO baseline
    python train_yolo.py \
        --data training_datasets/coco_100/dataset.yaml \
        --name coco_100

  Experiment B — 90% COCO + 10% synthetic
    python train_yolo.py \
        --data training_datasets/coco90_syn10/dataset.yaml \
        --name coco90_syn10

  Experiment C — 80% COCO + 20% synthetic
    python train_yolo.py \
        --data training_datasets/coco80_syn20/dataset.yaml \
        --name coco80_syn20

Run prepare_training_datasets.py first to build the dataset YAML configs.
After all three runs finish, use compare_runs.py to print a mAP comparison.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Train a baseline YOLO model on COCO."
	)
	parser.add_argument(
		"--scratch",
		action="store_true",
		help=(
			"Train from scratch with randomly initialized weights. "
			"Use a model architecture YAML such as yolov8n.yaml instead of a .pt checkpoint."
		),
	)
	parser.add_argument(
		"--model",
		type=str,
		default="yolov8n.pt",
		help="Ultralytics model checkpoint to start from.",
	)
	parser.add_argument(
		"--data",
		type=str,
		default="coco.yaml",
		help="Dataset YAML (default: COCO config from Ultralytics).",
	)
	parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
	parser.add_argument(
		"--imgsz", type=int, default=640, help="Input image size (square)."
	)
	parser.add_argument("--batch", type=int, default=16, help="Batch size.")
	parser.add_argument(
		"--device",
		type=str,
		default="0",
		help='Compute device, e.g. "0", "0,1", or "cpu".',
	)
	parser.add_argument(
		"--workers", type=int, default=8, help="Dataloader worker processes."
	)
	parser.add_argument(
		"--project",
		type=str,
		default="runs/yolo_baseline",
		help="Output project directory.",
	)
	parser.add_argument(
		"--name",
		type=str,
		default="coco_baseline",
		help="Run name inside --project.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--patience",
		type=int,
		default=50,
		help="Early-stopping patience in epochs.",
	)
	parser.add_argument(
		"--cache",
		action="store_true",
		help="Cache dataset in RAM/disk if available.",
	)
	parser.add_argument(
		"--validate",
		action="store_true",
		help="Run validation on the COCO val split after training and save mAP metrics.",
	)
	parser.add_argument(
		"--comparison_file",
		type=str,
		default="runs/experiment_comparison.json",
		help="Path to a shared JSON file where mAP results from all three experiments are accumulated.",
	)
	parser.add_argument(
		"--export",
		action="store_true",
		help="Export the best model after training.",
	)
	parser.add_argument(
		"--export-format",
		type=str,
		default="onnx",
		help="Export format for --export (onnx, torchscript, etc.).",
	)
	return parser


def _resolve_save_dir(train_results: Any, fallback_project: str, run_name: str) -> Path:
	if hasattr(train_results, "save_dir"):
		return Path(train_results.save_dir)
	return Path(fallback_project) / run_name


def main() -> None:
	args = build_arg_parser().parse_args()

	try:
		from ultralytics import YOLO
	except ImportError as exc:
		raise SystemExit(
			"Ultralytics is not installed. Install dependencies with:\n"
			"  pip install -r requirements.txt"
		) from exc

	model = YOLO(args.model)
	pretrained = not args.scratch

	if args.scratch and args.model.endswith(".pt"):
		raise SystemExit(
			"--scratch requires a model architecture YAML, for example yolov8n.yaml. "
			"A .pt checkpoint would still start from pretrained weights."
		)

	mode = "scratch" if args.scratch else "pretrained"
	print(f"[YOLO] Starting {mode} COCO training...")
	print(f"[YOLO] model={args.model} data={args.data}")

	train_results = model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		project=args.project,
		name=args.name,
		seed=args.seed,
		patience=args.patience,
		cache=args.cache,
		pretrained=pretrained,
	)

	save_dir = _resolve_save_dir(train_results, args.project, args.name)
	summary = {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"model": args.model,
		"data": args.data,
		"epochs": args.epochs,
		"imgsz": args.imgsz,
		"batch": args.batch,
		"device": args.device,
		"workers": args.workers,
		"seed": args.seed,
		"save_dir": str(save_dir),
	}

	if args.validate:
		print("[YOLO] Running validation on COCO val split...")
		val_results = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

		# Extract mAP metrics from the results object
		val_metrics: dict = {}
		if hasattr(val_results, "results_dict"):
			val_metrics = {
				k: float(v)
				for k, v in val_results.results_dict.items()
				if isinstance(v, (int, float))
			}
		elif hasattr(val_results, "box"):
			# Ultralytics >= 8.1 stores metrics under .box
			box = val_results.box
			val_metrics = {
				"mAP50":    float(getattr(box, "map50",  0.0)),
				"mAP50-95": float(getattr(box, "map",    0.0)),
				"precision":float(getattr(box, "mp",     0.0)),
				"recall":   float(getattr(box, "mr",     0.0)),
			}

		summary["validation"] = val_metrics
		print("[YOLO] Validation metrics:")
		for k, v in val_metrics.items():
			print(f"  {k:<20}: {v:.4f}")

		# ── Accumulate into shared comparison file ────────────────────
		comparison_path = Path(args.comparison_file)
		comparison_path.parent.mkdir(parents=True, exist_ok=True)
		comparison: dict = {}
		if comparison_path.exists():
			try:
				comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
			except json.JSONDecodeError:
				comparison = {}

		comparison[args.name] = {
			"timestamp":  datetime.now().isoformat(timespec="seconds"),
			"data":       args.data,
			"epochs":     args.epochs,
			"model":      args.model,
			"save_dir":   str(save_dir),
			"metrics":    val_metrics,
		}
		comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
		print(f"[YOLO] Comparison table updated → {comparison_path}")

	if args.export:
		print(f"[YOLO] Exporting model to {args.export_format}...")
		export_path = model.export(format=args.export_format)
		summary["export_path"] = str(export_path)

	save_dir.mkdir(parents=True, exist_ok=True)
	summary_path = save_dir / "run_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print("[YOLO] Training complete.")
	print(f"[YOLO] Run directory: {save_dir}")
	print(f"[YOLO] Summary: {summary_path}")


if __name__ == "__main__":
	main()
