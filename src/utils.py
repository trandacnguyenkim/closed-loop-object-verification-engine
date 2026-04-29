"""
utils.py — Shared utilities for the synthetic data pipeline.

Contents
────────
HardNegativeGenerator
    Appends difficulty factors to a normal scene prompt to produce a
    deliberately challenging variant. Difficulty is added WITHOUT
    rewriting the original sentence, so all original objects are
    preserved — guaranteeing the hard negative is a strict superset
    of the standard prompt's content. This avoids the failure mode
    where rephrasing accidentally drops objects and makes the hard
    negative easier than the standard.

    Difficulty factors:
      • low_contrast      – object colour blends with background
      • unusual_scale     – object much smaller or larger than expected
      • heavy_occlusion   – primary object partially blocked
      • adverse_lighting  – harsh shadows or strong backlighting
      • unusual_viewpoint – top-down or extreme wide-angle

save_run_manifest
    Persists the full iteration log as JSON for experiment tracking.

make_run_dir
    Ensures the dataset folder structure exists before any file is written.
"""

import os
import json
import random
import datetime
from dotenv import load_dotenv
from src.local_llm import LocalQwenChat

load_dotenv()


# ══════════════════════════════════════════════════════════════════════
# Hard Negative Generator
# ══════════════════════════════════════════════════════════════════════

# All supported difficulty axes
DIFFICULTY_FACTORS = [
    "low_contrast",
    "unusual_scale",
    "heavy_occlusion",
    "adverse_lighting",
    "unusual_viewpoint",
]

# Concrete description fragments for each factor.
# Multiple variants are sampled per factor so hard negatives stay diverse.
_FACTOR_FRAGMENTS = {
    "low_contrast": [
        "with the subjects in colours that blend into the background",
        "in low-contrast tones where foreground and background nearly match",
        "with muted desaturated colours making subjects hard to distinguish",
    ],
    "unusual_scale": [
        "with the main subjects appearing unusually small in the frame",
        "with the main subjects filling almost the entire frame at unusually large scale",
        "with extreme size disparity between the subjects",
    ],
    "heavy_occlusion": [
        "with the main subjects partially hidden behind other objects",
        "with significant occlusion blocking parts of each subject",
        "with the subjects half-obscured by foreground elements",
    ],
    "adverse_lighting": [
        "with strong backlighting casting the subjects into silhouette",
        "in harsh directional lighting creating deep shadows across the subjects",
        "in dim low-light conditions with minimal illumination",
    ],
    "unusual_viewpoint": [
        "viewed from a sharp overhead bird's-eye angle",
        "viewed from an extreme low ground-level angle looking up",
        "viewed through a wide-angle fisheye perspective",
    ],
}


class HardNegativeGenerator:
    """
    Appends difficulty descriptors to a scene prompt to make it visually
    harder for downstream object detectors WITHOUT changing the underlying
    objects in the scene.

    This deterministic templated approach is preferred over LLM-based
    rewriting because LLM rewrites frequently drop or substitute objects,
    accidentally making hard negatives easier than their standard pair.
    Templated appending guarantees the hard negative is the standard scene
    plus added difficulty — a strict superset.
    """

    def __init__(self, model: str = None, num_factors: int = None):
        """
        Args:
            model:       Kept for API compatibility with the previous version.
                         No LLM is called; the parameter is unused.
            num_factors: How many difficulty factors to append. If None, 1 or 2
                         are chosen at random each call.
        """
        self.model       = model or os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-14B-Instruct")
        self.num_factors = num_factors

    def transform(self, prompt: str) -> tuple[str, dict]:
        """
        Append difficulty descriptors to *prompt*.

        Args:
            prompt: Original scene description (the "standard" prompt).

        Returns:
            (hard_negative_prompt, info_dict)
            hard_negative_prompt is the original prompt with difficulty
            descriptors appended after a comma. info_dict contains the
            applied difficulty factors and the original prompt for
            traceability.
        """
        # Pick how many factors to apply (1 or 2 by default)
        k = self.num_factors if self.num_factors else random.randint(1, 2)
        chosen_factors = random.sample(DIFFICULTY_FACTORS, k)

        # Sample one concrete fragment per factor and join
        fragments = [
            random.choice(_FACTOR_FRAGMENTS[factor])
            for factor in chosen_factors
        ]

        # Append to the original prompt without modifying its words.
        # This guarantees all original objects are preserved.
        hard_prompt = f"{prompt.rstrip('. ')}, {', '.join(fragments)}"

        info = {
            "original_prompt":  prompt,
            "chosen_factors":   chosen_factors,
            "applied_factors":  chosen_factors,
            "fragments":        fragments,
        }
        return hard_prompt, info


# ══════════════════════════════════════════════════════════════════════
# Filesystem helpers
# ══════════════════════════════════════════════════════════════════════

def make_run_dir(output_dir: str = "./dataset") -> str:
    """
    Ensure a run directory and its subdirectories (images, labels, manifests) exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ("images", "labels", "manifests"):
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir


def save_run_manifest(manifest: dict, img_name: str, output_dir: str = "./dataset") -> str:
    """
    Persists the full run manifest as a pretty-printed JSON file in the
    manifests/ subdirectory.
    """
    make_run_dir(output_dir)
    manifest_path = os.path.join(output_dir, "manifests", f"{img_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Utils] Manifest saved → {manifest_path}")
    return manifest_path


# ══════════════════════════════════════════════════════════════════════
# Prompt utilities
# ══════════════════════════════════════════════════════════════════════

def sanitise_img_name(name: str) -> str:
    """
    Strips characters that are unsafe in filenames and replaces spaces
    with underscores.
    """
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in name)
    return safe.strip().replace(" ", "_")[:64]


# ══════════════════════════════════════════════════════════════════════
# COCO annotation builder
# ══════════════════════════════════════════════════════════════════════

def save_coco_annotations(
    records: list[dict],
    output_path: str,
    dataset_root: str = ".",
) -> str:
    """
    Build and save a COCO-format annotation JSON from accepted generation records.
    """
    from PIL import Image as PILImage

    all_class_names = sorted({
        obj["class_name"]
        for record in records
        for obj in record["layout"]
    })
    category_id_map = {name: i + 1 for i, name in enumerate(all_class_names)}
    categories = [
        {"id": cat_id, "name": name, "supercategory": "object"}
        for name, cat_id in category_id_map.items()
    ]

    images_list: list[dict] = []
    annotations_list: list[dict] = []
    ann_id = 1

    for img_id, record in enumerate(records, start=1):
        image_path = record["image_path"]
        layout = record["layout"]

        with PILImage.open(image_path) as pil_img:
            img_w, img_h = pil_img.size

        abs_img = os.path.abspath(image_path)
        abs_root = os.path.abspath(dataset_root)
        try:
            file_name = os.path.relpath(abs_img, abs_root)
        except ValueError:
            file_name = os.path.basename(image_path)

        images_list.append({
            "id":        img_id,
            "file_name": file_name,
            "width":     img_w,
            "height":    img_h,
        })

        for obj in layout:
            x_min, y_min, x_max, y_max = obj["box"]
            x = x_min * img_w
            y = y_min * img_h
            w = (x_max - x_min) * img_w
            h = (y_max - y_min) * img_h

            annotations_list.append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  category_id_map[obj["class_name"]],
                "bbox":         [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area":         round(w * h, 2),
                "segmentation": [],
                "iscrowd":      0,
            })
            ann_id += 1

    coco = {
        "info": {
            "description":  "CLOVE synthetic dataset — COCO annotations",
            "date_created": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        "licenses":    [],
        "images":      images_list,
        "annotations": annotations_list,
        "categories":  categories,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"[Utils] COCO annotations saved → {output_path}  "
        f"({len(images_list)} images, {len(annotations_list)} annotations, "
        f"{len(categories)} categories)"
    )
    return os.path.abspath(output_path)