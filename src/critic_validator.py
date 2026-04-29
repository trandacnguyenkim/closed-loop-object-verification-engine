"""
critic_validator.py — Zero-shot visual validation using GroundingDINO.

Verifies that generated images actually contain the requested objects in the
correct locations by running zero-shot object detection and comparing
predicted bounding boxes against expected ones via IoU.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


COCO_80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}


# ══════════════════════════════════════════════════════════════════════
# IoU + geometry helpers
# ══════════════════════════════════════════════════════════════════════

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Both boxes are in [x_min, y_min, x_max, y_max] format (normalised 0-1).
    """
    x_min_inter = max(box_a[0], box_b[0])
    y_min_inter = max(box_a[1], box_b[1])
    x_max_inter = min(box_a[2], box_b[2])
    y_max_inter = min(box_a[3], box_b[3])

    inter_width  = max(0.0, x_max_inter - x_min_inter)
    inter_height = max(0.0, y_max_inter - y_min_inter)
    area_overlap = inter_width * inter_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area_union = area_a + area_b - area_overlap

    if area_union <= 0:
        return 0.0
    return area_overlap / area_union


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_size(box: list[float]) -> tuple[float, float]:
    return (box[2] - box[0], box[3] - box[1])


def _format_box(box: list[float] | None) -> list[float] | None:
    if box is None:
        return None
    return [round(v, 2) for v in box]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _describe_region(box: list[float]) -> str:
    cx, cy = _box_center(box)
    if cx < 1 / 3:
        horiz = "left"
    elif cx > 2 / 3:
        horiz = "right"
    else:
        horiz = "center"
    if cy < 1 / 3:
        vert = "top"
    elif cy > 2 / 3:
        vert = "bottom"
    else:
        vert = "middle"
    if horiz == "center" and vert == "middle":
        return "center"
    return f"{vert}-{horiz}"


def _movement_hint(expected_box: list[float], pred_box: list[float]) -> str:
    exp_cx, exp_cy = _box_center(expected_box)
    pred_cx, pred_cy = _box_center(pred_box)
    exp_w, exp_h = _box_size(expected_box)
    pred_w, pred_h = _box_size(pred_box)

    directions: list[str] = []
    if exp_cx - pred_cx > 0.08:
        directions.append("move it right")
    elif pred_cx - exp_cx > 0.08:
        directions.append("move it left")
    if exp_cy - pred_cy > 0.08:
        directions.append("move it down")
    elif pred_cy - exp_cy > 0.08:
        directions.append("move it up")

    size_hints: list[str] = []
    if pred_w > 0 and exp_w / pred_w > 1.25:
        size_hints.append("make it wider")
    elif exp_w > 0 and pred_w / exp_w > 1.25:
        size_hints.append("make it narrower")
    if pred_h > 0 and exp_h / pred_h > 1.25:
        size_hints.append("make it taller")
    elif exp_h > 0 and pred_h / exp_h > 1.25:
        size_hints.append("make it shorter")

    hints = directions + size_hints
    if not hints:
        return "adjust its placement and scale to better match the requested box"
    if len(hints) == 1:
        return hints[0]
    return ", ".join(hints[:-1]) + f", and {hints[-1]}"


def _greedy_match_boxes(
    expected_boxes: list[list[float]],
    pred_boxes: list[list[float]],
) -> dict[int, tuple[int, float]]:
    """Greedily match expected boxes to unique predictions by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    for exp_idx, exp_box in enumerate(expected_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            candidates.append((compute_iou(exp_box, pred_box), exp_idx, pred_idx))

    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_expected: set[int] = set()
    matched_pred: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}

    for iou, exp_idx, pred_idx in candidates:
        if exp_idx in matched_expected or pred_idx in matched_pred:
            continue
        matches[exp_idx] = (pred_idx, iou)
        matched_expected.add(exp_idx)
        matched_pred.add(pred_idx)

    return matches


# ══════════════════════════════════════════════════════════════════════
# CriticValidator — class interface for the LoopManager
# ══════════════════════════════════════════════════════════════════════

class CriticValidator:
    """
    Wraps GroundingDINO zero-shot detection to validate every entity
    produced by the image generator against the planned layout.

    Compatible with loop_manager.py:
        critic = CriticValidator(threshold=0.5)
        passed, feedback = critic.check_image(image_path, layout)
    """

    _MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    _DEFAULT_SKIP_CLASSES: set[str] = {
        "tree",
        "wall",
        "sky",
        "road",
        "toy",
    }

    def __init__(
        self,
        threshold: float = 0.5,
        conf_threshold: float = 0.3,
        device: str | None = None,
        skip_classes: set[str] | None = None,
        overdetect_margin: int = 1,
    ):
        """
        Args:
            threshold:      IoU threshold for placement validation.
            conf_threshold: Minimum detection confidence score.
            device:         Torch device; auto-detected if None.
            skip_classes:   Additional class names to skip validation for,
                            merged with the default skip list.
            overdetect_margin:
                            Number of extra detections tolerated for strict
                            classes to reduce detector false-positive noise.
        """
        self.iou_threshold  = threshold
        self.conf_threshold = conf_threshold
        self.overdetect_margin = max(0, int(overdetect_margin))

        self.skip_classes = self._DEFAULT_SKIP_CLASSES.copy()
        if skip_classes:
            self.skip_classes.update(c.lower() for c in skip_classes)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[CriticValidator] Loading {self._MODEL_ID} on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(self._MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._MODEL_ID
        ).to(self.device)
        self.model.eval()
        print("[CriticValidator] Model ready.")

    # ------------------------------------------------------------------
    # Public API — called by LoopManager
    # ------------------------------------------------------------------

    def check_image(
        self,
        image_path: str,
        layout: list[dict],
    ) -> tuple[bool, str | None, list[dict]]:
        """
        Validate all entities in *layout* against the generated image.

        Each layout entry may carry an optional "strict_count" bool:
            {"class_name": "dog",    "box": [...], "strict_count": True}
            {"class_name": "person", "box": [...], "strict_count": False}

        strict_count defaults to False if omitted — meaning over-detection
        is accepted unless explicitly marked strict.

        Args:
            image_path: Path to the generated image file.
            layout:     List of entity dicts with "class_name", "box",
                        and optional "strict_count".

        Returns:
            (passed, feedback, detected_boxes)
                passed:         True only if every entity passes validation.
                feedback:       None on success; failure messages otherwise.
                detected_boxes: All detected boxes for accepted images,
                                used by loop_manager to write label files.
        """
        failures: list[str] = []
        detected_boxes: list[dict] = []

        # Group planned boxes by class.
        # If ANY entry for a class has strict_count=True, the whole class
        # is strict. Default is True (safer for exact-count prompts).
        expected_by_class: dict[str, list[list[float]]] = {}
        strict_by_class:   dict[str, bool]              = {}

        for entity in layout:
            cls = entity["class_name"]
            expected_by_class.setdefault(cls, []).append(entity["box"])
            # Once strict, always strict for that class
            current = strict_by_class.get(cls, False)
            strict_by_class[cls] = current or entity.get("strict_count", False)

        for cls_name, expected_boxes in expected_by_class.items():
            # Skip classes that are out-of-vocabulary or noisy for GDINO.
            # Trust planned boxes for these classes.
            cls_lower = cls_name.lower()
            if cls_lower not in COCO_80 or cls_lower in self.skip_classes:
                print(
                    f"[CriticValidator] Skipping '{cls_name}' "
                    "— not reliably detectable by GDINO."
                )
                detected_boxes.extend(
                    {"class_name": cls_name, "box": box}
                    for box in expected_boxes
                )
                continue

            pred_boxes = self._detect_boxes(image_path, cls_name)

            detected_boxes.extend(
                {"class_name": cls_name, "box": pred_box}
                for pred_box in pred_boxes
            )

            expected_n  = len(expected_boxes)
            actual_n    = len(pred_boxes)
            is_strict   = strict_by_class.get(cls_name, False)
            matches     = _greedy_match_boxes(expected_boxes, pred_boxes)

            # ── Count check ───────────────────────────────────────────
            if actual_n < expected_n:
                failures.append(
                    f"Count mismatch for '{cls_name}': expected {expected_n}, "
                    f"detected {actual_n}. Make each instance more visually distinct."
                )

            elif actual_n > expected_n:
                if is_strict and actual_n > expected_n + self.overdetect_margin:
                    # Prompt said "a dog" or "two cars" — extra is wrong
                    failures.append(
                        f"Too many '{cls_name}' instances: expected exactly "
                        f"{expected_n}, detected {actual_n}. "
                        "Reduce duplicate renderings or simplify the prompt."
                    )
                else:
                    # Prompt said "dogs" or "pedestrians" — extra is fine
                    print(
                        f"[CriticValidator] '{cls_name}': {actual_n} detected vs "
                        f"{expected_n} planned (non-strict) — extra instances accepted."
                    )

            # ── Placement check ───────────────────────────────────────
            # Always run on planned boxes regardless of strict_count.
            for exp_idx, expected_box in enumerate(expected_boxes, start=1):
                match = matches.get(exp_idx - 1)

                if match is None:
                    failures.append(
                        f"Missing the {_ordinal(exp_idx)} '{cls_name}' near "
                        f"{_describe_region(expected_box)} at "
                        f"{_format_box(expected_box)}. "
                        "Add one clear instance in that region."
                    )
                    continue

                pred_idx, best_iou = match
                pred_box = pred_boxes[pred_idx]

                if best_iou >= self.iou_threshold:
                    continue

                failures.append(
                    f"The {_ordinal(exp_idx)} '{cls_name}' is misplaced: "
                    f"expected {_format_box(expected_box)}, "
                    f"detected {_format_box(pred_box)}, "
                    f"IoU={best_iou:.2f}. "
                    f"{_movement_hint(expected_box, pred_box).capitalize()}."
                )

        if not failures:
            return (True, None, detected_boxes)

        summary = "Scene validation failed. Fix the following issues:"
        return (False, "\n".join([summary, *[f"- {msg}" for msg in failures]]), detected_boxes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_boxes(self, image_path: str, target_class: str) -> list[list[float]]:
        """Return normalised detected boxes for *target_class*."""
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        text_prompt = f"{target_class.lower().strip()}."

        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.conf_threshold,
                text_threshold=self.conf_threshold,
                target_sizes=[(h, w)],
            )
        except TypeError:
            # Backward compatibility with older transformers versions.
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.conf_threshold,
                text_threshold=self.conf_threshold,
                target_sizes=[(h, w)],
            )

        boxes = results[0]["boxes"]
        if len(boxes) == 0:
            return []

        norm_boxes = boxes.clone().float()
        norm_boxes[:, [0, 2]] /= w
        norm_boxes[:, [1, 3]] /= h
        return [box.tolist() for box in norm_boxes]

    def _count_detections(self, image_path: str, target_class: str) -> int:
        """Return the number of detected instances of *target_class*."""
        return len(self._detect_boxes(image_path, target_class))