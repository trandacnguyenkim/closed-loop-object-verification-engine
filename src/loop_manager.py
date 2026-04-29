"""
LoopManager: Orchestrates the closed-loop synthetic data generation pipeline.

Flow per iteration:
  1. LLMPlanner generates an enriched prompt + bounding box layout.
     On retry, the Critic's feedback is injected into the planner prompt.
  2. ImageGenerator produces a grounded image using GLIGEN.
     A fresh random seed is used on each retry so the diffusion model
     explores a different region of the noise space.
  3. CriticValidator scores every entity's bounding box crop with CLIP.
     If all entities pass, the image is accepted.  Otherwise, structured
     feedback is sent back to step 1.

Hard Negative mode: an LLM call transforms the original prompt into a
deliberately challenging variant (low contrast, unusual scale, heavy
occlusion) before entering the loop. This exercises the pipeline on
distribution-edge cases that matter most for downstream detector training.
"""

from __future__ import annotations

import os
import json
import random
import time
from shutil import copy2

from PIL import Image, ImageDraw, ImageFont

from src.llm_planner      import LLMPlanner
from src.image_generator  import generate_grounded_image
from src.critic_validator import CriticValidator
from src.utils            import HardNegativeGenerator, make_run_dir, save_run_manifest, sanitise_img_name


class LoopManager:
    def __init__(
        self,
        max_iterations: int = 3,
        critic_threshold: float = 0.5,
        critic_conf_threshold: float = 0.3,
        overdetect_margin: int = 1,
        output_dir: str = "./dataset",
    ):
        """
        Args:
            max_iterations:   Maximum plan-generate-validate cycles before giving up.
            critic_threshold: CLIP cosine similarity threshold forwarded to CriticValidator.
            critic_conf_threshold:
                              GroundingDINO confidence threshold.
            overdetect_margin:
                              Number of extra detections tolerated for strict classes.
            output_dir:       Root folder for images, labels, and manifests.
        """
        self.max_iterations   = max_iterations
        self.output_dir       = output_dir
        self.planner          = LLMPlanner()
        self.critic           = CriticValidator(
            threshold=critic_threshold,
            conf_threshold=critic_conf_threshold,
            overdetect_margin=overdetect_margin,
        )
        self.hard_neg_gen     = HardNegativeGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_prompt: str,
        img_name: str    = "output",
        hard_negative: bool = False,
        run_timestamp: str | None = None,
        initial_seed: int | None = None,
    ) -> dict:
        """
        Runs the full closed-loop pipeline for a single scene prompt.

        Args:
            user_prompt:   Natural-language description of the desired scene.
            img_name:      Base filename (no extension) for saved artefacts.
            hard_negative: When True, the prompt is first transformed into a
                           harder variant (e.g. low-contrast, unusual scale).
            run_timestamp: Optional shared timestamp (mmddyy-hhmmss). If provided,
                           outputs are grouped under output_dir/run_timestamp/.
            initial_seed:  Optional seed for the first generation attempt.
                           If omitted, a fixed seed is used for reproducibility.

        Returns:
            A result dict with keys:
              "success"     - bool
              "iterations"  - int (how many cycles were needed)
              "final_image" - str | None (path to accepted image)
              "manifest"    - dict (full run log for reproducibility)
        """
        run_start = time.time()

        # --- Optional: transform into a hard negative scene ---
        active_prompt = user_prompt
        hard_neg_info = None
        if hard_negative:
            active_prompt, hard_neg_info = self.hard_neg_gen.transform(user_prompt)
            print(f"[LoopManager] Hard-negative prompt: {active_prompt}")
            print(f"[LoopManager] Difficulty factors:   {hard_neg_info}")

        safe_img_name = sanitise_img_name(img_name) or "output"
        timestamp = run_timestamp or time.strftime("%m%d%y-%H%M%S")
        run_dir = make_run_dir(os.path.join(self.output_dir, timestamp, safe_img_name))

        feedback   = None     
        manifest   = {
            "original_prompt": user_prompt,
            "active_prompt":   active_prompt,
            "hard_negative":   hard_negative,
            "hard_neg_info":   hard_neg_info,
            "run_dir":         run_dir,
            "iterations":      [],
        }

        for iteration in range(1, self.max_iterations + 1):
            iter_log = {"iteration": iteration}
            print(f"\n[LoopManager] ── Iteration {iteration}/{self.max_iterations} ──")

            # ── Step 1: Plan ──────────────────────────────────────────
            layout_data = self.planner.generate_layout(active_prompt, feedback=feedback)
            if layout_data is None:
                print("[LoopManager] Planning failed. Aborting.")
                iter_log["status"] = "plan_failed"
                manifest["iterations"].append(iter_log)
                break

            enriched_prompt = layout_data["enriched_prompt"]
            layout          = layout_data["layout"]
            entities        = [obj["class_name"] for obj in layout]
            boxes           = [obj["box"]        for obj in layout]

            iter_log["enriched_prompt"] = enriched_prompt
            iter_log["layout"]          = layout

            print(f"[LoopManager] Entities: {entities}")

            # ── Step 2: Generate ──────────────────────────────────────
            # Use a caller-provided seed on the first attempt when batching,
            # otherwise fall back to a fixed seed for reproducibility.
            # Retries always use fresh random seeds.
            if iteration == 1:
                seed = initial_seed if initial_seed is not None else 6
            else:
                seed = random.randint(0, 2**31 - 1)
            iter_img_name = f"{safe_img_name}_iter{iteration}"

            iter_log["seed"] = seed

            try:
                image_path = generate_grounded_image(
                    prompt     = enriched_prompt,
                    img_name   = iter_img_name,
                    entity     = entities,
                    boxes      = boxes,
                    seed       = seed,           
                    output_dir = run_dir,
                )
            except Exception as exc:
                print(f"[LoopManager] Image generation error: {exc}")
                iter_log["status"] = "generation_failed"
                iter_log["error"]  = str(exc)
                manifest["iterations"].append(iter_log)
                feedback = f"Image generation itself failed with error: {exc}. Try a simpler layout."
                continue

            # ── Step 3: Validate ──────────────────────────────────────
            passed, feedback, detected_boxes = self.critic.check_image(image_path, layout)

            iter_log["critic_passed"]   = passed
            iter_log["critic_feedback"] = feedback
            iter_log["detected_boxes"]  = detected_boxes
            iter_log["image_path"]      = image_path

            print(f"[LoopManager] Critic: {'PASSED ✓' if passed else 'FAILED ✗'}")
            if not passed:
                print(f"[LoopManager] Feedback →\n{feedback}")
                self._save_annotated_image(
                    image_path, layout, detected_boxes, iter_img_name,
                )

            manifest["iterations"].append(iter_log)

            if passed:
                print(f"[LoopManager] Accepted on iteration {iteration}.")
                final_path = self._save_annotated_image(
                    image_path, layout, detected_boxes, iter_img_name,
                )
                manifest["success"]     = True
                manifest["final_image"] = final_path
                manifest["total_time"]  = round(time.time() - run_start, 2)
                save_run_manifest(manifest, safe_img_name, run_dir)
                return self._build_result(True, iteration, final_path, manifest)

        # Exhausted retries
        print(f"[LoopManager] Pipeline failed after {self.max_iterations} iteration(s).")
        manifest["success"]     = False
        manifest["final_image"] = None
        manifest["total_time"]  = round(time.time() - run_start, 2)
        save_run_manifest(manifest, safe_img_name, run_dir)
        return self._build_result(False, self.max_iterations, None, manifest)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_annotated_image(
        self,
        image_path: str,
        layout: list,
        detected_boxes: list[dict],
        iter_img_name: str,
    ) -> str:
        """
        Copy the image as *_final.jpg in the images/ subdirectory and draw:
          • GREEN boxes — planned bounding boxes from the LLM planner
          • RED   boxes — detected bounding boxes from the critic validator
        Each box is labelled with its class name.
        """
        images_dir = os.path.dirname(image_path)
        final_path = os.path.join(images_dir, f"{iter_img_name}_final.jpg")
        copy2(image_path, final_path)

        img = Image.open(final_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Try to load a TrueType font; fall back to the default bitmap font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # --- Green: LLM planner boxes ---
        for obj in layout:
            self._draw_box(draw, obj["box"], obj["class_name"], w, h,
                           color="green", font=font, tag="plan")

        # --- Red: Critic detected boxes ---
        for det in detected_boxes:
            self._draw_box(draw, det["box"], det["class_name"], w, h,
                           color="red", font=font, tag="det")

        img.save(final_path)
        print(f"[LoopManager] Annotated final image saved → {final_path}")
        return final_path

    @staticmethod
    def _draw_box(
        draw: ImageDraw.ImageDraw,
        box: list[float],
        cls_name: str,
        img_w: int,
        img_h: int,
        color: str,
        font,
        tag: str = "",
    ) -> None:
        """Draw a single labelled bounding box on *draw*."""
        xmin = box[0] * img_w
        ymin = box[1] * img_h
        xmax = box[2] * img_w
        ymax = box[3] * img_h

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        label = f"{cls_name} [{tag}]" if tag else cls_name
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [xmin, ymin - text_h - 4, xmin + text_w + 4, ymin],
            fill=color,
        )
        draw.text((xmin + 2, ymin - text_h - 2), label, fill="white", font=font)

    def _build_result(
        self,
        success: bool,
        iterations: int,
        final_image,
        manifest: dict,
    ) -> dict:
        return {
            "success":     success,
            "iterations":  iterations,
            "final_image": final_image,
            "manifest":    manifest,
        }


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    manager = LoopManager(max_iterations=10)

    # Standard run
    result = manager.run("A construction site with people working. Some operates heavy machines, some are carrying materials, and some are working near edges of scaffolding", img_name="construction")
    print(json.dumps(result["manifest"], indent=2))

    # Hard-negative run
    result = manager.run(
        "A black cat lying on a dark charcoal sofa",
        img_name  = "hard_neg_cat",
        hard_negative = True,
    )
    print(json.dumps(result["manifest"], indent=2))