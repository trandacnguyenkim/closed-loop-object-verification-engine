"""
main.py — Batch generation script for 250 diverse image scenarios.

By default, each scenario generates both a standard and hard-negative
version, yielding 500 total runs. Use --standard-only to run just the
standard mode for all scenarios. Results are saved in timestamped
directories under /scratch/{USER}/clove_output/ with full iteration logs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.loop_manager import LoopManager
from src.utils import save_coco_annotations
from images_generation.SCENARIOS import SCENARIOS

USER="kt68"  # Update this to your username for correct paths


# ══════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════

# Run this before executing main.py to set up the environment:
# git clone https://github.com/limuloo/MIGC
# cd MIGC && pip install -e .

def generate_images(smoke: bool = False, standard_only: bool = True, initial_seed: int = 91):
    """
    Run all scenarios in standard mode only, or in both standard and
    hard-negative modes by default.
    """
    max_iterations = 1 if smoke else 5
    manager = LoopManager(max_iterations=max_iterations, output_dir=f"/scratch/{USER}/clove_output")
    results_summary = []
    coco_records = []  # accumulates (image_path, layout) for accepted images
    batch_timestamp = time.strftime("%m%d%y-%H%M%S")
    print(f"[Main] Batch timestamp directory: {batch_timestamp}")

    scenarios = SCENARIOS[:1] if smoke else SCENARIOS
    run_modes = [False] if smoke or standard_only else [False, True]

    if smoke:
        print("[Main] Smoke mode enabled: 1 scenario, standard mode only, max_iterations=1")
    elif standard_only:
        print("[Main] Standard-only mode enabled: all scenarios, standard mode only")

    total_runs = len(scenarios) * len(run_modes)
    run_count = 0

    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_prompt = scenario["prompt"]

        for hard_neg in run_modes:
            run_count += 1
            mode = "hard_negative" if hard_neg else "standard"
            img_name = f"{scenario_name}_{mode}"

            print(f"\n{'='*70}")
            print(f"[Main] Run {run_count}/{total_runs}")
            print(f"[Main] Scenario: {scenario_name} ({mode.upper()})")
            print(f"[Main] Prompt: {scenario_prompt}")
            print(f"{'='*70}")

            result = manager.run(
                user_prompt=scenario_prompt,
                img_name=img_name,
                hard_negative=hard_neg,
                run_timestamp=batch_timestamp,
                initial_seed=initial_seed,
            )

            # Log the result
            run_summary = {
                "scenario": scenario_name,
                "mode": mode,
                "success": result["success"],
                "iterations": result["iterations"],
                "final_image": result["final_image"],
                "run_dir": result["manifest"].get("run_dir"),
            }
            results_summary.append(run_summary)

            if result["success"] and result["final_image"]:
                # The last iteration entry is always the accepted one when success=True.
                last_iter = result["manifest"]["iterations"][-1]
                coco_records.append({
                    "image_path": result["final_image"],
                    "layout":     last_iter["layout"],
                })

            if result["success"]:
                print(f"[Main] ✓ SUCCESS in {result['iterations']} iteration(s)")
                print(f"[Main] Final image: {result['final_image']}")
            else:
                print(f"[Main] ✗ FAILED after {result['iterations']} iteration(s)")

    # --- Save overall summary ---
    summary_dir = os.path.join(f"/scratch/{USER}/clove_output", batch_timestamp)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "batch_timestamp": batch_timestamp,
                "total_runs": total_runs,
                "successful_runs": sum(1 for r in results_summary if r["success"]),
                "results": results_summary,
            },
            f,
            indent=2,
        )
    print(f"\n[Main] Batch summary saved → {summary_path}")

    # --- Save COCO annotations for all accepted images ---
    save_coco_annotations(
        coco_records,
        output_path=os.path.join(summary_dir, "coco_annotations.json"),
        dataset_root="./dataset",
    )

    # --- Print completion stats ---
    successful = sum(1 for r in results_summary if r["success"])
    print(f"\n{'='*70}")
    print(f"[Main] BATCH COMPLETE")
    print(f"[Main] Successful: {successful}/{total_runs}")
    print(f"[Main] Success rate: {100 * successful / total_runs:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal smoke test: first scenario, standard mode, max_iterations=1.",
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Run all scenarios in standard mode only (250 total runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=91,
        help="Initial random seed used for generation. Change this to create new variations.",
    )
    args = parser.parse_args()
    generate_images(smoke=args.smoke, standard_only=args.standard_only, initial_seed=args.seed)
