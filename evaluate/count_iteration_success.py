#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt


def count_from_manifests(root: Path, include_prefixes=None):
    counts = Counter()
    total = 0
    for p in root.rglob('manifests/*_manifest.json'):
        try:
            if include_prefixes:
                try:
                    rel = p.relative_to(root)
                    top = rel.parts[0] if rel.parts else ''
                except Exception:
                    top = ''
                if not any(top.startswith(pref) for pref in include_prefixes):
                    continue
            payload = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue

        if payload.get('success') is not True:
            continue

        iterations = payload.get('iterations')
        if isinstance(iterations, list):
            # find last iteration where critic_passed is True
            success_iter = None
            for it in iterations:
                if it.get('critic_passed') is True:
                    success_iter = it.get('iteration')
            if success_iter is not None:
                counts[int(success_iter)] += 1
                total += 1
        elif isinstance(iterations, int):
            counts[int(iterations)] += 1
            total += 1

    return counts, total


def count_from_summaries(root: Path, include_prefixes=None):
    counts = Counter()
    total = 0
    for p in root.rglob('batch_summary.json'):
        try:
            if include_prefixes:
                try:
                    rel = p.relative_to(root)
                    top = rel.parts[0] if rel.parts else ''
                except Exception:
                    top = ''
                if not any(top.startswith(pref) for pref in include_prefixes):
                    continue
            payload = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        results = payload.get('results') or []
        for r in results:
            if not isinstance(r, dict):
                continue
            if r.get('success') is not True:
                continue
            iters = r.get('iterations')
            if isinstance(iters, int):
                counts[int(iters)] += 1
                total += 1

    return counts, total


def main():
    roots_arg = sys.argv[1] if len(sys.argv) > 1 else '/scratch/kt68/clove_output,/scratch/qmt1/clove/clove_output_run2'
    out_svg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / 'iteration_success_bar.svg'
    prefixes_arg = sys.argv[3] if len(sys.argv) > 3 else ''

    roots = [Path(s) for s in roots_arg.split(',') if s.strip()]
    include_prefixes = [s.strip() for s in prefixes_arg.split(',') if s.strip()]

    total_counts = Counter()
    grand_total = 0

    for r in roots:
        c1, t1 = count_from_manifests(r, include_prefixes)
        c2, t2 = count_from_summaries(r, include_prefixes)
        for k, v in c1.items():
            total_counts[k] += v
        for k, v in c2.items():
            total_counts[k] += v
        grand_total += t1 + t2
        print(f"Scanned root: {r}  successes found: {t1 + t2}")

    labels = [1, 2, 3, 4, 5]
    values = [total_counts.get(i, 0) for i in labels]

    print("Totals per iteration:")
    for i, v in zip(labels, values):
        pct = round((v / grand_total) * 100) if grand_total > 0 else 0
        print(f"  Iteration {i}: {v}  ({pct}% of successes)")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(i) for i in labels], values, color='#1f77b4')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Successful runs')
    ax.set_title('Successful runs by iteration (1-5)')
    # provide headroom for labels
    top = max(values) * 1.08 + 1 if values else 1
    ax.set_ylim(0, top)
    for b in bars:
        h = b.get_height()
        ax.annotate(str(int(h)), xy=(b.get_x() + b.get_width() / 2, h), xytext=(0,3),
                    textcoords='offset points', ha='center', va='bottom', clip_on=False)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format='svg')
    print(f"Saved iteration success bar chart to: {out_svg}")


if __name__ == '__main__':
    main()
