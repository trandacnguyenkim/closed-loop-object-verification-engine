#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def classify_manifest(manifest: dict):
    iters = manifest.get('iterations', [])
    success = manifest.get('success', False)
    if success:
        return None

    for it in iters:
        status = it.get('status')
        if status:
            if 'plan_incomplete' in status:
                return 'Planning incomplete'
            if 'plan_failed' in status:
                return 'Entity not detected'
            if 'generation' in status:
                return 'Generation error'

    for it in iters:
        fb = it.get('critic_feedback') or ''
        if not fb:
            continue
        fb_lower = fb.lower()
        if 'too many' in fb_lower or 'expected exactly' in fb_lower:
            return 'Over-detection'
        if 'missing' in fb_lower or 'detected 0' in fb_lower or 'count mismatch' in fb_lower:
            return 'Entity not detected'
        if 'iou=' in fb_lower or 'misplaced' in fb_lower:
            return 'IoU too low'
        if 'generation' in fb_lower:
            return 'Generation error'

    try:
        if isinstance(iters, list) and len(iters) >= 5:
            return 'Budget exhausted'
    except Exception:
        pass

    if manifest.get('final_image') in (None, ''):
        return 'Planning incomplete'
    return 'Entity not detected'


def collect_from_root(root: Path, include_prefixes=None):
    if include_prefixes is None:
        include_prefixes = []
    counts = {
        'Entity not detected': 0,
        'IoU too low': 0,
        'Generation error': 0,
        'Budget exhausted': 0,
        'Planning incomplete': 0,
        'Over-detection': 0,
    }
    total_failed = 0

    for p in root.rglob('manifests/*_manifest.json'):
        try:
            # optional filter by top-level folder name
            if include_prefixes:
                try:
                    rel = p.relative_to(root)
                    top = rel.parts[0] if rel.parts else ''
                except Exception:
                    top = ''
                if not any(top.startswith(pref) for pref in include_prefixes):
                    continue

            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        reason = classify_manifest(data)
        if reason is None:
            continue
        total_failed += 1
        if reason not in counts:
            counts[reason] = 0
        counts[reason] += 1

    return counts, total_failed


def main():
    # Usage: combine_failure_reasons.py root1,root2,... out.svg [prefix1,prefix2]
    roots_arg = sys.argv[1] if len(sys.argv) > 1 else '/scratch/kt68/clove_output,/scratch/qmt1/clove/clove_output_run2'
    out_svg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / 'failure_reasons_bar_combined.svg'
    prefixes_arg = sys.argv[3] if len(sys.argv) > 3 else ''

    roots = [Path(s) for s in roots_arg.split(',') if s.strip()]
    include_prefixes = [s.strip() for s in prefixes_arg.split(',') if s.strip()]

    total_counts = {
        'Entity not detected': 0,
        'IoU too low': 0,
        'Generation error': 0,
        'Budget exhausted': 0,
        'Planning incomplete': 0,
        'Over-detection': 0,
    }
    grand_failed = 0

    for r in roots:
        c, failed = collect_from_root(r, include_prefixes)
        print(f"Root: {r}  failed runs: {failed}")
        for k, v in c.items():
            total_counts[k] = total_counts.get(k, 0) + v
        grand_failed += failed

    labels = [k for k, v in total_counts.items() if v > 0]
    values = [total_counts[l] for l in labels]

    print(f"Combined failed runs: {grand_failed}")
    for k, v in total_counts.items():
        print(f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {
        'Entity not detected': '#d62728',
        'IoU too low': '#ff7f0e',
        'Generation error': '#2ca02c',
        'Budget exhausted': '#1f77b4',
        'Planning incomplete': '#9467bd',
        'Over-detection': '#8c564b',
    }
    colors = [palette.get(lbl, '#7f7f7f') for lbl in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Combined failure reasons for critic across runs')
    # Add a small headroom so the numeric labels above bars are not clipped
    if values:
        top = max(values) * 1.08 + 1
    else:
        top = 1
    ax.set_ylim(0, top)
    ax.tick_params(axis='x', rotation=20)
    for b in bars:
        h = b.get_height()
        # disable clipping so labels that slightly overflow the axes are visible
        ax.annotate(str(int(h)), xy=(b.get_x() + b.get_width() / 2, h), xytext=(0,3),
                    textcoords='offset points', ha='center', va='bottom', clip_on=False)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format='svg')
    print(f"Saved combined bar chart to: {out_svg}")


if __name__ == '__main__':
    main()
