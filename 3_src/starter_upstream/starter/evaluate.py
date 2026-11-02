#!/usr/bin/env python3
"""Score a public dataset selection, including missing and invalid predictions. CPU/NumPy only."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import metrics
from public_dataset import (load_manifest, load_reference, load_timestamps, read_json,
                            select_clips, sha256, validate_prediction, write_json)


def evaluate_clip(root, clip, prediction_path):
    rec = dict(key=clip['key'], split=clip['split'], status='missing_prediction', drift_pct=None)
    if not prediction_path.is_file():
        return rec, None
    try:
        frame_times = load_timestamps(root, clip)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(rec, status='invalid_dataset', reason=str(exc)), None
    try:
        pred = read_json(prediction_path)
        est_t, est_xyz = validate_prediction(pred, clip['key'], frame_times)
    except FloatingPointError as exc:
        return dict(rec, status='non_finite_estimate', reason=str(exc)), None
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return dict(rec, status='invalid_prediction', reason=str(exc)), None
    if clip['reference'] is None:
        return dict(rec, status='missing_gt', reason='test references are withheld'), None
    try:
        ref_t, ref_xyz = load_reference(root, clip)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(rec, status='invalid_dataset', reason=str(exc)), None
    overlap = metrics.overlap_window(est_t, ref_t)
    rec.update(reference_points=len(ref_t), overlap_points=int(overlap.sum()),
               temporal_coverage=float(overlap.sum()/len(ref_t)),
               max_prediction_gap_s=float(np.diff(est_t).max()))
    if overlap.sum() < 10:
        return dict(rec, status='no_overlap'), None
    vo = metrics.score_trajectory(est_t, est_xyz, ref_t, ref_xyz)
    if vo is None:
        return dict(rec, status='degenerate_estimate'), None
    if not np.isfinite(vo['drift_pct']):
        return dict(rec, status='no_eligible_segment'), vo
    return dict(rec, status='scored', drift_pct=round(float(vo['drift_pct']), 6),
                ate_m=round(float(vo['ate_m']), 6), sim3_scale=float(vo['scale']),
                path_length_m=float(vo['path_length_m'])), vo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, type=Path, help='public release_manifest.json')
    ap.add_argument('--predictions', required=True, type=Path, help='run containing predictions/')
    ap.add_argument('--split', default='val', help='expected split, independent of submitted files')
    ap.add_argument('--clips', help='explicit expected anonymous IDs; overrides --split')
    ap.add_argument('--output', type=Path, help='default: predictions run directory')
    args = ap.parse_args(argv)
    try:
        manifest = load_manifest(args.dataset)
        selected = select_clips(manifest, args.split, args.clips)
    except (OSError, ValueError, KeyError) as exc:
        ap.error(str(exc))
    root = args.dataset.resolve().parent
    output = args.output or args.predictions
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in selected:
        rec, vo = evaluate_clip(root, clip, args.predictions / 'predictions' / f"{clip['key']}.json")
        results.append(rec)
        if vo is not None:
            (output / 'trajectories').mkdir(exist_ok=True)
            np.savez(output / 'trajectories' / f"{clip['key']}.npz", times=vo['times'],
                     reference=vo['reference'], estimate=vo['aligned'])
        else:
            (output / 'trajectories' / f"{clip['key']}.npz").unlink(missing_ok=True)
        write_json(output / 'results.json', results)
        print(f"{clip['key']}: {rec['status']} drift={rec['drift_pct']}", flush=True)
    scored = [r['drift_pct'] for r in results if r['status'] == 'scored']
    selected_keys = {c['key'] for c in selected}
    unexpected = sorted(p.stem for p in (args.predictions / 'predictions').glob('*.json')
                        if p.stem not in selected_keys)
    summary = dict(dataset_sha256=sha256(args.dataset), expected=len(results), scored=len(scored),
                   unscored=len(results)-len(scored), by_status=dict(Counter(r['status'] for r in results)),
                   median_drift_pct_scored=float(np.median(scored)) if scored else None,
                   mean_drift_pct_scored=float(np.mean(scored)) if scored else None,
                   unexpected_prediction_ids=unexpected,
                   score_scope='Scored clips only; v0.1 has no official ranking/failure penalty.')
    write_json(output / 'summary.json', summary)
    print(f"EVAL_COMPLETE: {summary['expected']} expected, {summary['scored']} scored, "
          f"{summary['unscored']} unscored; median over scored clips={summary['median_drift_pct_scored']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
