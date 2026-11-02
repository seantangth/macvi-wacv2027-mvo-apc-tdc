#!/usr/bin/env python3
"""Local re-implementation of the SERVER's gap-aware coverage gate (RULES.md, "Held-out competition submissions").

Rule: a reference timestamp is covered if a predicted sample exists at that time, or if it lies between two
consecutive predicted samples that are no more than MAX_GAP_S (1.0 s) apart. Reference timestamps outside the
predicted time range are NOT covered. An entry is eligible only if every clip has coverage >= 0.95.

The starter's evaluate.py only reports an overlap-window diagnostic; it does NOT implement this gate.
For the test split we have no reference times, so we approximate them with the released frame PTS of the clip
(the organizers' reference clock is the same seconds clock as the video PTS; reference sampling rate is
unknown for test, so frame PTS is the strictest proxy).
"""
from __future__ import annotations
import numpy as np

MAX_GAP_S = 1.0
MIN_COVERAGE = 0.95


def gap_aware_coverage(pred_times, ref_times, max_gap_s: float = MAX_GAP_S) -> dict:
    p = np.asarray(pred_times, dtype=float)
    r = np.asarray(ref_times, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not np.all(np.diff(p) > 0):
        raise ValueError('pred_times must be strictly increasing with >= 2 samples')
    # index of the last predicted sample <= t
    prev = np.searchsorted(p, r, side='right') - 1
    covered = np.zeros(len(r), dtype=bool)
    inside = (prev >= 0) & (prev <= len(p) - 1)
    exact = inside & (p[np.clip(prev, 0, len(p) - 1)] == r)
    between = inside & (prev < len(p) - 1)
    gap = np.full(len(r), np.inf)
    gap[between] = p[prev[between] + 1] - p[prev[between]]
    covered = exact | (between & (gap <= max_gap_s))
    frac = float(covered.mean()) if len(r) else 0.0
    return dict(coverage=frac, eligible=bool(frac >= MIN_COVERAGE), n_reference=int(len(r)),
                n_covered=int(covered.sum()), max_pred_gap_s=float(np.diff(p).max()),
                first_uncovered_time=(float(r[~covered][0]) if (~covered).any() else None))


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description='Check the gap-aware coverage gate for a predictions run.')
    ap.add_argument('--dataset', required=True, type=Path, help='release_manifest.json')
    ap.add_argument('--predictions', required=True, type=Path, help='run dir containing predictions/')
    ap.add_argument('--split', default='val')
    ap.add_argument('--clips')
    a = ap.parse_args()
    import mvo_tools  # noqa: F401  (adds starter to sys.path)
    from public_dataset import load_manifest, load_reference, load_timestamps, read_json, select_clips, validate_prediction
    man = load_manifest(a.dataset)
    root = a.dataset.resolve().parent
    pred_dir = a.predictions / 'predictions' if (a.predictions / 'predictions').is_dir() else a.predictions
    rows, all_ok = [], True
    for clip in select_clips(man, a.split, a.clips):
        key = clip['key']
        f = pred_dir / f'{key}.json'
        frame_t = load_timestamps(root, clip)
        if not f.is_file():
            rows.append((key, 'missing', 0.0)); all_ok = False; continue
        try:
            est_t, _ = validate_prediction(read_json(f), key, frame_t)
        except Exception as exc:  # noqa: BLE001
            rows.append((key, f'invalid: {exc}', 0.0)); all_ok = False; continue
        ref_t = load_reference(root, clip)[0] if clip['reference'] else frame_t
        res = gap_aware_coverage(est_t, ref_t)
        all_ok &= res['eligible']
        rows.append((key, 'ok' if res['eligible'] else 'INELIGIBLE', res['coverage'], res['max_pred_gap_s'], res['first_uncovered_time']))
    for r in rows:
        print(*r)
    print('ALL_ELIGIBLE' if all_ok else 'NOT_ELIGIBLE')
    raise SystemExit(0 if all_ok else 1)
