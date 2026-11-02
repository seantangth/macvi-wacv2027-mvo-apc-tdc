#!/usr/bin/env python3
"""Re-integrate a trajectory from the HEADING of one run and the SPEED profile of another (e.g. horizon-band yaw
tracker heading + DPVO speed), or GT course as heading (oracle) to measure what a speed profile alone is worth.

Heading run: challenge-format predictions whose shape carries the heading (any planar trajectory; course = atan2 of
successive displacements after --smooth-h seconds of smoothing).  Speed run: speed = |d position| / dt smoothed over
--smooth-s seconds (DPVO scale drift is slow, so a long window keeps stop/go while damping drift); clips missing in the
speed run, or with non-finite speed, fall back to constant speed.  Output run: <out>/predictions/<key>.json on the
heading run's sample grid, plus inference_manifest.json (per clip: speed source used).

Usage: fuse_heading_speed.py --dataset <manifest> --heading-run runs/hyaw/rotw --speed-run runs/dpvo --out runs/fused \
          [--smooth-s 60] [--smooth-h 2] [--gt-heading] [--clips a,b] [--split train,val]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_reference, load_timestamps, read_json, select_clips, validate_prediction, write_json
from mvo_tools.select_policy import smooth


def heading_of(t, xyz, smooth_s):
    p = smooth(xyz, t, smooth_s)[:, :2] if abs(np.ptp(xyz[:, 2])) < 1e-9 else smooth(xyz, t, smooth_s)[:, [0, 2]]
    d = np.diff(p, axis=0); hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); return np.concatenate([[hd[0]], hd])


def speed_of(t, xyz, smooth_s):
    sp = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6); sp = np.concatenate([[sp[0]], sp])
    return np.maximum(smooth(sp[:, None], t, smooth_s)[:, 0], 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='train,val'); ap.add_argument('--clips')
    ap.add_argument('--heading-run', type=Path); ap.add_argument('--gt-heading', action='store_true', help='oracle: use the reference course as heading')
    ap.add_argument('--speed-run', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--smooth-s', type=float, default=60.0); ap.add_argument('--smooth-h', type=float, default=2.0); ap.add_argument('--min-speed-frac', type=float, default=0.0, help='floor speed at this fraction of its median (0 = allow full stops)')
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True); recs = []
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; ft = load_timestamps(root, c)
        if a.gt_heading:
            rt, rx = load_reference(root, c); t = rt; hd = heading_of(rt, rx, 0.0); fi = np.searchsorted(ft, np.clip(t, ft[0], ft[-1])); fi = np.clip(fi, 0, len(ft) - 1)
            fi = np.unique(fi); t = ft[fi]; hd = np.interp(t, rt, hd)
        else:
            f = a.heading_run / 'predictions' / f'{key}.json'
            if not f.is_file():
                recs.append(dict(key=key, status='missing_heading')); continue
            d = read_json(f); t = np.asarray(d['times'], float); fi = np.asarray(d['frame_indices'], int); hd = heading_of(t, np.asarray(d['positions'], float), a.smooth_h)
        sp = np.ones(len(t)); src = 'const'
        g = a.speed_run / 'predictions' / f'{key}.json'
        if g.is_file():
            try:
                e = read_json(g); ts = np.asarray(e['times'], float); xs = np.asarray(e['positions'], float)
                if np.all(np.isfinite(xs)) and len(ts) > 10:
                    s = speed_of(ts, xs, a.smooth_s); s = np.interp(t, ts, s); med = np.median(s)
                    if np.isfinite(med) and med > 0:
                        sp = np.maximum(s / med, a.min_speed_frac); src = 'speed_run'
            except Exception as exc:  # noqa: BLE001
                src = f'const (speed run unreadable: {str(exc)[:40]})'
        dt = np.diff(t); x = np.cumsum(np.concatenate([[0], sp[1:] * dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], sp[1:] * dt * np.sin(hd[1:])]))
        pred = dict(key=key, frame_indices=[int(v) for v in fi], times=t.tolist(), positions=np.column_stack([x, y, np.zeros_like(x)]).tolist())
        validate_prediction(pred, key, ft); write_json(a.out / 'predictions' / f'{key}.json', pred); recs.append(dict(key=key, status='predicted', speed=src))
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/fuse_heading_speed.py', heading_run=str(a.heading_run), gt_heading=a.gt_heading, speed_run=str(a.speed_run), smooth_s=a.smooth_s, clips=recs))
    from collections import Counter; print('speed sources:', dict(Counter(r.get('speed', r['status']) for r in recs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
