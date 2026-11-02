#!/usr/bin/env python3
"""Apply a yaw GAIN to the heading carried by a run (course of its trajectory), re-integrate at constant speed and write
a new run dir (<out>/predictions/<key>.json) that policy_v004.py / fuse_headings.py can consume.

Why (2026-10-20, E006c): on every confined-water flir-VIS dev clip the visual yaw (VGGT, horizon tracker) is 1.3-1.8x
the GPS course change (camera zoomed beyond the nominal calibration, or pan); gain 0.6-0.7 takes dev flir-VIS from
12.8 to 7.3 (true-heading oracle 6.1).  Gains can be given per camera mode (--gain-mode flir-VIS=0.65 wide=0.9) or per
clip (--gain-json {key: gain}); unlisted clips use --gain (default 1.0).

Usage: scale_heading.py --dataset <manifest> --split test --run runs/fused --out runs/fused_g --gain-mode flir-VIS=0.65 [--clips a,b]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_timestamps, read_json, select_clips, validate_prediction, write_json
from mvo_tools.fuse_heading_speed import heading_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='train,val'); ap.add_argument('--clips')
    ap.add_argument('--run', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--gain', type=float, default=1.0); ap.add_argument('--gain-mode', nargs='*', default=[], help='mode=gain pairs'); ap.add_argument('--gain-json', type=Path)
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent
    gm = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.gain_mode}; gj = json.load(open(a.gain_json)) if a.gain_json else {}
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True); recs = []
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; f = a.run / 'predictions' / f'{key}.json'
        if not f.is_file(): recs.append(dict(key=key, status='missing')); continue
        g = float(gj.get(key, gm.get(c['camera_mode'], a.gain))); ft = load_timestamps(root, c)
        d = read_json(f); t = np.asarray(d['times'], float); fi = np.asarray(d['frame_indices'], int); hd = heading_of(t, np.asarray(d['positions'], float), 0.0)
        hd = hd[0] + g * (hd - hd[0]); dt = np.diff(t)
        x = np.cumsum(np.concatenate([[0], dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], dt * np.sin(hd[1:])]))
        pred = dict(key=key, frame_indices=[int(v) for v in fi], times=t.tolist(), positions=np.column_stack([x, y, np.zeros_like(x)]).tolist())
        validate_prediction(pred, key, ft); write_json(a.out / 'predictions' / f'{key}.json', pred); recs.append(dict(key=key, status='predicted', gain=g))
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/scale_heading.py', run=str(a.run), gain=a.gain, gain_mode=gm, gain_json=gj, clips=recs))
    print('gains used:', sorted(set(r.get('gain') for r in recs if 'gain' in r)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
