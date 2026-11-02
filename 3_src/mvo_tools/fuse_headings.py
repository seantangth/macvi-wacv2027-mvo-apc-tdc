#!/usr/bin/env python3
"""Fuse the HEADING of two or more runs (weighted average of yaw increments on a common time grid), re-integrate at
constant speed, write a challenge-format run (<out>/predictions/<key>.json) that policy_v004.py can route to.

Each input run carries its heading in the shape of its trajectory (course = atan2 of successive displacements; any
planar constant-speed or real trajectory works).  Yaw is unwrapped, interpolated onto the FIRST run's sample grid,
and the per-step increments are averaged with the given weights (weights renormalised over the runs that have the
clip).  --agree-deg N (optional): per clip, if the RMS yaw difference between the runs exceeds N degrees, fall back
to the first run alone (diagnostic in inference_manifest.json).

Usage: fuse_headings.py --dataset <manifest> --split train,val --clips a,b --runs runs/vggt/heading runs/hyaw/sel \
          --weights 0.5 0.5 --out runs/fused [--smooth-h 0] [--agree-deg 0]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_timestamps, read_json, select_clips, validate_prediction, write_json
from mvo_tools.fuse_heading_speed import heading_of


def load_heading(run: Path, key: str, smooth_h: float):
    f = run / 'predictions' / f'{key}.json'
    if not f.is_file(): return None
    d = read_json(f); t = np.asarray(d['times'], float); fi = np.asarray(d['frame_indices'], int); xyz = np.asarray(d['positions'], float)
    return t, fi, heading_of(t, xyz, smooth_h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='train,val'); ap.add_argument('--clips')
    ap.add_argument('--runs', nargs='+', required=True, type=Path); ap.add_argument('--weights', nargs='+', type=float); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--smooth-h', type=float, default=0.0); ap.add_argument('--agree-deg', type=float, default=0.0); ap.add_argument('--median', action='store_true', help='per-step median of the increments across runs instead of the weighted mean')
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent
    w = np.asarray(a.weights if a.weights else [1.0] * len(a.runs), float); assert len(w) == len(a.runs)
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True); recs = []
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; ft = load_timestamps(root, c); hs = [load_heading(r, key, a.smooth_h) for r in a.runs]
        if hs[0] is None:
            recs.append(dict(key=key, status='missing_base')); print(f'{key}: base run missing'); continue
        t, fi, h0 = hs[0]; incs = []; ws = []; used = []; rms = []
        for (h, wi, r) in zip(hs, w, a.runs):
            if h is None: continue
            th, _, hh = h; hi = np.interp(t, th, hh) if len(th) != len(t) or np.any(th != t) else hh
            hi = hi - hi[0] + h0[0]; rms.append(float(np.sqrt(np.mean((hi - h0) ** 2)) * 180 / np.pi)); incs.append(np.diff(hi)); ws.append(wi); used.append(r.name)
        ws = np.asarray(ws) / np.sum(ws); dz = np.median(np.stack(incs), axis=0) if (a.median and len(incs) >= 2) else sum(wi * inc for wi, inc in zip(ws, incs)); fallback = False
        if a.agree_deg > 0 and len(rms) > 1 and max(rms[1:]) > a.agree_deg:
            dz = incs[0]; fallback = True
        hd = np.concatenate([[h0[0]], h0[0] + np.cumsum(dz)]); dt = np.diff(t)
        x = np.cumsum(np.concatenate([[0], dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], dt * np.sin(hd[1:])]))
        pred = dict(key=key, frame_indices=[int(v) for v in fi], times=t.tolist(), positions=np.column_stack([x, y, np.zeros_like(x)]).tolist())
        validate_prediction(pred, key, ft); write_json(a.out / 'predictions' / f'{key}.json', pred)
        recs.append(dict(key=key, status='predicted', runs=used, weights=[round(float(v), 3) for v in ws], rms_diff_deg=[round(v, 1) for v in rms], fallback_first=fallback))
        print(f"{key}: fused {used} w={np.round(ws, 2).tolist()} rms_diff_deg={np.round(rms, 1).tolist()}{' FALLBACK->first' if fallback else ''}")
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/fuse_headings.py', runs=[str(r) for r in a.runs], weights=w.tolist(), agree_deg=a.agree_deg, clips=recs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
