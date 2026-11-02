#!/usr/bin/env python3
"""Per-clip selection policy (no ground truth needed): decide, from a VO run's own trajectory statistics, whether to
keep the VO trajectory, re-integrate its heading at constant speed, or fall back to a timestamp-only straight line.

Rationale (E001/E001a/E001b, 2026-10-19): the metric rewards 'straight + constant speed' on transit clips; DPVO's
failures show up as implausible trajectories (path length / net displacement >> 1, speed CV >> 1) or NaN.
Rule (mode-aware, thresholds tuned on 34 dev clips):
    insane  = failed or L/net > T or speedCV > C      -> straight line
    wide    -> VO raw;   flir-VIS / flir-IR / other -> VO heading + constant speed  (unless --alt raw)
Writes a run dir: predictions/*.json + inference_manifest.json (clips: key, choice, stats) usable by assemble_submission.py.

Usage: select_policy.py --dataset <manifest> --split test --run runs/dpvo_fps8_s0_test --out runs/selected_v002 [--T 3 --C 1.5 --alt mode|raw|hc]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_timestamps, read_json, select_clips, validate_prediction, write_json


def smooth(x, t, win_s):
    out = np.empty_like(x)
    for i in range(len(t)):
        a = np.searchsorted(t, t[i] - win_s / 2); b = np.searchsorted(t, t[i] + win_s / 2, side='right'); out[i] = x[a:b].mean(axis=0)
    return out


def heading_const_speed(t, xyz, plane=(0, 2), smooth_s=2.0):
    p = smooth(xyz, t, smooth_s)[:, list(plane)]; d = np.diff(p, axis=0); dt = np.diff(t)
    hd = np.arctan2(d[:, 1], d[:, 0]); hd = np.concatenate([[hd[0]], hd])
    x = np.cumsum(np.concatenate([[0], dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], dt * np.sin(hd[1:])]))
    return np.column_stack([x, y, np.zeros_like(x)])


def straight_line(ft, stride=3):
    fi = np.arange(0, len(ft), stride)
    if fi[-1] != len(ft) - 1:
        fi = np.append(fi, len(ft) - 1)
    t = ft[fi]; return fi, np.column_stack([t - t[0], np.zeros(len(fi)), np.zeros(len(fi))])


def traj_stats(t, xyz):
    sp = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(t); L = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    net = float(np.linalg.norm(xyz[-1] - xyz[0]))
    return dict(lnet=L / max(net, 1e-9), spdcv=float(np.std(sp) / max(sp.mean(), 1e-9)), rawspd=float(sp.mean()), n=len(t))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='test'); ap.add_argument('--clips')
    ap.add_argument('--run', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--T', type=float, default=3.0); ap.add_argument('--C', type=float, default=1.5)
    ap.add_argument('--alt', default='mode', choices=['mode', 'raw', 'hc']); ap.add_argument('--nan-max', type=float, default=0.2, help='treat a VO trajectory with NaN-repaired fraction above this as failed')
    a = ap.parse_args(argv)
    man = load_manifest(a.dataset); root = a.dataset.resolve().parent; clips = select_clips(man, a.split, a.clips)
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True); recs = []
    # NaN-repair bookkeeping from infer_dpvo.py: a heavily repaired trajectory is treated as failed
    nanfrac = {}
    mf = a.run / 'inference_manifest.json'
    if mf.is_file():
        try:
            for r in json.load(open(mf)).get('clips', []):
                if r.get('fallback'): nanfrac[r['key']] = 1.0
                elif r.get('n_nan'): nanfrac[r['key']] = float(r.get('nan_frac') or 0.0)
        except Exception:  # noqa: BLE001
            pass
    for c in clips:
        key = c['key']; ft = load_timestamps(root, c); f = a.run / 'predictions' / f'{key}.json'; rec = dict(key=key, split=c['split'], mode=c['camera_mode'])
        pred = None
        if f.is_file():
            try:
                d = read_json(f); validate_prediction(d, key, ft); t = np.asarray(d['times'], float); xyz = np.asarray(d['positions'], float); fi = np.asarray(d['frame_indices'], int)
                st = traj_stats(t, xyz); rec.update(st)
                insane = st['lnet'] > a.T or st['spdcv'] > a.C or nanfrac.get(key, 0.0) > a.nan_max
                if not insane:
                    choice = a.alt if a.alt != 'mode' else ('raw' if c['camera_mode'] == 'wide' else 'hc')
                    out = xyz if choice == 'raw' else heading_const_speed(t, xyz)
                    pred = dict(key=key, frame_indices=fi.tolist(), times=t.tolist(), positions=out.tolist()); rec['choice'] = choice
                else:
                    rec['choice'] = 'line'; rec['reason'] = (f"nan_frac={nanfrac.get(key, 0.0):.2f}>{a.nan_max}" if nanfrac.get(key, 0.0) > a.nan_max else f"lnet={st['lnet']:.2f}>{a.T}" if st['lnet'] > a.T else f"spdcv={st['spdcv']:.2f}>{a.C}")
            except Exception as exc:  # noqa: BLE001
                rec.update(choice='line', reason=f'invalid VO prediction: {str(exc)[:80]}')
        else:
            rec.update(choice='line', reason='no VO prediction (failed)')
        if pred is None:
            fi, xyz = straight_line(ft); pred = dict(key=key, frame_indices=fi.tolist(), times=ft[fi].tolist(), positions=xyz.tolist())
        validate_prediction(pred, key, ft); write_json(a.out / 'predictions' / f'{key}.json', pred)
        rec['fallback'] = rec['choice'] == 'line'; rec['status'] = 'predicted'; recs.append(rec)
        print(f"{key} {c['camera_mode']:8s} -> {rec['choice']:4s} {rec.get('reason','')} lnet={rec.get('lnet', float('nan')):.2f} spdcv={rec.get('spdcv', float('nan')):.2f}")
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/select_policy.py', source_run=str(a.run), T=a.T, C=a.C, alt=a.alt, clips=recs))
    from collections import Counter; print('choices:', dict(Counter(r['choice'] for r in recs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
