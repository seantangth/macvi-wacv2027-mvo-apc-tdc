#!/usr/bin/env python3
"""E021 (2026-10-24): calibrate the ROTATION SCALE of a heading family on the TRAINING split (what the 12.2 / 12.4
leaderboard teams disclose for VGGT / Pi3X on the wide camera): per clip, regress GT course increments on the family's
course increments over --step seconds (robust: Theil-Sen median of pairwise slopes on the 2 s increments), report the
slope's sign (mirror) and magnitude, and the robust median gain over clips.  The gain is then applied with
scale_heading.py --gain-mode wide=<gain> (after align_heading_sign.py), and validated on val wide 21.

Usage: fit_yaw_gain.py --dataset <dev manifest> --run runs/pi3x/heading --clips a,b,... [--step 2] [--out gain.json]
Prints per clip: n, corr, slope (signed), |slope|; summary: median |slope| over clips with corr >= --min-corr.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_reference, read_json


def course(t, pos, step):
    tt = np.arange(t[0], t[-1], step)
    if len(tt) < 6: return None, None
    x = np.interp(tt, t, pos[:, 0]); y = np.interp(tt, t, pos[:, 1]); return tt[1:], np.unwrap(np.arctan2(np.diff(y), np.diff(x)))


def theil_sen(x, y, max_pairs=20000):
    n = len(x); rng = np.random.default_rng(0)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs); m = i != j; i, j = i[m], j[m]
    dx = x[j] - x[i]; dy = y[j] - y[i]; ok = np.abs(dx) > 1e-6; return float(np.median(dy[ok] / dx[ok])) if ok.sum() > 10 else float('nan')


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--run', required=True, type=Path); ap.add_argument('--clips', required=True)
    ap.add_argument('--step', type=float, default=2.0); ap.add_argument('--min-corr', type=float, default=0.5); ap.add_argument('--min-turn-deg', type=float, default=20.0); ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent; clips = {c['key']: c for c in man['clips']}; rows = []
    for key in a.clips.split(','):
        f = a.run / 'predictions' / f'{key}.json'
        if not f.is_file(): print(f'{key}: missing', file=sys.stderr); continue
        d = read_json(f); rt, rx = load_reference(root, clips[key])
        ta, ha = course(np.asarray(d['times'], float), np.asarray(d['positions'], float), a.step); tr, hr = course(rt, rx, a.step)
        if ta is None or tr is None: continue
        hri = np.interp(ta, tr, hr); da, dr = np.diff(ha), np.diff(hri); turn = float(np.degrees(np.sum(np.abs(dr))))
        c = float(np.corrcoef(da, dr)[0, 1]) if da.std() > 0 and dr.std() > 0 else float('nan'); s = theil_sen(da, dr)
        rows.append(dict(key=key, mode=clips[key]['camera_mode'], n=int(len(da)), corr=round(c, 3), slope=round(s, 3), abs_slope=round(abs(s), 3) if np.isfinite(s) else None, turn_deg=round(turn, 1)))
        print(f"{key} {clips[key]['camera_mode']:8s} n={len(da):4d} corr={c:+.2f} slope={s:+.3f} turn={turn:6.1f} deg")
    good = [r for r in rows if r['abs_slope'] is not None and abs(r['corr']) >= a.min_corr and r['turn_deg'] >= a.min_turn_deg]
    gains = np.array([r['abs_slope'] for r in good]); neg = sum(1 for r in good if r['slope'] < 0)
    med = float(np.median(gains)) if len(gains) else float('nan'); q = np.percentile(gains, [25, 75]).round(3).tolist() if len(gains) else []
    print(f'== {len(good)}/{len(rows)} clips usable (|corr|>={a.min_corr}, turn>={a.min_turn_deg} deg); median |gain| = {med:.3f}  IQR {q}; negative-slope (mirrored) clips: {neg}/{len(good)}')
    if a.out: json.dump(dict(run=str(a.run), step=a.step, median_gain=med, iqr=q, n_used=len(good), n_mirrored=neg, clips=rows), open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
