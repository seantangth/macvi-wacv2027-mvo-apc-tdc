#!/usr/bin/env python3
"""Compare the outputs of the four GPU stages of a re-run (WORK dir) with the cached outputs that produced the submitted bundle (cache/gpu).
  windows  : per clip, difference of the chained yaw of the VGGT windows (deg): mean |d(yaw increment)| per 1 s and max |d(cumulative yaw)|
  probe    : focal-probe ratio per clip and the resulting gain
  flow     : RAFT band statistics: mean |d| of the flow vectors (px/s) and correlation of the magnitudes
  dpvo     : clip_112 DPVO trajectory: official drift of the re-run scored against the cached trajectory
Usage: compare_intermediates.py WORK CACHE_GPU"""
import json, sys, glob, os
from pathlib import Path
import numpy as np
import metrics
from mvo_tools.vggt_window_vo import chain_windows, heading_const_speed

def yaw_of(npz):
    z = np.load(npz); c2w, _ = chain_windows([w for w in z['c2w']], [int(s) for s in z['starts']], int(z['n'])); _, hd = heading_const_speed(z['times'], c2w, -1.0)
    return np.unwrap(np.asarray(hd, float)), np.asarray(z['times'], float)

def main():
    W, C = Path(sys.argv[1]), Path(sys.argv[2])
    print('== VGGT windows (chained yaw, re-run vs cache)')
    rows = []
    for f in sorted(glob.glob(str(C / 'vggt/windows/clip_*.npz'))):
        g = W / 'vggt/windows' / os.path.basename(f)
        if not g.is_file(): print('  missing', g.name); continue
        a, ta = yaw_of(f); b, tb = yaw_of(g); n = min(len(a), len(b))
        step = max(1, int(round(1.0 / np.median(np.diff(ta))))) if len(ta) > 3 else 1
        da = np.diff(a[:n:step]) - np.diff(b[:n:step]); rows.append((os.path.basename(f)[:-4], np.degrees(np.mean(np.abs(da))), np.degrees(np.max(np.abs(a[:n] - b[:n]))), len(a) == len(b)))
    for k, m1, m2, same in rows: print(f'  {k}: mean |d yaw increment| {m1:.4f} deg/s, max |d cumulative yaw| {m2:.3f} deg, same length {same}')
    if rows: print(f'  mean over {len(rows)} clips: {np.mean([r[1] for r in rows]):.4f} deg/s; worst cumulative {np.max([r[2] for r in rows]):.3f} deg')
    print('== focal probe (ratio re-run vs cache)')
    pa = json.load(open(C / 'vggt/focal_probe_test.json')); pb = json.load(open(W / 'vggt/focal_probe_test.json')) if (W / 'vggt/focal_probe_test.json').is_file() else {}
    d = [abs(pa[k]['ratio'] - pb[k]['ratio']) for k in pa if k in pb]
    for k in pa:
        if k in pb: print(f'  {k}: cache {pa[k]["ratio"]:.4f}  re-run {pb[k]["ratio"]:.4f}')
    if d: print(f'  max |d ratio| {max(d):.4f}, mean {np.mean(d):.4f}')
    print('== RAFT band statistics (re-run vs cache)')
    for f in sorted(glob.glob(str(C / 'flow/clip_*.npz'))):
        g = W / 'flow' / os.path.basename(f)
        if not g.is_file(): print('  missing', g.name); continue
        a, b = np.load(f), np.load(g); va, vb = a['vec'], b['vec']; n = min(len(va), len(vb)); ma, mb = a['mag'][:n].ravel(), b['mag'][:n].ravel(); ok = np.isfinite(ma) & np.isfinite(mb)
        c = float(np.corrcoef(ma[ok], mb[ok])[0, 1]) if ok.sum() > 10 else float('nan')
        print(f'  {os.path.basename(f)[:-4]}: pairs {len(va)}/{len(vb)}, mean |d vec| {np.nanmean(np.abs(va[:n] - vb[:n])):.4f} px/s, corr(mag) {c:.5f}')
    print('== DPVO clip_112 (re-run scored against the cached trajectory, official drift %)')
    fa, fb = C / 'dpvo112/predictions/clip_112.json', W / 'dpvo112/predictions/clip_112.json'
    if fb.is_file():
        A, B = json.load(open(fa)), json.load(open(fb))
        s = metrics.score_trajectory(np.asarray(B['times'], float), np.asarray(B['positions'], float), np.asarray(A['times'], float), np.asarray(A['positions'], float))
        print('  drift of re-run vs cache:', None if s is None else round(s['drift_pct'], 3), '%  (poses', len(A['times']), '/', len(B['times']), ')')
    else: print('  no DPVO re-run output')

if __name__ == '__main__':
    main()
