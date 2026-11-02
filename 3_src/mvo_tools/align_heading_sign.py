#!/usr/bin/env python3
"""Mirror a heading run per clip so its yaw sign matches a reference run (E020, 2026-10-24).

Why: feed-forward 3D models (VGGT / DA3) return trajectories that are planar mirror images of the tracker's; the
official Sim(3) alignment absorbs a mirror for a single run, but fusing/median-ing increments of opposite sign cancels
every turn (dev 075 went 4.7 -> 39.8 when fused unaligned).  Per clip: sample the course every --step seconds, take the
correlation of yaw increments with the reference; if negative, flip y.

Usage: align_heading_sign.py --ref runs/v006 --run runs/ff_hconst_g --out runs/ff_hconst_g_al [--clips a,b] [--step 2]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def course(t, pos, step):
    tt = np.arange(t[0], t[-1], step)
    if len(tt) < 4: return None, None
    x = np.interp(tt, t, pos[:, 0]); y = np.interp(tt, t, pos[:, 1])
    return tt[1:], np.unwrap(np.arctan2(np.diff(y), np.diff(x)))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--ref', required=True, type=Path); ap.add_argument('--run', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--clips'); ap.add_argument('--step', type=float, default=2.0)
    a = ap.parse_args(argv); (a.out / 'predictions').mkdir(parents=True, exist_ok=True)
    want = set(a.clips.split(',')) if a.clips else None; flips = []; done = []
    for f in sorted((a.run / 'predictions').glob('clip_*.json')):
        key = f.stem
        if want and key not in want: continue
        rf = a.ref / 'predictions' / f.name
        if not rf.is_file(): print(f'{key}: no reference, copied unchanged', file=sys.stderr); (a.out / 'predictions' / f.name).write_bytes(f.read_bytes()); continue
        d = json.load(open(f)); r = json.load(open(rf))
        ta, ha = course(np.asarray(d['times'], float), np.asarray(d['positions'], float), a.step)
        tr, hr = course(np.asarray(r['times'], float), np.asarray(r['positions'], float), a.step)
        c = float('nan')
        if ta is not None and tr is not None:
            hri = np.interp(ta, tr, hr); da, dr = np.diff(ha), np.diff(hri)
            if da.std() > 0 and dr.std() > 0: c = float(np.corrcoef(da, dr)[0, 1])
        pos = np.asarray(d['positions'], float)
        if np.isfinite(c) and c < 0: pos[:, 1] *= -1; flips.append(key)
        d['positions'] = pos.tolist(); json.dump(d, open(a.out / 'predictions' / f.name, 'w')); done.append((key, round(c, 3)))
    json.dump(dict(ref=str(a.ref), run=str(a.run), step=a.step, corr=dict(done), flipped=flips), open(a.out / 'align_manifest.json', 'w'), indent=1)
    print(f'aligned {len(done)} clips, flipped {len(flips)}: {" ".join(flips)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
