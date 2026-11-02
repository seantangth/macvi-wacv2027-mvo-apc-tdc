#!/usr/bin/env python3
"""Re-chain saved VGGT window outputs (vggt_window_vo.py --save-windows) into a heading run with the yaw sign flipped (sign -1),
exactly as done inline in the development script build_v006.sh.  Output: <out>/predictions/<clip>.json (constant-speed heading trajectory).
Usage: windows_to_heading.py --windows DIR_WITH_NPZ --out OUT [--sign -1] [--min-clip 92]"""
import argparse, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import write_json
from mvo_tools.vggt_window_vo import chain_windows, heading_const_speed

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--windows', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sign', type=float, default=-1.0); ap.add_argument('--min-clip', type=int, default=0); a = ap.parse_args()
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True)
    for f in sorted(a.windows.glob('clip_*.npz')):
        key = f.stem
        if int(key.split('_')[1]) < a.min_clip: continue
        z = np.load(f); c2w, _ = chain_windows([w for w in z['c2w']], [int(s) for s in z['starts']], int(z['n']))
        hp, hd = heading_const_speed(z['times'], c2w, a.sign)
        write_json(a.out / 'predictions' / f'{key}.json', dict(key=key, frame_indices=[int(v) for v in z['fi']], times=z['times'].tolist(), positions=hp.tolist()))
        print(key, 'heading written, total turn %.0f deg' % (np.sum(np.abs(np.diff(hd))) * 180 / np.pi))
    return 0

if __name__ == '__main__':
    sys.exit(main())
