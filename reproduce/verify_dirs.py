#!/usr/bin/env python3
"""Compare the prediction JSONs of two run dirs clip by clip. Usage: verify_dirs.py A B [clip,clip,...]  -> bytes-identical count, or max abs position difference."""
import json, sys
from pathlib import Path
import numpy as np
A, B = Path(sys.argv[1]) / 'predictions', Path(sys.argv[2]) / 'predictions'; want = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None
same = diff = 0; worst = 0.0; rows = []
for f in sorted(A.glob('clip_*.json')):
    if want and f.stem not in want: continue
    g = B / f.name
    if not g.is_file(): rows.append((f.stem, 'missing in B')); continue
    if f.read_bytes() == g.read_bytes(): same += 1; continue
    a, b = json.load(open(f)), json.load(open(g)); diff += 1
    d = float(np.max(np.abs(np.asarray(a['positions']) - np.asarray(b['positions'])))) if len(a['positions']) == len(b['positions']) else float('inf'); worst = max(worst, d); rows.append((f.stem, f'max|dpos|={d:.3g}'))
print(f'{A.parent.name} vs {B.parent.name}: identical {same}, different {diff}' + (f', worst max|dpos| {worst:.3g}' if diff else '')); [print('  ', r) for r in rows[:8]]
