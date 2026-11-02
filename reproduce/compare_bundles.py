#!/usr/bin/env python3
"""Compare a reproduced JSON bundle with the bundle of the final entry.
(1) byte identity (SHA-256);  (2) per clip, the official drift metric of the reproduced trajectory scored AGAINST the original trajectory as reference
(Sim(3)-aligned, 100-800 m segments; 0 = identical).  The mean over the 20 clips is a direct, GT-free measure of the run-to-run deviation in the
metric's own units: the leaderboard score of the reproduction can differ from the original score by at most about this amount.
Usage: compare_bundles.py REPRODUCED.json EXPECTED.json"""
import hashlib, json, sys
import numpy as np
import metrics  # starter metrics.py (PYTHONPATH set by reproduce/common.sh)

def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def main():
    a, b = sys.argv[1], sys.argv[2]; A = json.load(open(a)); B = json.load(open(b)); ha, hb = sha(a), sha(b)
    print(f'reproduced sha256 {ha}\nexpected   sha256 {hb}\nbyte-identical: {ha == hb}')
    if set(A) != set(B): print('clip sets differ:', sorted(set(A) ^ set(B))); return 2
    rows = []
    for k in sorted(B):
        if A[k] == B[k]: rows.append((k, 0.0)); continue
        s = metrics.score_trajectory(np.asarray(A[k]['times'], float), np.asarray(A[k]['positions'], float), np.asarray(B[k]['times'], float), np.asarray(B[k]['positions'], float))
        rows.append((k, float('nan') if s is None else s['drift_pct']))
    for k, d in rows: print(f'  {k}: drift of reproduction vs original = {d:.4f} %')
    v = [d for _, d in rows if d == d]; print(f'mean deviation over {len(v)} clips = {np.mean(v):.4f} %   max = {np.max(v):.4f} %')
    return 0 if ha == hb else 1

if __name__ == '__main__':
    sys.exit(main())
