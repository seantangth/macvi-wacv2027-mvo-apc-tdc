#!/usr/bin/env python3
"""Ensemble of the two learned relative-speed models of the final entry (log-domain mean of the clip-median-normalised profiles):
  A = ridge regression on RAFT band statistics (learned_speed.py predict, fitted on the 46 training wide clips)
  B = gated dilated 1-D CNN (speednet.py --mode gate, trained on the 68 training clips, 5 seeds)
Clips without an A prediction use B alone.  Output: <out>/predictions/<clip>.json, a speed run (x = cumulative relative speed, 1 Hz) for policy_v004.py --speed-run.
Usage: ensemble_speed.py --ridge DIR --net DIR --out DIR   (DIR = run dir with predictions/)
(Moved from an inline command of the development session into a script; same arithmetic.)"""
import argparse, glob, json, os, sys
import numpy as np
import mvo_tools  # noqa: F401
from mvo_tools.fuse_heading_speed import speed_of

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ridge', required=True); ap.add_argument('--net', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = os.path.join(a.out, 'predictions'); os.makedirs(out, exist_ok=True); n = 0
    for f in sorted(glob.glob(os.path.join(a.net, 'predictions', '*.json'))):
        k = os.path.basename(f)[:-5]; g = os.path.join(a.ridge, 'predictions', k + '.json')
        b = json.load(open(f)); tb = np.array(b['times']); sb = speed_of(tb, np.array(b['positions']), 5.0)
        if os.path.isfile(g):
            r = json.load(open(g)); ta = np.array(r['times']); sa = np.interp(tb, ta, speed_of(ta, np.array(r['positions']), 5.0))
            la = np.log(np.maximum(sa / np.median(sa), 1e-3)); lb = np.log(np.maximum(sb / np.median(sb), 1e-3)); s = np.exp(0.5 * (la + lb))
        else: s = sb / np.median(sb)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(tb))])
        json.dump(dict(key=k, frame_indices=b['frame_indices'], times=tb.tolist(), positions=np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]).tolist()), open(os.path.join(out, k + '.json'), 'w')); n += 1
    print('ensemble runs', n)
    return 0

if __name__ == '__main__':
    sys.exit(main())
