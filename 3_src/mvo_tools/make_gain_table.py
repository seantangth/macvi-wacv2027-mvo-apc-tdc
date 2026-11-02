#!/usr/bin/env python3
"""Per-clip yaw gain of the flir-VIS heading from the VGGT focal probe:  gain = clip(0.87 / ratio, 0.6, 1.0),  ratio = VGGT focal estimate / nominal focal.
0.87 is the typical probe ratio of the 8 training-split flir-VIS clips (fitted on the train split, E006c/d).  Manual override (disclosed in the factsheet):
clip_098 has a mid-clip optical zoom (~2x) which makes the probe ratio ambiguous (computed 0.6); a compromise value 0.78 was fixed by hand.
Usage: make_gain_table.py --probe focal_probe.json --out gain_test.json [--override clip_098=0.78]"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probe', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--override', nargs='*', default=['clip_098=0.78']); ap.add_argument('--clips', default='clip_093,clip_094,clip_095,clip_096,clip_098,clip_099,clip_104,clip_107,clip_108,clip_109,clip_111'); a = ap.parse_args()
    probe = json.load(open(a.probe)); ov = dict(x.split('=') for x in a.override); g = {}
    for k in a.clips.split(','): g[k] = float(ov[k]) if k in ov else round(float(np.clip(0.87 / probe[k]['ratio'], 0.6, 1.0)), 3)   # the table used for the final entry is rounded to 3 decimals
    json.dump(g, open(a.out, 'w'), indent=1); print({k: round(v, 3) for k, v in g.items()})

if __name__ == '__main__':
    main()
