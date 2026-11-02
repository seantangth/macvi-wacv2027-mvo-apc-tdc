#!/usr/bin/env python3
"""Create an artificial CPU-evaluation fixture. No video or real data is included."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'starter'))
from public_dataset import write_json


def make_example(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    key = 'clip_000'
    times = np.linspace(0, 30, 301)
    angle = np.linspace(0, 4, len(times))
    reference = np.column_stack((300*np.sin(angle), 300*(1-np.cos(angle)), np.zeros(len(times))))
    # Smooth, deliberately imperfect estimates make the scoring output nontrivial.
    estimate = reference + np.column_stack((np.sin(times), np.cos(times)-1, np.zeros(len(times))))
    clip = dict(key=key, split='val', camera_mode='synthetic-no-video',
                calibration=dict(fx=200., fy=200., cx=160., cy=120., width=320, height=240,
                                 container_width=320, container_height=240,
                                 active_rect=[0, 0, 320, 240], provenance='estimated'),
                video=f'videos/{key}.mp4', timestamps=f'timing/{key}.json',
                reference=f'references/{key}.json')
    write_json(output/'release_manifest.json', dict(schema_version=1, version='synthetic-evaluator-example',
               counts=dict(train=0, val=1, test=0, total=1), clips=[clip]))
    write_json(output/clip['timestamps'], dict(key=key, times=times.tolist()))
    write_json(output/clip['reference'], dict(key=key, times=times.tolist(), positions=reference.tolist()))
    write_json(output/'run'/'predictions'/f'{key}.json', dict(key=key,
               frame_indices=list(range(len(times))), times=times.tolist(), positions=estimate.tolist()))
    (output/'README.txt').write_text('Artificial evaluator fixture only. No video exists at the manifest video path.\n'
                                   'Not a dataset, inference example or real-data baseline.\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='a new directory')
    args = parser.parse_args()
    output = make_example(args.output)
    print(f'Created synthetic evaluator example at {output}. No video or real data is included.')
