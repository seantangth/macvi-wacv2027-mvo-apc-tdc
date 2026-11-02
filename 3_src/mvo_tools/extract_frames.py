#!/usr/bin/env python3
"""PTS-aware frame extraction for feed-forward / streaming methods that consume image folders.

For each clip: build a uniform time grid at --fps using the RELEASED timing array, pick the decoded frame whose PTS
is nearest to each grid point (never the same frame twice), crop to the manifest's active_rect, optionally resize
(longest side --max-side), and write JPEGs named by decoded frame index plus frames_manifest.json
({key, fps, frames:[{file, decoded_index, time}], calibration (scaled to the written image)}).
convert_poses.py uses that manifest to map a method's per-frame poses back to decoded indices and released times.

Usage: extract_frames.py --dataset <manifest> --clips clip_005,clip_016 --fps 8 --out frames/ [--max-side 640] [--quality 95]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import mvo_tools  # noqa: F401
from public_dataset import asset_path, load_manifest, load_timestamps, select_clips


def pick_indices(times: np.ndarray, fps: float):
    grid = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    idx = np.searchsorted(times, grid); idx = np.clip(idx, 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1); use_prev = np.abs(times[prev] - grid) < np.abs(times[idx] - grid)
    idx = np.where(use_prev, prev, idx); idx = np.unique(idx)
    if idx[0] != 0: idx = np.concatenate([[0], idx])
    if idx[-1] != len(times) - 1: idx = np.concatenate([idx, [len(times) - 1]])
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='val'); ap.add_argument('--clips')
    ap.add_argument('--fps', type=float, default=8.0); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--max-side', type=int, default=0, help='resize so the longest side equals this (0 = keep)'); ap.add_argument('--quality', type=int, default=95); ap.add_argument('--clahe', type=float, default=0.0, help='apply CLAHE (clip limit) on the L channel before writing (leader used CLAHE on FLIR clips)')
    a = ap.parse_args(argv)
    man = load_manifest(a.dataset); root = a.dataset.resolve().parent
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; ft = load_timestamps(root, c); want = pick_indices(ft, a.fps); want_set = set(int(i) for i in want)
        cal = c['calibration']; x, y, w, h = cal['active_rect']; scale = 1.0
        if a.max_side and max(w, h) > a.max_side:
            scale = a.max_side / max(w, h)
        out = a.out / key; out.mkdir(parents=True, exist_ok=True); cap = cv2.VideoCapture(str(asset_path(root, c['video'])))
        frames = []; i = 0; written = 0
        while True:
            ok, im = cap.read()
            if not ok: break
            if i in want_set:
                im = im[y:y + h, x:x + w]
                if scale != 1.0: im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                if a.clahe > 0:
                    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB); lab[:, :, 0] = cv2.createCLAHE(clipLimit=float(a.clahe), tileGridSize=(8, 8)).apply(lab[:, :, 0]); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                fn = f'{i:06d}.jpg'; cv2.imwrite(str(out / fn), im, [cv2.IMWRITE_JPEG_QUALITY, a.quality]); frames.append(dict(file=fn, decoded_index=i, time=float(ft[i]))); written += 1
            i += 1
        cap.release()
        if i != len(ft): print(f'WARNING {key}: decoded {i} frames but timing has {len(ft)}', file=sys.stderr)
        calib = dict(fx=cal['fx'] * scale, fy=cal['fy'] * scale, cx=cal['cx'] * scale, cy=cal['cy'] * scale, width=int(round(w * scale)), height=int(round(h * scale)))
        json.dump(dict(key=key, fps=a.fps, camera_mode=c['camera_mode'], scale=scale, calibration=calib, frames=frames), open(out / 'frames_manifest.json', 'w'), indent=1)
        with open(out / 'intrinsics.txt', 'w') as fh: fh.write(f"{calib['fx']} {calib['fy']} {calib['cx']} {calib['cy']}\n")
        print(f'{key}: wrote {written} frames at ~{a.fps} fps (scale {scale:.3f}, {calib["width"]}x{calib["height"]})')


if __name__ == '__main__':
    main()
