#!/usr/bin/env python3
"""Per-clip static-structure mask (boat bow, rails, radar dome, poles, burned-in logos/icons).

Idea (same discriminator the starter uses for pillarbox columns, applied per pixel): pixels that belong to the
boat or an overlay keep (nearly) the same value over the whole clip, while scene pixels change as the boat moves.
We sample N frames spread over the clip, compute the temporal std of grey values at reduced resolution, threshold,
clean with morphology, drop tiny blobs, and save a full-resolution mask PNG (255 = static, mask it out) plus a
preview. Uniform sky can also have low temporal std; that is harmless for VO (no features there) but we keep the
mask conservative by requiring a minimum blob size and dilating only a little.

Usage: python static_mask.py --video clip.mp4 --out masks/ [--n 64] [--thr 4] [--min-area 0.0004]
v4 (2026-10-18): region prior added after v3 over-masked calm water (clip_096) and cloud banks (clip_106).
       python static_mask.py --dataset release_manifest.json --out masks/ [--split val|train|test|all]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2


def compute_static_mask(video: str, n_samples: int = 64, work_w: int = 480, thr: float = 6.0,
                        min_area_frac: float = 0.0004, dilate_px: int = 4, min_tex: float = 10.0,
                        bottom_band: float = 0.72, corner_frac: float = 0.16, pole_frac: float = 0.6):
    """Static structure = pixels whose grey value is (nearly) constant over the whole clip (temporal std < thr)
    AND that lie in a locally textured area of the temporal-median image (so uniform sky/water is not flagged).
    No morphological opening (thin poles/rails survive); closing + small-blob removal + dilation only.
    NOTE (2026-10-18): an alternative 'static edge' detector (edges that stay sharp in the median image) was tried
    and rejected: during straight runs the horizon/shoreline is static in image space too, so it over-masked
    25-50% of the frame including scene content that VO needs."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f'cannot open {video}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = work_w / W; wh = (work_w, int(round(H * scale)))
    frames = []
    for idx in np.linspace(n * 0.03, n * 0.97, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, im = cap.read()
        if ok:
            frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), wh, interpolation=cv2.INTER_AREA).astype(np.float32))
    cap.release()
    if len(frames) < 8:
        raise ValueError(f'too few frames sampled from {video}')
    stack = np.stack(frames); tstd = stack.std(axis=0); med = np.median(stack, axis=0)
    gmed = cv2.magnitude(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    loc_tex = cv2.blur(gmed, (7, 7))
    static_raw = (tstd < thr) & (loc_tex > min_tex)
    # Region prior (v4): boat structure sits in the bottom band, overlays in the corners, and masts/poles are thin
    # full-height vertical structures. Calm water and cloud banks also look 'static' but must NOT be masked
    # (they are the only near-field parallax / horizon cues), so anything outside these regions is rejected.
    Hw, Ww = static_raw.shape; allow = np.zeros_like(static_raw)
    allow[int(Hw * bottom_band):, :] = True
    ch, cw = int(Hw * corner_frac), int(Ww * corner_frac)
    allow[:ch, :cw] = True; allow[:ch, Ww - cw:] = True; allow[Hw - ch:, :cw] = True; allow[Hw - ch:, Ww - cw:] = True
    col_static = static_raw.mean(axis=0) > pole_frac            # columns that are static over most of the height
    allow[:, col_static] = True
    static = (static_raw & allow).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, k, iterations=2)
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    keep = np.zeros_like(static); min_area = min_area_frac * static.size
    for i in range(1, ncc):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    if dilate_px > 0:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    full = cv2.resize(keep * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    stats_out = dict(frac_static=float(keep.mean()), n_blobs=int(ncc - 1), thr=thr, min_tex=min_tex, n_samples=len(frames),
                     tstd_median=float(np.median(tstd)), frac_static_raw=float(static_raw.mean()), n_pole_cols=int(col_static.sum()))
    preview = cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    sel = keep.astype(bool); preview[sel] = (0.4 * preview[sel] + np.array([0, 0, 153])).astype(np.uint8)
    return full, preview, stats_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video'); ap.add_argument('--dataset', type=Path); ap.add_argument('--split', default='all')
    ap.add_argument('--out', required=True, type=Path); ap.add_argument('--n', type=int, default=64)
    ap.add_argument("--thr", type=float, default=6.0); ap.add_argument('--min-area', type=float, default=0.0004)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.video:
        jobs.append((Path(a.video).stem, a.video))
    if a.dataset:
        import mvo_tools  # noqa: F401
        from public_dataset import load_manifest, select_clips, asset_path
        man = load_manifest(a.dataset); root = a.dataset.resolve().parent
        for c in select_clips(man, a.split):
            jobs.append((c['key'], str(asset_path(root, c['video']))))
    summary = {}
    for key, video in jobs:
        mask, preview, st = compute_static_mask(video, a.n, thr=a.thr, min_area_frac=a.min_area)
        cv2.imwrite(str(a.out / f'{key}.png'), mask); cv2.imwrite(str(a.out / f'{key}_preview.jpg'), preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        summary[key] = st; print(key, json.dumps(st))
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
