#!/usr/bin/env python3
"""Windowed VGGT visual odometry ("vggt-heading-hybrid" style): run VGGT on overlapping windows of frames, chain the
per-window camera poses through the overlap, and output a trajectory.  Two outputs per clip: (a) chained VGGT
trajectory (positions), (b) heading-only variant re-integrated at constant speed (window scale is unreliable at sea,
the heading usually is not).

Chaining (v2, 2026-10-20): the window-to-window rotation is the orthogonal Procrustes fit of the overlap ROTATIONS
(robust on straight runs where camera centres are collinear and a centre-based Umeyama rotation is ambiguous);
scale + translation come from the overlap centres (scale clipped, 1.0 when the centres do not move).  Heading is
therefore independent of the (unreliable) window scale.  Per-window raw poses are saved (--save-windows) so the
chaining / heading extraction can be re-run offline on CPU (--from-windows) without the GPU.

Requires: pip install git+https://github.com/facebookresearch/vggt.git (weights facebook/VGGT-1B via HF, CC BY-NC).
Frames from extract_frames.py (frames_manifest.json).
Usage: vggt_window_vo.py --frames frames/clip_005 --timing timing/clip_005.json --out runs/vggt/raw/predictions/clip_005.json \
         --out-heading runs/vggt/heading/predictions/clip_005.json --save-windows runs/vggt/windows/clip_005.npz \
         [--window 48 --stride 24 --max-windows 0 --sign 1 --dry-run]
--dry-run: no model; synthesises window poses (CPU) to validate chaining.  --from-windows X.npz: re-chain saved windows.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import validate_prediction, write_json


def procrustes_rot(Rs_src, Rs_dst):
    """Rotation R minimising sum ||R Rs - Rd||_F  (R = argmax tr(R^T M), M = sum Rd Rs^T)."""
    M = np.zeros((3, 3))
    for Rs, Rd in zip(Rs_src, Rs_dst): M += Rd @ Rs.T
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R


def chain_windows(windows, starts, n_frames, min_ov=2):
    """windows: list of c2w (W_i,4,4); starts: first frame index per window. Returns (N,4,4) c2w chained + diag."""
    glob = [None] * n_frames; diag = dict(n_windows=len(windows), rigid_fallback=0, scales=[])
    for wi, (P, s) in enumerate(zip(windows, starts)):
        W = P.shape[0]
        if wi == 0:
            for k in range(W): glob[s + k] = P[k].copy()
            continue
        ov = [k for k in range(W) if glob[s + k] is not None]
        if len(ov) >= min_ov:
            R = procrustes_rot([P[k][:3, :3] for k in ov], [glob[s + k][:3, :3] for k in ov])
            src = np.array([P[k][:3, 3] for k in ov]); dst = np.array([glob[s + k][:3, 3] for k in ov])
            sm, dm = src.mean(0), dst.mean(0); ss = np.linalg.norm(src - sm); ds = np.linalg.norm(dst - dm)
            sc = float(np.clip(ds / ss, 0.05, 20.0)) if ss > 1e-6 and ds > 1e-6 else 1.0
            t = dm - sc * (R @ sm); mid = ov[len(ov) // 2]; diag['scales'].append(sc)
            for k in range(W):
                if glob[s + k] is None or k >= mid:
                    M = np.eye(4); M[:3, :3] = R @ P[k][:3, :3]; M[:3, 3] = sc * (R @ P[k][:3, 3]) + t; glob[s + k] = M
            continue
        diag['rigid_fallback'] += 1
        k0 = ov[0] if ov else 0
        if glob[s + k0] is None: glob[s + k0] = np.eye(4)
        A = glob[s + k0] @ np.linalg.inv(P[k0])
        for k in range(W):
            if glob[s + k] is None: glob[s + k] = A @ P[k]
    for i in range(n_frames):
        if glob[i] is None: glob[i] = glob[i - 1].copy() if i > 0 else np.eye(4)
    return np.stack(glob), diag


def yaw_series(c2w, forward_axis=2, down_axis=1):
    """Heading angle per frame: camera forward axis projected on the plane normal to the mean camera 'down' axis."""
    f = c2w[:, :3, forward_axis]
    g = np.mean(c2w[:, :3, down_axis], axis=0); g /= np.linalg.norm(g) + 1e-9
    f = f - np.outer(f @ g, g); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
    e1 = f[0] / (np.linalg.norm(f[0]) + 1e-9); e2 = np.cross(g, e1)
    return np.unwrap(np.arctan2(f @ e2, f @ e1))


def heading_const_speed(times, c2w, sign=1.0):
    hd = sign * yaw_series(c2w); dt = np.diff(times)
    x = np.cumsum(np.concatenate([[0], dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], dt * np.sin(hd[1:])]))
    return np.column_stack([x, y, np.zeros_like(x)]), hd


def run_vggt(images, device='cuda'):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    if not hasattr(run_vggt, 'model'):
        run_vggt.model = VGGT.from_pretrained('facebook/VGGT-1B').to(device).eval()
    model = run_vggt.model
    imgs = load_and_preprocess_images(images).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        agg, ps_idx = model.aggregator(imgs[None]); pose_enc = model.camera_head(agg)[-1]
        extri, intri = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
    E = extri[0].float().cpu().numpy()  # (W,3,4) world-to-camera
    c2w = np.tile(np.eye(4), (E.shape[0], 1, 1)); R = E[:, :, :3]; t = E[:, :, 3]
    c2w[:, :3, :3] = np.transpose(R, (0, 2, 1)); c2w[:, :3, 3] = -np.einsum('nji,nj->ni', R, t)
    return c2w


def synth_window(s, W):
    th = np.linspace(0, 0.3, W) + s * 0.01; P = np.tile(np.eye(4), (W, 1, 1))
    P[:, 0, 3] = 10 * np.cos(th) * (1 + 0.1 * (s % 3)); P[:, 2, 3] = 10 * np.sin(th) * (1 + 0.1 * (s % 3))
    for k in range(W):
        c, sn = np.cos(th[k]), np.sin(th[k]); P[k, :3, :3] = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    return P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, type=Path); ap.add_argument('--timing', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--window', type=int, default=48); ap.add_argument('--stride', type=int, default=24); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-windows', type=int, default=0, help='process only the first N windows (DRY smoke test); 0 = all')
    ap.add_argument('--out-heading', type=Path, help='also write the heading+constant-speed variant here'); ap.add_argument('--sign', type=float, default=1.0)
    ap.add_argument('--save-windows', type=Path, help='save raw per-window c2w poses (npz) for offline re-chaining')
    ap.add_argument('--from-windows', type=Path, help='skip the model: re-chain the windows saved by --save-windows')
    ap.add_argument('--diag', type=Path, help='write chaining / heading diagnostics JSON here')
    a = ap.parse_args(argv)
    fm = json.load(open(a.frames / 'frames_manifest.json')); frames = fm['frames']; key = fm['key']; ft = np.asarray(json.load(open(a.timing))['times'], float)
    files = [str(a.frames / f['file']) for f in frames]; n = len(files); fi = np.array([f['decoded_index'] for f in frames], int)
    t0 = time.time()
    if a.from_windows:
        z = np.load(a.from_windows); windows = [w for w in z['c2w']]; starts = [int(s) for s in z['starts']]
        if int(z['n']) != n: print(f'{key}: WARNING saved windows cover {int(z["n"])} frames, manifest has {n}', file=sys.stderr)
    else:
        W = min(a.window, n); starts = list(range(0, max(1, n - W + 1), a.stride))
        if starts[-1] + W < n: starts.append(n - W)
        if a.max_windows > 0: starts = starts[:a.max_windows]
        windows = []
        for s in starts:
            windows.append(synth_window(s, W) if a.dry_run else run_vggt(files[s:s + W]))
            print(f'{key}: window {s}/{n} done ({len(windows)}/{len(starts)}, {time.time() - t0:.0f}s)', flush=True)
        if a.save_windows:
            a.save_windows.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.save_windows, c2w=np.stack(windows), starts=np.array(starts), n=n, fi=fi, times=ft[fi], window=W, stride=a.stride)
    n_cov = starts[-1] + windows[-1].shape[0] if windows else 0
    c2w, dg = chain_windows(windows, starts, n)
    pos = c2w[:, :3, 3]
    pred = dict(key=key, frame_indices=fi.tolist(), times=ft[fi].tolist(), positions=pos.tolist()); validate_prediction(pred, key, ft); write_json(a.out, pred)
    print(f'{key}: wrote {a.out} ({n} poses, {len(windows)} windows, covered {n_cov}/{n} frames, {time.time() - t0:.0f}s)')
    hd = None
    if a.out_heading:
        hp, hd = heading_const_speed(ft[fi], c2w, a.sign); pred2 = dict(key=key, frame_indices=fi.tolist(), times=ft[fi].tolist(), positions=hp.tolist()); validate_prediction(pred2, key, ft); write_json(a.out_heading, pred2)
        print(f'{key}: wrote heading variant {a.out_heading} (total turn {np.sum(np.abs(np.diff(hd))) * 180 / np.pi:.0f} deg, net {(hd[-1] - hd[0]) * 180 / np.pi:.0f} deg)')
    if a.diag:
        a.diag.parent.mkdir(parents=True, exist_ok=True)
        hd = hd if hd is not None else yaw_series(c2w)
        json.dump(dict(key=key, n=n, n_windows=len(windows), covered=int(n_cov), rigid_fallback=dg['rigid_fallback'], scale_med=float(np.median(dg['scales'])) if dg['scales'] else None,
                       scale_min=float(np.min(dg['scales'])) if dg['scales'] else None, scale_max=float(np.max(dg['scales'])) if dg['scales'] else None,
                       total_turn_deg=float(np.sum(np.abs(np.diff(hd))) * 180 / np.pi), net_turn_deg=float((hd[-1] - hd[0]) * 180 / np.pi), seconds=time.time() - t0), open(a.diag, 'w'), indent=1)


if __name__ == '__main__':
    main()
