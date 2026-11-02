#!/usr/bin/env python3
"""Far-field (horizon-band) rotation tracker -> heading + constant-speed trajectory.  CPU only, streaming decode.

Motivation (oracle_per_clip.csv, 2026-10-19): with the TRUE heading and a constant speed the test-weighted drift is
~10.6% vs 24.1% for a straight line, so heading is the lever; DPVO's heading breaks on open water because its patches
lock onto water/wake.  Here only features in a band around the horizon (distant shore, skyline, masts, clouds) are
tracked with KLT (stage `track`, expensive, once per clip) and the camera rotation is estimated offline from the
stored track table (stage `solve`, cheap, so window length / thresholds / selection rules can be swept):
  consecutive grid frames (1/fps s): rotation-only model (Kabsch + RANSAC).  Exact for far features, but the parallax
      of mid-field structure (skyline at 1-2 km, 9 m/s) is 0.2-0.4 px per step: invisible to RANSAC, yet it
      integrates to -10..-14 deg/min (clip_090, v1).  Used only to fill in between window estimates.
  windows of --win-s seconds from chained KLT tracks: parallax becomes measurable (2-5 px) so a model selection is
      possible per window (ORB-SLAM style):
        rotw : rotation-only on the window correspondences
        essw : 5-point essential matrix (cv2.findEssentialMat), the R candidate closest to the chained small-step
               rotation (cheirality is useless for far points)
        kd0w : epipolar model with KNOWN translation direction (focus of expansion at cx on the horizon)
        sel  : rotw if the rotation-only model keeps >= --sel-ratio of the correspondences as inliers (no parallax),
               otherwise essw
The yaw increment is taken about gravity, with gravity from the detected horizon line (K^T l), camera-y as fallback.
Heading is integrated and re-integrated at constant speed (the metric's Sim(3) absorbs the scale).

Outputs: <out>/tracks/<key>.npz (track table + horizon), <out>/<est>/predictions/<key>.json for est in ESTS,
<out>/diag/<key>.json (confidence: horizon fraction, inlier counts, held windows, selection fractions), <out>/diag/<key>.npz.

Usage: PYTHONPATH=3_src python3 3_src/mvo_tools/horizon_yaw_vo.py --dataset <manifest> --split val --out runs/hyaw --stage both \
          [--fps 12] [--win-s 1.0] [--work-width 960] [--mask-dir 1_data/processed/masks_v4] [--clips a,b] [--shard i/n]
       ... --stage solve --out runs/hyaw --solve-out runs/hyaw_w2 --win-s 2.0   (re-solve from stored tracks)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import mvo_tools  # noqa: F401
from public_dataset import asset_path, load_manifest, load_timestamps, select_clips, validate_prediction, write_json
from mvo_tools.extract_frames import pick_indices

LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
ESTS = ('rot', 'rotw', 'essw', 'kd0w', 'sel')
WIN_ESTS = ('rotw', 'essw', 'kd0w', 'sel')


# ----------------------------------------------------------------------------------------------------- horizon
def detect_horizon(g, prev, lo=0.08, hi=0.80, max_deg=20.0):
    """Longest near-horizontal Hough segment in rows [lo,hi]*H.  Returns (row_at_center, angle_rad, line_homog, found)."""
    H, W = g.shape
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    edges[: int(lo * H)] = 0; edges[int(hi * H):] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=int(0.12 * W), minLineLength=int(0.30 * W), maxLineGap=int(0.04 * W))
    best = None
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if abs(ang) > np.radians(max_deg):
                continue
            L = np.hypot(x2 - x1, y2 - y1)
            if best is None or L > best[0]:
                best = (L, x1, y1, x2, y2, ang)
    if best is None:
        return prev[0], prev[1], prev[2], False
    _, x1, y1, x2, y2, ang = best
    row = y1 + (y2 - y1) * ((W / 2 - x1) / ((x2 - x1) or 1e-9))
    l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
    return float(row), float(ang), l, True


def band_mask(shape, row, ang, above, below):
    H, W = shape; t = np.tan(ang); x = np.array([0, W]); yc = row + (x - W / 2) * t
    top = yc - above * H; bot = yc + below * H
    poly = np.array([[0, top[0]], [W, top[1]], [W, bot[1]], [0, bot[0]]], np.int32)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [poly], 255); return m


def gravity_from_line(l, K):
    n = K.T @ np.asarray(l, float); n = n / (np.linalg.norm(n) + 1e-12)
    return n if n[1] > 0 else -n  # 'down' in camera frame (y down)


# ---------------------------------------------------------------------------------------------- rotation solvers
def kabsch(b1, b2):
    Hm = b1.T @ b2; U, _, Vt = np.linalg.svd(Hm); d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T  # R b1 ~ b2


def ransac_rot(b1, b2, p2, K, thr_px, rng, iters=200):
    n = len(b1); best_inl = None; best_cnt = 0
    def score(R):
        pr = (K @ (R @ b1.T)).T; pr = pr[:, :2] / pr[:, 2:3]; err = np.linalg.norm(pr - p2, axis=1); inl = err < thr_px; return inl, int(inl.sum())
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False); inl, cnt = score(kabsch(b1[idx], b2[idx]))
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
            if cnt > 0.95 * n:
                break
    if best_cnt < 3:
        return None, np.zeros(n, bool)
    R = kabsch(b1[best_inl], b2[best_inl]); inl, _ = score(R)
    if inl.sum() >= 3:
        R = kabsch(b1[inl], b2[inl])
    return R, inl


def rodrigues(w):
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1)); return R


def rotvec(R):
    r, _ = cv2.Rodrigues(R); return r[:, 0]


def gn_known_d(m, b2, R=None, iters=3):
    """Gauss-Newton for R minimising sum (b2_i . R m_i)^2, m_i = b1_i x d.  Small-angle update R <- exp(dw) R."""
    R = np.eye(3) if R is None else R
    for _ in range(iters):
        Rm = (R @ m.T).T; r = np.einsum('ij,ij->i', b2, Rm); A = np.cross(Rm, b2)
        dw, *_ = np.linalg.lstsq(A, -r, rcond=None)
        R = rodrigues(dw) @ R
    return R


def residual_known_d(m, b2, R, f, floor=np.sin(np.radians(3))):
    Rm = (R @ m.T).T; r = np.einsum('ij,ij->i', b2, Rm)
    return np.abs(r) / np.maximum(np.linalg.norm(m, axis=1), floor) * f  # ~ pixels off the epipolar plane


def ransac_known_d(b1, b2, d, f, thr_px, rng, iters=150):
    m = np.cross(b1, d); n = len(b1); best_inl = None; best_cnt = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False); R = gn_known_d(m[idx], b2[idx], iters=3)
        inl = residual_known_d(m, b2, R, f) < thr_px; cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
            if cnt > 0.95 * n:
                break
    if best_cnt < 3:
        return None, np.zeros(n, bool)
    R = gn_known_d(m[best_inl], b2[best_inl], iters=4); inl = residual_known_d(m, b2, R, f) < thr_px
    if inl.sum() >= 3:
        R = gn_known_d(m[inl], b2[inl], R=R, iters=2)
    return R, inl


def essential_R(q1, q2, K, thr_px, R_ref, min_inl):
    try:
        E, me = cv2.findEssentialMat(q1, q2, K, method=cv2.RANSAC, prob=0.999, threshold=thr_px)
    except cv2.error:
        return None, 0
    if E is None or E.shape != (3, 3) or me is None or int((me > 0).sum()) < min_inl:
        return None, 0
    R1, R2, _ = cv2.decomposeEssentialMat(E)
    R = min((R1, R2), key=lambda Rx: np.linalg.norm(rotvec(Rx @ R_ref.T)))  # candidate closest to the chained small-step rotation
    return R, int((me > 0).sum())


# ------------------------------------------------------------------------------------------- stage 1: tracking
def clip_geometry(c, a):
    cal = c['calibration']; x0, y0, w, h = cal['active_rect']; s = a.work_width / w; W, H = int(round(w * s)), int(round(h * s))
    K = np.array([[cal['fx'] * s, 0, cal['cx'] * s], [0, cal['fy'] * s, cal['cy'] * s], [0, 0, 1]])
    return (x0, y0, w, h), W, H, K


def track_clip(c, root, a, mask_dir):
    key = c['key']; ft = load_timestamps(root, c); grid = pick_indices(ft, a.fps); gset = {int(i): k for k, i in enumerate(grid)}
    (x0, y0, w, h), W, H, K = clip_geometry(c, a)
    static = None
    if mask_dir is not None and (mask_dir / f'{key}.png').is_file():
        mk = cv2.imread(str(mask_dir / f'{key}.png'), cv2.IMREAD_GRAYSCALE)[y0:y0 + h, x0:x0 + w]
        static = cv2.dilate(cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8)) > 0
    cap = cv2.VideoCapture(str(asset_path(root, c['video'])))
    n = len(grid); t0 = time.time()
    hz_rows = np.full(n, np.nan); hz_ang = np.zeros(n); hz_found = np.zeros(n, bool); hz_lines = np.zeros((n, 3))
    hz = (0.45 * H, 0.0, np.array([0.0, 1.0, -0.45 * H])); n_feat = []
    T_frame, T_id, T_xy = [], [], []
    prev = None; i = 0; ids = np.zeros(0, np.int32); pts = np.zeros((0, 1, 2), np.float32); next_id = 0
    while True:
        if i in gset:
            ok, im = cap.read()
        else:
            ok = cap.grab(); im = None
        if not ok:
            break
        if i in gset:
            k = gset[i]; g = cv2.cvtColor(im[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
            if getattr(a, 'clahe', 0.0) > 0: g = cv2.createCLAHE(clipLimit=float(a.clahe), tileGridSize=(8, 8)).apply(g)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            row, ang, l, found = detect_horizon(g, hz); hz = (row, ang, l); hz_rows[k] = row; hz_ang[k] = ang; hz_found[k] = found; hz_lines[k] = l
            if prev is not None and len(pts) > 0:
                p2, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
                p1b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p2, None, **LK)
                fb = np.linalg.norm(p1b - pts, axis=2)[:, 0]; ok_ = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 0.5)
                ok_ &= (p2[:, 0, 0] > 2) & (p2[:, 0, 0] < W - 3) & (p2[:, 0, 1] > 2) & (p2[:, 0, 1] < H - 3)
                q1 = pts[ok_, 0, :]; q2 = p2[ok_, 0, :]; kept = ids[ok_]
                if len(q1) >= 8:
                    dd = np.linalg.norm(q2 - q1, axis=1); med = np.median(dd)
                    if med > 2.0:  # static overlays / own-vessel structure: zero motion while the far field moves
                        keep = dd > 0.3; q2, kept = q2[keep], kept[keep]
                ids = kept; pts = q2.astype(np.float32).reshape(-1, 1, 2)
            else:
                ids = np.zeros(0, np.int32); pts = np.zeros((0, 1, 2), np.float32)
            if len(pts) < a.max_feat:  # replenish tracks in the horizon band
                fm = band_mask((H, W), row, ang, a.above, a.below)
                if static is not None:
                    fm[static] = 0
                fm[:, :4] = 0; fm[:, -4:] = 0
                for x, y in pts[:, 0, :]:
                    cv2.circle(fm, (int(x), int(y)), 8, 0, -1)
                newp = cv2.goodFeaturesToTrack(g, a.max_feat - len(pts), 0.01, 6, mask=fm, blockSize=5)
                if newp is not None:
                    pts = np.concatenate([pts, newp.astype(np.float32)]); ids = np.concatenate([ids, np.arange(next_id, next_id + len(newp), dtype=np.int32)]); next_id += len(newp)
            n_feat.append(len(pts)); T_frame.append(np.full(len(ids), k, np.int32)); T_id.append(ids.copy()); T_xy.append(pts[:, 0, :].copy())
            prev = g
        i += 1
    cap.release()
    (a.out / 'tracks').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'tracks' / f'{key}.npz', grid=grid.astype(np.int32), t=ft[grid], hz_row=hz_rows, hz_ang=hz_ang, hz_found=hz_found, hz_line=hz_lines,
                        frame=np.concatenate(T_frame) if T_frame else np.zeros(0, np.int32), tid=np.concatenate(T_id) if T_id else np.zeros(0, np.int32),
                        xy=np.concatenate(T_xy) if T_xy else np.zeros((0, 2), np.float32), decoded=i, W=W, H=H, K=K, feat_med=float(np.median(n_feat)) if n_feat else 0.0, seconds=time.time() - t0)
    return dict(key=key, decoded=i, n_grid=n, horizon_frac=float(hz_found.mean()), feat_med=float(np.median(n_feat)) if n_feat else 0.0, seconds=round(time.time() - t0, 1))


# -------------------------------------------------------------------------------------------- stage 2: solving
def solve_clip(c, root, a, rng):
    key = c['key']; z = np.load(a.out / 'tracks' / f'{key}.npz'); ft = load_timestamps(root, c)
    grid = z['grid']; t = z['t']; n = len(grid); K = z['K'].astype(np.float64).copy()
    fs = float(getattr(a, 'fx_scale', 1.0) or 1.0)
    if fs != 1.0: K[0, 0] *= fs; K[1, 1] *= fs  # focal override (zoomed PTZ camera: nominal calibration too wide)
    Kinv = np.linalg.inv(K); f = float(K[0, 0]); W = int(z['W']); cx = K[0, 2]
    hz_rows, hz_ang, hz_found, hz_lines = z['hz_row'], z['hz_ang'], z['hz_found'], z['hz_line']
    L = max(2, int(round(a.win_s * a.fps))); out = a.solve_out or a.out
    frame, tid, xy = z['frame'], z['tid'], z['xy'].astype(np.float64)
    starts = np.searchsorted(frame, np.arange(n + 1))  # frame is sorted (appended per grid frame)
    def table(k):
        s0, s1 = starts[k], starts[k + 1]; return tid[s0:s1], xy[s0:s1]
    def matches(k0, k1):
        i0, x0 = table(k0); i1, x1 = table(k1); common, a0, a1 = np.intersect1d(i0, i1, return_indices=True); return x0[a0], x1[a1]
    bear = lambda q: (lambda b: b / np.linalg.norm(b, axis=1, keepdims=True))((Kinv @ np.column_stack([q, np.ones(len(q))]).T).T)
    def d_foe(k):
        row, ang = hz_rows[k], hz_ang[k]; v = row + (cx - W / 2) * np.tan(ang); dv = Kinv @ np.array([cx, v, 1.0]); return dv / np.linalg.norm(dv)
    def up_at(k):
        return -gravity_from_line(hz_lines[k], K) if hz_found[: k + 1].any() else np.array([0.0, -1.0, 0.0])
    # ---- consecutive rotation-only increments
    Rc = [None] * n; dpsi = np.zeros(n); inl_c = []
    for k in range(1, n):
        q1, q2 = matches(k - 1, k)
        if len(q1) >= 8:
            Rr, inr = ransac_rot(bear(q1), bear(q2), q2, K, a.thr_px, rng)
            if Rr is not None and inr.sum() >= a.min_inl:
                Rc[k] = Rr; inl_c.append(int(inr.sum())); dpsi[k] = a.sign * float(rotvec(Rr.T) @ up_at(k))
    held_c = sum(1 for k in range(1, n) if Rc[k] is None)
    # ---- window estimates at k = L, 2L, ... (stride L), model selection
    yaw = {e: np.zeros(n) for e in ESTS}; yaw['rot'] = np.cumsum(dpsi)
    inl_w = {e: [] for e in ('rotw', 'essw', 'kd0w')}; held_w = {e: 0 for e in WIN_ESTS}; n_win = 0; n_sel_rot = 0; n_has = 0
    for e in WIN_ESTS:
        yaw[e][:L] = yaw['rot'][:L]
    for k in range(L, n, L):
        k0 = k - L; n_win += 1; dpw = {}
        Rchain = np.eye(3)
        for j in range(k0 + 1, k + 1):
            if Rc[j] is not None:
                Rchain = Rc[j] @ Rchain
        q1, q2 = matches(k0, k)
        if len(q1) >= 8:
            n_has += 1; b1, b2 = bear(q1), bear(q2); up = up_at(k0)
            Rr, inr = ransac_rot(b1, b2, q2, K, a.thr_px_win, rng); r_ratio = inr.mean() if Rr is not None else 0.0
            if Rr is not None and inr.sum() >= a.min_inl:
                dpw['rotw'] = a.sign * float(rotvec(Rr.T) @ up); inl_w['rotw'].append(int(inr.sum()))
            Re, ne = essential_R(q1, q2, K, a.thr_px_win, Rchain, a.min_inl)
            if Re is not None:
                dpw['essw'] = a.sign * float(rotvec(Re.T) @ up); inl_w['essw'].append(ne)
            Rk, ink = ransac_known_d(b1, b2, d_foe(k0), f, a.thr_px_win, rng)
            if Rk is not None and ink.sum() >= a.min_inl:
                dpw['kd0w'] = a.sign * float(rotvec(Rk.T) @ up); inl_w['kd0w'].append(int(ink.sum()))
            if 'rotw' in dpw and (r_ratio >= a.sel_ratio or 'essw' not in dpw):
                dpw['sel'] = dpw['rotw']; n_sel_rot += 1
            elif 'essw' in dpw:
                dpw['sel'] = dpw['essw']
        for e in WIN_ESTS:
            base = yaw['rot'][k0 + 1:k + 1] - yaw['rot'][k0]  # small-step shape inside the window
            if e in dpw:
                corr = (dpw[e] - base[-1]) * (np.arange(1, L + 1) / L)  # distribute the window correction uniformly
                yaw[e][k0 + 1:k + 1] = yaw[e][k0] + base + corr
            else:
                held_w[e] += 1; yaw[e][k0 + 1:k + 1] = yaw[e][k0] + base
    k_last = (n - 1) // L * L
    for e in WIN_ESTS:  # tail after the last full window: small-step shape
        yaw[e][k_last + 1:] = yaw[e][k_last] + (yaw['rot'][k_last + 1:] - yaw['rot'][k_last])
    dt = np.diff(t)
    for e in ESTS:
        ps = yaw[e]; x = np.cumsum(np.concatenate([[0], dt * np.cos(ps[1:])])); y = np.cumsum(np.concatenate([[0], dt * np.sin(ps[1:])]))
        pred = dict(key=key, frame_indices=[int(v) for v in grid], times=t.tolist(), positions=np.column_stack([x, y, np.zeros_like(x)]).tolist())
        validate_prediction(pred, key, ft); write_json(out / e / 'predictions' / f'{key}.json', pred)
    diag = dict(key=key, mode=c['camera_mode'], n_grid=n, decoded=int(z['decoded']), L=L, horizon_frac=float(hz_found.mean()), horizon_row_med=float(np.nanmedian(hz_rows) / int(z['H'])),
                feat_med=float(z['feat_med']), win_frac=n_has / max(n_win, 1),
                rot_inl_med=float(np.median(inl_c)) if inl_c else 0.0, rot_held_frac=held_c / max(n - 1, 1),
                **{f'{e}_inl_med': (float(np.median(inl_w[e])) if inl_w[e] else 0.0) for e in inl_w},
                **{f'{e}_held_frac': held_w[e] / max(n_win, 1) for e in held_w}, sel_rot_frac=n_sel_rot / max(n_win, 1),
                **{f'{e}_total_turn_deg': float(np.degrees(np.abs(np.diff(yaw[e])).sum())) for e in ESTS},
                **{f'{e}_net_turn_deg': float(np.degrees(yaw[e][-1] - yaw[e][0])) for e in ESTS}, track_seconds=float(z['seconds']))
    (out / 'diag').mkdir(parents=True, exist_ok=True); write_json(out / 'diag' / f'{key}.json', diag)
    np.savez_compressed(out / 'diag' / f'{key}.npz', t=t, hz_row=hz_rows, hz_found=hz_found, **{f'yaw_{e}': yaw[e] for e in ESTS})
    return diag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='val'); ap.add_argument('--clips')
    ap.add_argument('--out', required=True, type=Path); ap.add_argument('--solve-out', type=Path, help='where solve writes (default: --out)')
    ap.add_argument('--stage', default='both', choices=['track', 'solve', 'both'])
    ap.add_argument('--fps', type=float, default=12.0); ap.add_argument('--work-width', type=int, default=960); ap.add_argument('--clahe', type=float, default=0.0, help='track stage: CLAHE clip limit on the grayscale frame (0 = off; leader used CLAHE on FLIR)')
    ap.add_argument('--mask-dir', type=Path); ap.add_argument('--max-feat', type=int, default=600); ap.add_argument('--thr-px', type=float, default=1.0)
    ap.add_argument('--thr-px-win', type=float, default=1.5); ap.add_argument('--win-s', type=float, default=1.0); ap.add_argument('--sel-ratio', type=float, default=0.85)
    ap.add_argument('--min-inl', type=int, default=15); ap.add_argument('--sign', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--fx-scale', type=float, default=1.0, help='solve stage: multiply fx, fy of the stored K (per-mode zoom correction)')
    ap.add_argument('--above', type=float, default=0.25, help='band above the horizon (fraction of H)'); ap.add_argument('--below', type=float, default=0.015)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--shard', default='0/1', help='i/n: process every n-th clip starting at i'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent; rng = np.random.default_rng(a.seed); cv2.setNumThreads(a.threads)
    out = a.solve_out or a.out
    for e in ESTS:
        (out / e / 'predictions').mkdir(parents=True, exist_ok=True)
    si, sn = (int(v) for v in a.shard.split('/'))
    for j, c in enumerate(select_clips(man, a.split, a.clips)):
        if j % sn != si:
            continue
        key = c['key']; tr = a.out / 'tracks' / f'{key}.npz'
        if a.stage in ('track', 'both'):
            if tr.is_file() and not a.force:
                print(key, 'tracks exist, skip', flush=True)
            else:
                print(json.dumps(dict(stage='track', **track_clip(c, root, a, a.mask_dir))), flush=True)
        if a.stage in ('solve', 'both'):
            if not tr.is_file():
                print(key, 'no tracks', flush=True); continue
            print(json.dumps(solve_clip(c, root, a, rng)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
