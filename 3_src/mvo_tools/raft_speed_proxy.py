#!/usr/bin/env python3
"""E025-B (2026-10-25): turn raft_water_flow.py band statistics into a relative-speed run (challenge prediction format),
usable as a second, DPVO-independent witness for the water-speed gate and as a speed source for policy_v004 --speed-run.

Per frame pair (10 fps): far-field flow = median [u,v] of the band just above the horizon (distant scene ~ pure rotation)
is subtracted from every water band (de-rotation).  Water bands = bands whose top edge is below horizon + 0.08 H; only the
LEFT and RIGHT column thirds are used (the own bow sits bottom-centre); bands with > --zmax near-static pixels are dropped.
For a planar water surface the forward-motion flow grows like ((yc - y_h)/H)^2, so |flow| / ((yc - y_h)/H)^2 is ~ one scalar
per frame: the proxy = median over (band, side).  Quality: lr_agree = fraction of frames whose left/right medians agree within 2x;
valid_frac = fraction of frames with a proxy.  Rolling median (7) + --smooth-s mean, then a 1 Hz grid, positions = cumsum.
Horizon row per clip from --diag-dir/<key>.json (horizon_row_med, fraction) else --tracks-dir/<key>.npz (hz_row/H) else --horizon.
Usage: raft_speed_proxy.py --flow-dir runs/e025/flow --dataset <manifest> --split test --clips a,b --out <run dir> [--diag-dir d1 d2] [--smooth-s 5]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa
from public_dataset import load_manifest, select_clips, write_json
from mvo_tools.fuse_heading_speed import smooth


def horizon_frac(diag_dirs, key, default, tracks_dirs=()):
    for d in diag_dirs:
        f = Path(d) / f'{key}.json'
        if f.is_file():
            try:
                v = json.load(open(f)).get('horizon_row_med')
                if v is not None and 0.05 < float(v) < 0.95: return float(v), str(f)
            except Exception: pass
    for d in tracks_dirs:  # horizon tracker dumps (e.g. 5_outputs/hyaw/e023_h/tracks): hz_row per frame at height H
        f = Path(d) / f'{key}.npz'
        if f.is_file():
            try:
                z = np.load(f, allow_pickle=True); v = float(np.nanmedian(np.asarray(z['hz_row'], float))) / float(z['H'])
                if 0.05 < v < 0.95: return v, str(f)
            except Exception: pass
    return default, 'default'


def proxy_from_npz(z, yh_frac, zmax=0.5, margin=0.08, nearest=0):
    t = z['t']; vec = z['vec']; zf = z['zfrac']; rows = z['rows']; H = float(z['height'])
    T, R = vec.shape[0], vec.shape[1]; yc = 0.5 * (rows[:-1] + rows[1:]); yh = yh_frac * H
    above = np.where(yc < yh)[0]; far_i = int(above[-1]) if len(above) else -1
    water = [i for i in range(R) if rows[i] >= yh + margin * H]
    if nearest > 0: water = water[-nearest:]   # nearest bands = largest flow = least sub-pixel noise
    out = np.full(T, np.nan); lr = np.full(T, np.nan); absmag = []
    for k in range(T):
        far = np.nanmedian(vec[k, far_i, :, :], axis=0) if far_i >= 0 else np.zeros(2)
        if not np.all(np.isfinite(far)): far = np.zeros(2)
        vals = {0: [], 2: []}
        for i in water:
            for side in (0, 2):
                v = vec[k, i, side]; 
                if not np.all(np.isfinite(v)) or (np.isfinite(zf[k, i, side]) and zf[k, i, side] > zmax): continue
                d = v - far; m = float(np.hypot(d[0], d[1])); vals[side].append(m / ((yc[i] - yh) / H) ** 2); absmag.append(float(np.hypot(v[0], v[1])))
        allv = vals[0] + vals[2]
        if len(allv) >= 2: out[k] = float(np.median(allv))
        if vals[0] and vals[2]:
            a, b = np.median(vals[0]), np.median(vals[2]); lr[k] = 1.0 if (a > 0 and b > 0 and 0.5 <= a / b <= 2.0) else 0.0
    return t, out, lr, len(water), far_i, (float(np.median(absmag)) if absmag else float('nan'))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--flow-dir', required=True, type=Path); ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='train,val')
    ap.add_argument('--clips'); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--diag-dir', type=Path, nargs='*', default=[]); ap.add_argument('--tracks-dir', type=Path, nargs='*', default=[]); ap.add_argument('--horizon', type=float, default=0.5)
    ap.add_argument('--smooth-s', type=float, default=5.0); ap.add_argument('--zmax', type=float, default=0.5); ap.add_argument('--margin', type=float, default=0.08); ap.add_argument('--min-valid', type=float, default=0.3); ap.add_argument('--nearest', type=int, default=0); ap.add_argument('--min-flow', type=float, default=0.0, help='px/s: median |flow| in the used water bands below this -> no prediction (sub-pixel flow = noise; dev clip_000 = 5 px/s)')
    a = ap.parse_args(argv); man = load_manifest(a.dataset); (a.out / 'predictions').mkdir(parents=True, exist_ok=True); rows = []
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; f = a.flow_dir / f'{key}.npz'
        if not f.is_file(): rows.append(dict(key=key, mode=c['camera_mode'], status='missing')); continue
        z = np.load(f); yh, src = horizon_frac(a.diag_dir, key, a.horizon, a.tracks_dir)
        t, p, lr, nw, far_i, fmag = proxy_from_npz(z, yh, a.zmax, a.margin, a.nearest); ok = np.isfinite(p); valid = float(ok.mean()) if len(p) else 0.0
        r = dict(key=key, mode=c['camera_mode'], n_pairs=int(len(t)), n_water_bands=nw, far_band=far_i, horizon=round(yh, 3), horizon_src=src, valid_frac=round(valid, 3),
                 lr_agree=round(float(np.nanmean(lr)), 3) if np.isfinite(lr).any() else float('nan'), flow_px_s=round(fmag, 1))
        if a.min_flow > 0 and (not np.isfinite(fmag) or fmag < a.min_flow): r['status'] = 'low_flow'; rows.append(r); print(f"{key}: low flow {fmag:.1f} px/s < {a.min_flow} -> no prediction (constant speed)"); continue
        if ok.sum() < 20 or valid < a.min_valid: r['status'] = 'too_few'; rows.append(r); print(f"{key}: too few valid frames ({ok.sum()}, valid {valid:.2f})"); continue
        tt, pp = t[ok], p[ok]; med = np.array([np.median(pp[max(0, i - 3):i + 4]) for i in range(len(pp))]); sm = smooth(med[:, None], tt, a.smooth_s)[:, 0]
        tg = np.arange(float(t[0]), float(t[-1]) + 1e-6, 1.0); sg = np.maximum(np.interp(tg, tt, sm), 1e-6); x = np.cumsum(np.concatenate([[0.0], sg[1:] * np.diff(tg)]))
        fi = z['frame_index'][np.clip(np.searchsorted(t, tg), 0, len(t) - 1)]
        pred = dict(key=key, frame_indices=[int(v) for v in fi], times=tg.tolist(), positions=np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]).tolist())
        write_json(a.out / 'predictions' / f'{key}.json', pred)
        s45 = smooth(np.interp(tg, tt, sm)[:, None], tg, 45.0)[:, 0]
        r.update(est_cv=round(float(np.std(s45) / max(np.mean(s45), 1e-9)), 3), coh30=r['lr_agree'], inl_med=float(nw), inl_ratio=r['valid_frac'], status='predicted'); rows.append(r)
        print(f"{key} {c['camera_mode']:8s} pairs {len(t)} water-bands {nw} far {far_i} flow {fmag:.0f}px/s horizon {yh:.2f} ({src.split('/')[-1] if src != 'default' else 'default'}) valid {valid:.2f} lr_agree {r['lr_agree']:.2f} est_cv {r['est_cv']:.2f}")
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/raft_speed_proxy.py', smooth_s=a.smooth_s, zmax=a.zmax, margin=a.margin, nearest=a.nearest, min_flow=a.min_flow, clips=rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
