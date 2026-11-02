#!/usr/bin/env python3
"""E029 (2026-10-26 22:40): LEARNED relative-speed model fitted on the train split (allowed by the rules), replacing the
hand-built median-over-bands proxy of raft_speed_proxy.py.  Same inputs (RAFT band statistics, horizon), same output format
(a speed run usable by policy_v004 --speed-run).  Model: ridge regression from per-frame log band flows (de-rotated by the far
field, planar-normalised, nearest NB water bands x left/right, static fraction, far-field magnitude, hand proxy) to
log(GPS speed / clip median).  Features and target are both clip-median-normalised, so the model learns the SHAPE weighting
across bands, which is the only thing the Sim(3)-aligned metric can see.
  fit:     learned_speed.py fit --flow-dir F --dataset D --split train --clips a,b --hmani m1 m2 --out model.json
  predict: learned_speed.py predict --model model.json --flow-dir F --dataset D --split val --clips a,b --hmani m1 --proxy-manifest p --out run_dir
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa
from public_dataset import load_manifest, select_clips, load_reference, write_json
from mvo_tools.fuse_heading_speed import smooth

SYM = False
NB = 6  # nearest water bands used as features (bottom-most = largest flow = least sub-pixel noise)


def horizon_from_manifests(mani_paths, key, default=0.5):
    for p in mani_paths:
        for r in json.load(open(p)).get('clips', []):
            if r['key'] == key and r.get('horizon') is not None: return float(r['horizon'])
    return default


def features(z, yh_frac, zmax=0.5, margin=0.08):
    """per-frame feature matrix (T x F), hand proxy (T,), far magnitude (T,)"""
    t = z['t']; vec = z['vec']; zf = z['zfrac']; rows = z['rows']; H = float(z['height'])
    T, R = vec.shape[0], vec.shape[1]; yc = 0.5 * (rows[:-1] + rows[1:]); yh = yh_frac * H
    above = np.where(yc < yh)[0]; far_i = int(above[-1]) if len(above) else -1
    water = [i for i in range(R) if rows[i] >= yh + margin * H]; near = water[-NB:]
    far = np.array([np.nanmedian(vec[k, far_i, :, :], axis=0) if far_i >= 0 else np.zeros(2) for k in range(T)]); far[~np.isfinite(far)] = 0.0
    farmag = np.hypot(far[:, 0], far[:, 1])
    def norm_flow(i, side):
        d = vec[:, i, side, :] - far; m = np.hypot(d[:, 0], d[:, 1]); n = m / ((yc[i] - yh) / H) ** 2
        bad = ~np.isfinite(n) | (np.isfinite(zf[:, i, side]) & (zf[:, i, side] > zmax)); n = n.copy(); n[bad] = np.nan; return n
    cols = []
    for side in (0, 2):
        for j in range(NB):
            if j < len(near): cols.append(np.log(norm_flow(near[-1 - j], side) + 1e-3))
            else: cols.append(np.full(T, np.nan))
    allw = np.column_stack([norm_flow(i, s) for i in water for s in (0, 2)]) if water else np.full((T, 1), np.nan)
    proxy = np.nanmedian(allw, axis=1) if water else np.full(T, np.nan)
    zl = np.nanmean(zf[:, water, 0], axis=1) if water else np.full(T, np.nan); zr = np.nanmean(zf[:, water, 2], axis=1) if water else np.full(T, np.nan)
    if SYM:  # left/right averaged per band: removes the side asymmetry a 46-clip fit cannot support
        cols = [np.nanmean(np.column_stack([cols[j], cols[NB + j]]), axis=1) for j in range(NB)]; zl = zr = np.nanmean(np.column_stack([zl, zr]), axis=1)
        X = np.column_stack(cols + [np.log(proxy + 1e-3), np.log(farmag + 1.0), zl])
    else: X = np.column_stack(cols + [np.log(proxy + 1e-3), np.log(farmag + 1.0), zl, zr])
    # clip-median normalisation of the log features (first NB*2+2 columns); zfrac columns stay raw; NaN -> 0 (= at clip median)
    nlog = (NB if SYM else NB * 2) + 2
    med = np.nanmedian(X[:, :nlog], axis=0); med[~np.isfinite(med)] = 0.0; X[:, :nlog] -= med
    X[~np.isfinite(X)] = 0.0
    return t, X, proxy, len(water)


def gps_rel_speed(root, c, t_frames):
    rt, rp = load_reference(root, c); v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) / np.diff(rt); tm = 0.5 * (rt[:-1] + rt[1:])
    vs = smooth(v[:, None], tm, 5.0)[:, 0]; vf = np.interp(t_frames, tm, vs); med = float(np.median(vf))
    if med < 0.5: return None
    return np.clip(np.log(np.maximum(vf, 1e-3) / med), -3.0, 1.5)


def ridge_fit(X, y, alpha):
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xs = (X - mu) / sd; A = Xs.T @ Xs + alpha * np.eye(X.shape[1]); w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=float(y.mean()))


def ridge_predict(m, X): return ((X - np.array(m['mu'])) / np.array(m['sd'])) @ np.array(m['w']) + m['b']


def cmd_fit(a):
    man = load_manifest(a.dataset); root = a.dataset.resolve().parent; Xs, ys, gs = [], [], []
    for c in select_clips(man, a.split, a.clips):
        f = a.flow_dir / f"{c['key']}.npz"
        if not f.is_file(): continue
        z = np.load(f); t, X, proxy, nw = features(z, horizon_from_manifests(a.hmani, c['key']))
        y = gps_rel_speed(root, c, t)
        if y is None or nw == 0: print(f"{c['key']}: skipped"); continue
        if a.fit_smooth > 0:  # fit at the time-scale the policy uses (45 s): frame-level noise otherwise attenuates the amplitude (ridge -> k~0.3)
            X = smooth(X, t, a.fit_smooth); y = smooth(y[:, None], t, a.fit_smooth)[:, 0]
        sub = np.arange(0, len(y), 10); X, y = X[sub], y[sub]  # ~1 Hz: frames within a window are near-duplicates
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), len(gs)))
    X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs); n = len(Xs); print(f'fit frames {len(y)} clips {n} features {X.shape[1]}')
    best = None
    for alpha in [1, 10, 100, 1000, 10000, 100000]:
        err = []
        for fold in range(5):
            te = (g % 5) == fold; m = ridge_fit(X[~te], y[~te], alpha); err.append(np.mean((ridge_predict(m, X[te]) - y[te]) ** 2))
        mse = float(np.mean(err)); print(f'  alpha {alpha:>7}: grouped-CV mse {mse:.4f}')
        if best is None or mse < best[0]: best = (mse, alpha)
    m = ridge_fit(X, y, best[1]); m.update(alpha=best[1], cv_mse=best[0], n_frames=int(len(y)), n_clips=n, NB=NB,
                                         feature_names=([f'B{j}' for j in range(NB)] + ['proxy', 'far', 'z']) if SYM else ([f'{s}{j}' for s in ('L', 'R') for j in range(NB)] + ['proxy', 'far', 'zl', 'zr']), sym=SYM)
    base_mse = float(np.mean((X[:, NB if SYM else NB * 2] - y) ** 2)); m['proxy_mse_insample'] = base_mse
    print('weights (standardised):', {k: round(v, 3) for k, v in zip(m['feature_names'], m['w'])}); print(f'proxy-alone in-sample mse {base_mse:.4f} vs model CV mse {best[0]:.4f}')
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(m, open(a.out, 'w'), indent=1); return 0


def cmd_predict(a):
    man = load_manifest(a.dataset); m = json.load(open(a.model)); globals()['SYM'] = bool(m.get('sym', False)); (a.out / 'predictions').mkdir(parents=True, exist_ok=True); rows = []
    pm = {r['key']: r for r in json.load(open(a.proxy_manifest)).get('clips', [])} if a.proxy_manifest else {}
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; f = a.flow_dir / f'{key}.npz'; pr = pm.get(key, {})
        r = dict(key=key, mode=c['camera_mode'], flow_px_s=pr.get('flow_px_s'), lr_agree=pr.get('lr_agree'), horizon=horizon_from_manifests(a.hmani, key))
        if not f.is_file(): r['status'] = 'missing'; rows.append(r); continue
        if pr and pr.get('status') != 'predicted': r['status'] = pr.get('status'); rows.append(r); print(f"{key}: proxy status {r['status']} -> no prediction"); continue
        z = np.load(f); t, X, proxy, nw = features(z, r['horizon'])
        if nw == 0: r['status'] = 'no_water'; rows.append(r); continue
        sp = np.exp(ridge_predict(m, X)); ok = np.isfinite(sp); tt, pp = t[ok], sp[ok]
        med = np.array([np.median(pp[max(0, i - 3):i + 4]) for i in range(len(pp))]); sm = smooth(med[:, None], tt, a.smooth_s)[:, 0]
        tg = np.arange(float(t[0]), float(t[-1]) + 1e-6, 1.0); sg = np.maximum(np.interp(tg, tt, sm), 1e-6); x = np.cumsum(np.concatenate([[0.0], sg[1:] * np.diff(tg)]))
        fi = z['frame_index'][np.clip(np.searchsorted(t, tg), 0, len(t) - 1)]
        write_json(a.out / 'predictions' / f'{key}.json', dict(key=key, frame_indices=[int(v) for v in fi], times=tg.tolist(), positions=np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]).tolist()))
        s45 = smooth(np.interp(tg, tt, sm)[:, None], tg, 45.0)[:, 0]
        r.update(est_cv=round(float(np.std(s45) / max(np.mean(s45), 1e-9)), 3), coh30=r['lr_agree'], inl_med=float(nw), inl_ratio=round(float(ok.mean()), 3), status='predicted'); rows.append(r)
        print(f"{key} {c['camera_mode']:8s} est_cv {r['est_cv']:.2f}")
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/learned_speed.py', model=str(a.model), smooth_s=a.smooth_s, clips=rows)); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fit', 'predict'):
        p = sub.add_parser(name); p.add_argument('--flow-dir', required=True, type=Path); p.add_argument('--dataset', required=True, type=Path); p.add_argument('--split', default='train')
        p.add_argument('--clips'); p.add_argument('--hmani', type=Path, nargs='*', default=[]); p.add_argument('--out', required=True, type=Path)
        if name == 'fit': p.add_argument('--fit-smooth', type=float, default=45.0)
        if name == 'predict': p.add_argument('--model', required=True, type=Path); p.add_argument('--proxy-manifest', type=Path); p.add_argument('--smooth-s', type=float, default=5.0)
    ap.add_argument('--sym', action='store_true')
    a = ap.parse_args(argv); globals()['SYM'] = bool(a.sym)
    return cmd_fit(a) if a.cmd == 'fit' else cmd_predict(a)


if __name__ == '__main__':
    sys.exit(main())
