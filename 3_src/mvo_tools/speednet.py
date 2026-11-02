#!/usr/bin/env python3
"""E036 (2026-10-27): cross-camera relative-speed network trained on the TRAIN split (all cameras), validated on val.

Input per clip: RAFT band statistics (raft_water_flow.py npz: per frame, 16 row bands x 3 column thirds of median flow
[u, v], |f| and static fraction) + horizon fraction + camera type.  Per frame features: far-field de-rotation, the nearest
NB water bands (bottom-most first) x 3 columns of planar-normalised log |flow - far|, their static fractions, the far-field
log magnitude, and the hand proxy (median over bands); log features are clip-median normalised (only the relative profile
matters under the Sim(3)-aligned metric).  Aggregated to 1 Hz.  Global features (camera one-hot, log clip flow level) are
broadcast over time.  Target: log(GPS speed / clip median), GPS smoothed 5 s.  Model: residual dilated 1-D CNN
(receptive field ~4 min), loss on the clip-median-centred prediction, ensemble of seeds.
Writes speed runs (positions = cumulative predicted relative speed at 1 Hz) usable by policy_v004 --speed-run.
Usage: speednet.py --flow-dirs d1 d2 --hmani m1 m2 --out-root O [--epochs 300 --seeds 5]
"""
import argparse, json, sys, os, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, '3_src'); sys.path.insert(0, '3_src/starter_upstream/starter')
from public_dataset import load_manifest, load_reference
from mvo_tools.fuse_heading_speed import smooth

NB = 8; CAMS = ['wide', 'flir-VIS', 'flir-IR', 'infrared', 'sionyx', 'zoom']
DEV = Path('1_data/raw/release/release_manifest.json'); TST = Path('1_data/raw/lookout-vo-test-v0.1/release_manifest.json')


def clip_features(z, yh, margin=0.08):
    t = z['t']; vec = z['vec']; zf = z['zfrac']; rows = z['rows']; H = float(z['height']); T = vec.shape[0]
    yc = 0.5 * (rows[:-1] + rows[1:]); yhp = yh * H; above = np.where(yc < yhp)[0]; far_i = int(above[-1]) if len(above) else -1
    water = [i for i in range(len(yc)) if rows[i] >= yhp + margin * H]; near = water[::-1][:NB]   # bottom-most first
    far = np.nanmedian(vec[:, far_i], axis=1) if far_i >= 0 else np.zeros((T, 2)); far[~np.isfinite(far)] = 0.0
    feats = []; masks = []
    for j in range(NB):
        for c in range(3):
            if j < len(near):
                i = near[j]; d = vec[:, i, c, :] - far; m = np.hypot(d[:, 0], d[:, 1]) / ((yc[i] - yhp) / H) ** 2
                bad = ~np.isfinite(m) | (np.isfinite(zf[:, i, c]) & (zf[:, i, c] > 0.5)); lm = np.log(m + 1e-3); lm[bad] = np.nan
                feats.append(lm); masks.append(np.where(np.isfinite(zf[:, i, c]), zf[:, i, c], 1.0))
            else:
                feats.append(np.full(T, np.nan)); masks.append(np.ones(T))
    F = np.column_stack(feats); proxy = np.nanmedian(F, axis=1); farm = np.log(np.hypot(far[:, 0], far[:, 1]) + 1.0)
    X = np.column_stack([F, proxy, farm]); med = np.nanmedian(X, axis=0); med[~np.isfinite(med)] = 0.0; X = X - med
    valid = np.isfinite(X).astype(float); X[~np.isfinite(X)] = 0.0
    Z = np.column_stack(masks); flow_level = float(np.nanmedian(z['mag'][:, -6:, :])) if np.isfinite(z['mag']).any() else 0.0
    return t, np.column_stack([X, valid[:, :NB * 3], Z]), flow_level, len(water)


def to_1hz(t, X):
    tg = np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1e-6, 1.0); out = np.zeros((len(tg), X.shape[1]))
    idx = np.clip(np.searchsorted(tg, t) - 0, 0, len(tg) - 1)
    cnt = np.bincount(idx, minlength=len(tg)).astype(float)
    for k in range(X.shape[1]): out[:, k] = np.bincount(idx, weights=X[:, k], minlength=len(tg)) / np.maximum(cnt, 1)
    keep = cnt > 0; return tg[keep], out[keep]


def gps_target(root, c, tg):
    rt, rp = load_reference(root, c); v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) / np.diff(rt); tm = 0.5 * (rt[:-1] + rt[1:])
    vs = smooth(v[:, None], tm, 5.0)[:, 0]; vg = np.interp(tg, tm, vs); med = float(np.median(vg))
    if med < 0.5: return None
    return np.log(np.maximum(vg, 0.02 * med) / med)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--flow-dirs', nargs='+', required=True); ap.add_argument('--hmani', nargs='+', required=True); ap.add_argument('--hjson', nargs='*', default=[])
    ap.add_argument('--out-root', required=True); ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--train-split', default='train'); ap.add_argument('--mode', default='direct', choices=['direct', 'gate'])
    ap.add_argument('--save-weights', type=Path, help='directory: write seed<k>.pt (state_dict) after training each seed')
    ap.add_argument('--load-weights', type=Path, help='directory with seed<k>.pt: skip training, predict only (inference on the released test clips)')
    ap.add_argument('--proxy-dirs', nargs='*', default=[], help='gate mode: RAFT rule proxy speed runs; the net only learns a trust gate g(t) in [0, 1.2] applied to the proxy log profile'); a = ap.parse_args()
    import torch, torch.nn as nn
    torch.set_num_threads(8)
    hz = {}
    for p in a.hmani:
        for r in json.load(open(p)).get('clips', []):
            if r.get('horizon') is not None: hz.setdefault(r['key'], float(r['horizon']))
    for p in a.hjson:
        for k, v in json.load(open(p)).items():
            if isinstance(v, dict) and v.get('horizon') is not None: hz.setdefault(k, float(v['horizon']))
    flows = {}
    for d in a.flow_dirs:
        for f in glob.glob(os.path.join(d, 'clip_*.npz')): flows.setdefault(os.path.basename(f)[:-4], f)
    from mvo_tools.fuse_heading_speed import speed_of
    prox = {}
    for d in a.proxy_dirs:
        for f in glob.glob(os.path.join(d, 'predictions', 'clip_*.json')): prox.setdefault(os.path.basename(f)[:-5], f)
    dev = load_manifest(DEV) if DEV.is_file() else dict(clips=[]); tst = load_manifest(TST); root_dev = DEV.resolve().parent   # inference-only runs need no dev package
    data = {}
    for c in dev['clips'] + tst['clips']:
        k = c['key']
        if k not in flows: continue
        z = np.load(flows[k]); t, X, lvl, nw = clip_features(z, hz.get(k, 0.5)); tg, X1 = to_1hz(t, X)
        cam = np.zeros(len(CAMS)); cam[CAMS.index(c['camera_mode'])] = 1.0
        G = np.tile(np.concatenate([cam, [np.log(lvl + 1.0)]]), (len(tg), 1)); X1 = np.column_stack([X1, G]).astype(np.float32)
        pp = np.zeros(len(tg), np.float32)
        if k in prox:
            e = json.load(open(prox[k])); tp = np.asarray(e['times'], float); xp = np.asarray(e['positions'], float); sp_ = np.interp(tg, tp, speed_of(tp, xp, 45.0)); md = np.median(sp_)
            if np.isfinite(md) and md > 0: pp = np.clip(np.log(np.maximum(sp_ / md, 1e-3)), -3, 1.5).astype(np.float32)
        X1 = np.column_stack([X1, pp]).astype(np.float32)
        y = gps_target(root_dev, c, tg) if c['split'] in ('train', 'val') else None
        data[k] = dict(split=c['split'], mode=c['camera_mode'], t=tg, X=X1, y=y, level=lvl, nwater=nw, p=pp)
    tr = [k for k, d in data.items() if d['split'] in a.train_split.split(',') and d['y'] is not None]
    print(f'clips with features: {len(data)}; training clips: {len(tr)} ({", ".join(sorted(set(data[k]["mode"] for k in tr)))})', flush=True)
    C = next(iter(data.values()))['X'].shape[1]
    class Net(nn.Module):
        def __init__(s, w):
            super().__init__(); s.inp = nn.Conv1d(C, w, 9, padding=4); s.blocks = nn.ModuleList([nn.Conv1d(w, w, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8, 16)]); s.out = nn.Conv1d(w, 1, 1); s.act = nn.GELU(); s.drop = nn.Dropout(0.1)
        def forward(s, x):
            h = s.act(s.inp(x))
            for b in s.blocks: h = h + s.drop(s.act(b(h)))
            return s.out(h)[:, 0]
    preds = {k: [] for k in data}
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed); net = Net(a.width); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        if a.load_weights:
            net.load_state_dict(torch.load(a.load_weights / f'seed{seed}.pt', map_location='cpu'))
        else:
            for ep in range(a.epochs):
                net.train(); np.random.shuffle(tr); tot = 0.0
                for k in tr:
                    d = data[k]; X = torch.from_numpy(d['X'].T[None]); y = torch.from_numpy(d['y'].astype(np.float32))
                    if X.shape[-1] > 200 and np.random.rand() < 0.5:   # random crop for augmentation
                        s0 = np.random.randint(0, X.shape[-1] - 180); X = X[..., s0:s0 + 180]; y = y[s0:s0 + 180]
                    o = net(X)[0]
                    if a.mode == 'gate': pb = X[0, -1]; p = 1.2 * torch.sigmoid(o + 1.0) * pb
                    else: p = o
                    p = p - p.median(); yy = y - y.median(); loss = ((p - yy) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
                if ep % 50 == 0 or ep == a.epochs - 1: print(f'seed {seed} epoch {ep} train mse {tot / len(tr):.4f}', flush=True)
            if a.save_weights:
                a.save_weights.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), a.save_weights / f'seed{seed}.pt')
        net.eval()
        with torch.no_grad():
            for k, d in data.items():
                o = net(torch.from_numpy(d['X'].T[None]))[0]
                p = (1.2 * torch.sigmoid(o + 1.0) * torch.from_numpy(d['p'])).numpy() if a.mode == 'gate' else o.numpy()
                preds[k].append(p - np.median(p))
    O = Path(a.out_root); summary = {}
    for k, d in data.items():
        p = np.mean(preds[k], axis=0); sp = np.exp(p); x = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(d['t']))])
        out = O / d['split'] / 'predictions'; out.mkdir(parents=True, exist_ok=True)
        json.dump(dict(key=k, frame_indices=list(range(len(d['t']))), times=d['t'].tolist(), positions=np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]).tolist()), open(out / f'{k}.json', 'w'))
        if d['y'] is not None: summary[k] = dict(split=d['split'], mode=d['mode'], r=float(np.corrcoef(p, d['y'])[0, 1]), pred_cv=float(np.std(sp) / np.mean(sp)), gps_cv=float(np.std(np.exp(d['y'])) / np.mean(np.exp(d['y']))))
    json.dump(summary, open(O / 'summary.json', 'w'), indent=1)
    for sp_ in ('train', 'val'):
        ks = [k for k in summary if summary[k]['split'] == sp_]
        if ks: print(f'{sp_}: n={len(ks)} median r {np.median([summary[k]["r"] for k in ks]):+.2f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
