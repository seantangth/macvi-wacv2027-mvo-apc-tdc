#!/usr/bin/env python3
"""E021 (2026-10-24): GPU dense optical flow (torchvision RAFT-large) between consecutive sampled frames, summarised per
ROW BAND x COLUMN THIRD so the water-surface speed proxies can be assembled offline (horizon-relative bands, far-field
de-rotation, bow exclusion) without re-running the GPU.  Why RAFT and not Farneback: at 10 fps the near-water texture
moves tens of px per step at cruising speed; Farneback saturates there (E021 CPU proxies: r ~ 0), RAFT does not.

Per consecutive frame pair (px/s, divided by the real dt): for each of --rows horizontal bands (full height) x 3 column
thirds: median u, median v, median |f| over pixels with |f| > 0.1 px/frame, and the fraction of near-static pixels.
Writes <out>.npz: t (T,), vec (T,rows,3,2) [u,v], mag (T,rows,3), zfrac (T,rows,3), rows, width, height.
Usage: raft_water_flow.py --frames frames/clip_005 --timing timing/clip_005.json --out runs/flow/clip_005.npz [--rows 16 --batch 8 --iters 12 --max-pairs N]
E054 (2026-10-28): --model searaft swaps ONLY the flow network for SEA-RAFT (github.com/princeton-vl/SEA-RAFT, clone at --sea-root,
config --sea-cfg = config/eval/spring-M.json, weights --sea-url on HuggingFace or a local .pth via --weights); frames, resize to
multiples of 8, dt and the band statistics are the same code path, so the npz feeds raft_speed_proxy.py unchanged.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np


def load_searaft(a):
    """E054: SEA-RAFT exactly as the repo's custom.py builds it: config json -> argparse.Namespace -> RAFT.from_pretrained(url, args=args)
    (or RAFT(args) + utils.utils.load_ckpt for a local .pth).  core/*.py import their siblings bare (from update import ...), so core/ goes
    on sys.path.  Constructing RAFT(args) with pretrain=resnet34 also fetches torchvision's ImageNet ResNet34 once (then overwritten)."""
    import os
    root = Path(a.sea_root or os.environ.get('SEA_RAFT_ROOT', 'SEA-RAFT')).expanduser().resolve(); sys.path.insert(0, str(root / 'core'))
    from raft import RAFT
    cfg = Path(a.sea_cfg) if a.sea_cfg else root / 'config' / 'eval' / 'spring-M.json'; args = argparse.Namespace(**json.load(open(cfg)))
    if a.weights:
        from utils.utils import load_ckpt
        model = RAFT(args); load_ckpt(model, str(a.weights)); src = str(a.weights)
    else: model = RAFT.from_pretrained(a.sea_url, args=args); src = a.sea_url
    print(f'SEA-RAFT {src} cfg {cfg.name} iters {args.iters} (cfg) scale {a.sea_scale}', flush=True)
    return model, args


def searaft_flow(model, A, B, iters, scale=0):
    """E054: SEA-RAFT forward as custom.py calc_flow/forward_flow: float images in 0..255 (the model maps them to [-1,1] itself, the same
    range torchvision's RAFT transforms produce, and pads to /8), optional 2**scale resize, final flow output['flow'][-1] rescaled to input px."""
    import torch.nn.functional as F
    hw = A.shape[-2:]
    if scale:
        A = F.interpolate(A, scale_factor=2 ** scale, mode='bilinear', align_corners=False); B = F.interpolate(B, scale_factor=2 ** scale, mode='bilinear', align_corners=False)
    fl = model(A, B, iters=iters, test_mode=True)['flow'][-1]
    if scale: fl = F.interpolate(fl, size=hw, mode='bilinear', align_corners=False) * (0.5 ** scale)
    return fl.float()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', required=True, type=Path); ap.add_argument('--timing', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--rows', type=int, default=16); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--iters', type=int, default=12); ap.add_argument('--max-pairs', type=int, default=0); ap.add_argument('--weights', type=Path, help='E035: fine-tuned state dict (raft_selfsup.py) loaded over the C_T_SKHT_V2 weights; with --model searaft: a local SEA-RAFT .pth instead of --sea-url')
    ap.add_argument('--model', choices=('raft', 'searaft'), default='raft', help='E054: flow network (default raft = torchvision RAFT-large C_T_SKHT_V2, unchanged)')
    ap.add_argument('--sea-root', help='SEA-RAFT clone (default $SEA_RAFT_ROOT or ./SEA-RAFT)'); ap.add_argument('--sea-cfg', help='SEA-RAFT config json (default <sea-root>/config/eval/spring-M.json)')
    ap.add_argument('--sea-url', default='MemorySlices/Tartan-C-T-TSKH-spring540x960-M', help='HuggingFace repo of the SEA-RAFT weights')
    ap.add_argument('--sea-iters', type=int, default=0, help='SEA-RAFT refinement iterations (0 = the config value, 4 for spring-M; --iters is RAFT-only)')
    ap.add_argument('--sea-scale', type=int, default=0, help='SEA-RAFT input resize 2**scale (spring-M.json says -1 for 1080p Spring frames; ours are 640 px -> 0 = native, as RAFT)')
    a = ap.parse_args(argv)
    import torch
    from PIL import Image
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    man = json.load(open(a.frames / 'frames_manifest.json')); files = [a.frames / f['file'] for f in man['frames']]; fidx = [int(f['decoded_index']) for f in man['frames']]
    ft = np.asarray(json.load(open(a.timing))['times'], float)  # challenge timing json: {key, times}
    t = ft[np.clip(fidx, 0, len(ft) - 1)]
    dev = 'cuda'
    if a.model == 'searaft':
        model, sargs = load_searaft(a); model = model.to(dev).eval(); tf = None; sea_iters = a.sea_iters if a.sea_iters > 0 else int(sargs.iters)
    else:
        W = Raft_Large_Weights.C_T_SKHT_V2; model = raft_large(weights=W, progress=False)
        if a.weights: sd = torch.load(a.weights, map_location='cpu'); model.load_state_dict(sd.get('model', sd)); print(f'loaded fine-tuned weights {a.weights}', flush=True)
        model = model.to(dev).eval(); tf = W.transforms()
    im0 = Image.open(files[0]).convert('RGB'); w, h = im0.size; w8, h8 = (w // 8) * 8, (h // 8) * 8
    def load(f):
        im = Image.open(f).convert('RGB').resize((w8, h8), Image.BILINEAR); return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    npairs = len(files) - 1
    if a.max_pairs: npairs = min(npairs, a.max_pairs)
    rows = np.linspace(0, h8, a.rows + 1).astype(int); cols = np.linspace(0, w8, 4).astype(int)
    VEC = np.full((npairs, a.rows, 3, 2), np.nan, np.float32); MAG = np.full((npairs, a.rows, 3), np.nan, np.float32); ZF = np.full((npairs, a.rows, 3), np.nan, np.float32); T = np.zeros(npairs)
    t0 = time.time(); cache = {}
    def get(i):
        if i not in cache: cache[i] = load(files[i])
        return cache[i]
    with torch.no_grad():
        for s in range(0, npairs, a.batch):
            ids = list(range(s, min(npairs, s + a.batch)))
            A = torch.stack([get(i) for i in ids]); B = torch.stack([get(i + 1) for i in ids])
            if a.model == 'searaft': fl = searaft_flow(model, A.to(dev).float(), B.to(dev).float(), sea_iters, a.sea_scale)  # uint8 0..255 -> float, fp32 (as the repo's eval)
            else:
                A, B = tf(A, B); A = A.to(dev); B = B.to(dev)
                with torch.autocast('cuda', dtype=torch.float16):
                    fl = model(A, B, num_flow_updates=a.iters)[-1].float()  # (b,2,H,W) px/frame
            fl = fl.cpu().numpy()
            for k, i in enumerate(ids):
                dt = max(float(t[i + 1] - t[i]), 1e-3); T[i] = 0.5 * (t[i] + t[i + 1]); f = fl[k]; mag = np.sqrt(f[0] ** 2 + f[1] ** 2); mv = mag > 0.1
                for r in range(a.rows):
                    for c3 in range(3):
                        sl = (slice(rows[r], rows[r + 1]), slice(cols[c3], cols[c3 + 1])); m = mv[sl]; ZF[i, r, c3] = 1.0 - m.mean()
                        if m.sum() > 30:
                            VEC[i, r, c3, 0] = np.median(f[0][sl][m]) / dt; VEC[i, r, c3, 1] = np.median(f[1][sl][m]) / dt; MAG[i, r, c3] = np.median(mag[sl][m]) / dt
            for i in ids[:-1]: cache.pop(i, None)
            if s // a.batch % 50 == 0: print(f'pair {s}/{npairs} {time.time() - t0:.0f}s', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); extra = dict(flow_model=np.asarray(f'searaft:{a.weights or a.sea_url}:iters{sea_iters}:scale{a.sea_scale}')) if a.model == 'searaft' else {}
    np.savez(a.out, t=T, vec=VEC, mag=MAG, zfrac=ZF, rows=rows, width=w8, height=h8, frame_index=np.asarray(fidx[:npairs]), **extra)
    print(f'wrote {a.out} pairs={npairs} secs={time.time() - t0:.0f} median|f| bottom-third={np.nanmedian(MAG[:, -a.rows // 3:, :]):.1f} px/s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
