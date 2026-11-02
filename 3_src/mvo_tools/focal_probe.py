#!/usr/bin/env python3
"""VGGT focal-length probe of the flir-VIS clips (E006c; development script focal_probe.py generalised with a CLI).
Per clip: 5 windows of 12 consecutive frames (positions at 10/30/50/70/90 % of the clip) -> VGGT camera head -> median focal length (px, at the
518-px frame width); ratio = median(VGGT focal) / nominal focal of the released calibration.  Frames = the 8 fps / 518 px frames of the
VGGT window stage (frames/<clip>).  Output JSON {clip: {mode, fx_manifest, fx_vggt_med, fx_vggt, ratio}} -> make_gain_table.py.
Usage (GPU): focal_probe.py --frames-root frames --clips clip_093,... --out focal_probe_test.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames-root', required=True, type=Path); ap.add_argument('--clips', required=True); ap.add_argument('--out', required=True, type=Path); a = ap.parse_args()
    m = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval(); out = {}
    for key in a.clips.split(','):
        d = a.frames_root / key; fm = json.load(open(d / 'frames_manifest.json')); files = [str(d / f['file']) for f in fm['frames']]; cal = fm['calibration']; n = len(files); fx = []
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            s = max(0, min(n - 12, int(n * q))); imgs = load_and_preprocess_images(files[s:s + 12]).to('cuda')
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                agg, _ = m.aggregator(imgs[None]); pe = m.camera_head(agg)[-1]; _, intri = pose_encoding_to_extri_intri(pe, imgs.shape[-2:])
            fx.append(float(intri[0, :, 0, 0].float().median()))
        out[key] = dict(mode=fm['camera_mode'], fx_manifest=cal['fx'], fx_vggt_med=float(np.median(fx)), fx_vggt=fx, ratio=float(np.median(fx) / cal['fx']))
        print(key, fm['camera_mode'], 'ratio=%.2f' % out[key]['ratio'], [round(v) for v in fx], flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)
    print('PROBE DONE', flush=True)

if __name__ == '__main__':
    sys.exit(main())
