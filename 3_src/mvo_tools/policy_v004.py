#!/usr/bin/env python3
"""Per-clip policy v004 (2026-10-19 evening, dev-mix 15.68 vs v002 policy 21.87; see EXPERIMENT_LOG E005b):
  flir-IR  -> horizon-band yaw tracker heading (rotw) + constant speed
  wide     -> rotw heading + DPVO speed profile (20 s smoothed) when the DPVO trajectory is sane (select_policy
              rule: L/net <= T, speed CV <= C, NaN-repaired fraction <= nan_max), else rotw + constant speed
  flir-VIS -> window model-selection heading (sel: rotation-only vs essential matrix, 5 s windows) + constant speed
  other    -> rotw + constant speed
  missing  -> timestamp-only straight line (coverage gate always holds)
Writes <out>/predictions/*.json + inference_manifest.json (per clip: choice, reason, DPVO stats).

Usage: policy_v004.py --dataset <manifest> --split test --hyaw runs/hyaw_test_w5 --dpvo runs/dpvo_fps8_s0_test --out runs/v004
       [--vis sel|essw|rotw|line] [--T 3 --C 1.5 --nan-max 0.2 --smooth-s 20]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mvo_tools  # noqa: F401
from public_dataset import load_manifest, load_timestamps, read_json, select_clips, validate_prediction, write_json
from mvo_tools.select_policy import straight_line, traj_stats, smooth
from mvo_tools.fuse_heading_speed import heading_of, speed_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path); ap.add_argument('--split', default='test'); ap.add_argument('--clips')
    ap.add_argument('--hyaw', required=True, type=Path); ap.add_argument('--dpvo', type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--vis', default='sel', help="estimator dir under --hyaw for flir-VIS (sel|essw|rotw|line|<any dir, e.g. vggt_heading>)"); ap.add_argument('--ir', default='rotw'); ap.add_argument('--wide', default='rotw')
    ap.add_argument('--T', type=float, default=3.0); ap.add_argument('--C', type=float, default=1.5); ap.add_argument('--nan-max', type=float, default=0.2)
    ap.add_argument('--max-rawspd', type=float, default=0.0, help='wide: use DPVO speed only if DPVO raw speed (its own scale units) <= this; 0 = off. dev: the 4 big harms (037/059/007/052, >=10 m/s open water) have rawspd 0.11-27, rescues <=0.036')
    ap.add_argument('--speed-run', type=Path, help='speed-profile run (e.g. water_speed.py output): used before/after DPVO per --speed-order')
    ap.add_argument('--speed-modes', default='wide', help='camera modes that may take the speed-run profile (comma list)')
    ap.add_argument('--speed-order', default='water,dpvo', help='priority of speed sources: water (speed-run) and/or dpvo')
    ap.add_argument('--speed-shrink', type=float, default=1.0, help='linear: sp = 1 + shrink*(sp/median - 1); log: sp = (sp/median)**shrink')
    ap.add_argument('--speed-shrink-mode', default='linear', choices=['linear', 'log'], help='E023 (10-25): log-domain blend is what the leaderboard teams use; linear shrink cannot represent a stop (0.02x at k=0.3 -> 0.71x)')
    ap.add_argument('--speed-floor', type=float, default=0.05, help='minimum relative speed')
    ap.add_argument('--water-min-coh', type=float, default=0.85, help='water speed gate: fraction of windows whose translation direction is within 30 deg of the clip median (reflections/garbage scatter)')
    ap.add_argument('--water-min-inl', type=float, default=100, help='water speed gate: median RANSAC inliers per window')
    ap.add_argument('--water-min-inlr', type=float, default=0.40, help='water speed gate: median inlier ratio')
    ap.add_argument('--water-force-clips', default='', help='comma list of clips that take the water speed profile regardless of the gate (e.g. clean sustained-stop profile on test)')
    ap.add_argument('--water-skip-clips', default='', help='comma list of clips that never take the water speed profile')
    ap.add_argument('--speed-combine', default='first', choices=['first', 'mean', 'geomean'], help='when several speed sources pass their gates: take the first in --speed-order, or combine')
    ap.add_argument('--speed-smooth', type=float, default=5.0, help='seconds of smoothing applied to the speed-run profile')
    ap.add_argument('--override-clips', default='', help='comma list of clips routed to --override-est regardless of mode (e.g. guard-triggered / low-horizon clips -> vggt)')
    ap.add_argument('--override-est', default='', help='estimator dir under --hyaw used for --override-clips')
    ap.add_argument('--guard-ratio', type=float, default=1.5, help='flir-VIS: if sel total turn > ratio x rotw total turn (model selection ran wild), use rotw instead; 0 = off')
    ap.add_argument('--smooth-s', type=float, default=20.0); ap.add_argument('--smooth-h', type=float, default=2.0); ap.add_argument('--smooth-mode', default='angle', choices=['angle', 'pos'], help='angle: time-window mean of the unwrapped heading (v006+); pos: legacy position smoothing (v004/v005; irregular sampling makes the windowed mean step backwards -> heading spikes)'); ap.add_argument('--no-dpvo-speed', action='store_true')
    a = ap.parse_args(argv); man = load_manifest(a.dataset); root = a.dataset.resolve().parent
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True); recs = []
    nanfrac = {}; wdiag = {}
    if a.speed_run is not None and (a.speed_run / 'inference_manifest.json').is_file():
        wdiag = {r['key']: r for r in json.load(open(a.speed_run / 'inference_manifest.json')).get('clips', [])}
    if a.dpvo is not None and (a.dpvo / 'inference_manifest.json').is_file():
        for r in json.load(open(a.dpvo / 'inference_manifest.json')).get('clips', []):
            if r.get('fallback'): nanfrac[r['key']] = 1.0
            elif r.get('n_nan'): nanfrac[r['key']] = float(r.get('nan_frac') or 0.0)
    for c in select_clips(man, a.split, a.clips):
        key = c['key']; mode = c['camera_mode']; ft = load_timestamps(root, c); rec = dict(key=key, mode=mode); pred = None
        est = {'flir-IR': a.ir, 'wide': a.wide, 'flir-VIS': a.vis}.get(mode, 'rotw')
        ov = set(x for x in a.override_clips.split(',') if x)
        if key in ov and a.override_est:
            rec['override'] = f'{est} -> {a.override_est}'; est = a.override_est
        dg = a.hyaw / 'diag' / f'{key}.json'
        if est in ('sel', 'essw') and a.guard_ratio > 0 and dg.is_file() and key not in ov:
            d0 = json.load(open(dg)); ratio = d0.get(f'{est}_total_turn_deg', 0.0) / max(d0.get('rotw_total_turn_deg', 0.0), 1.0); rec['turn_ratio'] = round(ratio, 2)
            if ratio > a.guard_ratio:
                rec['guard'] = f'{est} total turn {ratio:.2f}x rotw -> rotw'; est = 'rotw'
        f = a.hyaw / est / 'predictions' / f'{key}.json' if est != 'line' else None
        if f is not None and f.is_file():
            d = read_json(f); t = np.asarray(d['times'], float); fi = np.asarray(d['frame_indices'], int); xyz = np.asarray(d['positions'], float)
            choice = f'{est}+const'; sp = np.ones(len(t))
            hd = smooth(heading_of(t, xyz, 0.0)[:, None], t, a.smooth_h)[:, 0] if (a.smooth_mode == 'angle' and a.smooth_h > 0) else heading_of(t, xyz, a.smooth_h)
            sources = [x for x in a.speed_order.split(',') if x]; cands = []
            for src in sources:
                if choice != f'{est}+const' and a.speed_combine == 'first': break
                if src == 'water' and a.speed_run is not None and mode in a.speed_modes.split(','):
                    g = a.speed_run / 'predictions' / f'{key}.json'; wd = wdiag.get(key, {})
                    wok = (wd.get('coh30') or 0.0) >= a.water_min_coh and (wd.get('inl_med') or 0.0) >= a.water_min_inl and (wd.get('inl_ratio') or 0.0) >= a.water_min_inlr
                    if key in a.water_force_clips.split(','): wok = True; rec['water_forced'] = True
                    if key in a.water_skip_clips.split(','): wok = False; rec['water_skipped'] = True
                    rec['water_gate'] = dict(coh30=round(wd.get('coh30') or 0.0, 2), inl_med=wd.get('inl_med'), inl_ratio=round(wd.get('inl_ratio') or 0.0, 2), ok=bool(wok))
                    if g.is_file() and wok:
                        e = read_json(g); ts = np.asarray(e['times'], float); xs = np.asarray(e['positions'], float); s = np.interp(t, ts, speed_of(ts, xs, a.speed_smooth)); med = np.median(s)
                        if np.isfinite(med) and med > 0:
                            rel = np.maximum(s / med, 1e-3); shr = np.exp(a.speed_shrink * np.log(rel)) if a.speed_shrink_mode == 'log' else 1.0 + a.speed_shrink * (rel - 1.0)
                            cands.append(('water', np.maximum(shr, a.speed_floor)))
                            if a.speed_combine == 'first': sp = cands[-1][1]; choice = f'{est}+water{int(a.speed_smooth)}'
                    elif not g.is_file():
                        rec['water_error'] = 'no speed-run prediction'
                elif src == 'dpvo' and mode == 'wide' and a.dpvo is not None and not a.no_dpvo_speed:
                    g = a.dpvo / 'predictions' / f'{key}.json'
                    if g.is_file():
                        try:
                            e = read_json(g); validate_prediction(e, key, ft); ts = np.asarray(e['times'], float); xs = np.asarray(e['positions'], float); st = traj_stats(ts, xs); rec.update({f'dpvo_{k}': v for k, v in st.items()})
                            sane = st['lnet'] <= a.T and st['spdcv'] <= a.C and nanfrac.get(key, 0.0) <= a.nan_max and np.all(np.isfinite(xs)) and (a.max_rawspd <= 0 or st.get('rawspd', 0.0) <= a.max_rawspd)
                            rec['dpvo_sane'] = bool(sane)
                            if sane:
                                s = np.interp(t, ts, speed_of(ts, xs, a.smooth_s)); med = np.median(s)
                                if np.isfinite(med) and med > 0:
                                    cands.append(('dpvo', s / med))
                                    if a.speed_combine == 'first': sp = cands[-1][1]; choice = f'{est}+dpvo_speed{int(a.smooth_s)}'
                        except Exception as exc:  # noqa: BLE001
                            rec['dpvo_error'] = str(exc)[:80]
                    else:
                        rec['dpvo_sane'] = False; rec['dpvo_error'] = 'no DPVO prediction'
            if a.speed_combine != 'first' and cands:
                arr = np.stack([v for _, v in cands]); sp = np.exp(np.mean(np.log(np.maximum(arr, 1e-3)), axis=0)) if a.speed_combine == 'geomean' else np.mean(arr, axis=0); choice = f"{est}+{a.speed_combine}({'+'.join(n for n, _ in cands)})"
            dt = np.diff(t); x = np.cumsum(np.concatenate([[0], sp[1:] * dt * np.cos(hd[1:])])); y = np.cumsum(np.concatenate([[0], sp[1:] * dt * np.sin(hd[1:])]))
            pred = dict(key=key, frame_indices=[int(v) for v in fi], times=t.tolist(), positions=np.column_stack([x, y, np.zeros_like(x)]).tolist()); rec['choice'] = choice
        if pred is None:
            fi, xyz = straight_line(ft); pred = dict(key=key, frame_indices=fi.tolist(), times=ft[fi].tolist(), positions=xyz.tolist()); rec['choice'] = 'line'; rec['reason'] = f'no {est} prediction' if est != 'line' else 'policy'
        validate_prediction(pred, key, ft); write_json(a.out / 'predictions' / f'{key}.json', pred); rec['fallback'] = rec['choice'] == 'line'; rec['status'] = 'predicted'; recs.append(rec)
        print(f"{key} {mode:8s} -> {rec['choice']:22s} {rec.get('reason', '')} {rec.get('guard', '')} " + (f"dpvo lnet={rec['dpvo_lnet']:.2f} cv={rec['dpvo_spdcv']:.2f} sane={rec.get('dpvo_sane')}" if 'dpvo_lnet' in rec else ''))
    write_json(a.out / 'inference_manifest.json', dict(wrapper='mvo_tools/policy_v004.py', hyaw=str(a.hyaw), dpvo=str(a.dpvo), vis=a.vis, T=a.T, C=a.C, nan_max=a.nan_max, max_rawspd=a.max_rawspd, guard_ratio=a.guard_ratio, smooth_h=a.smooth_h, smooth_mode=a.smooth_mode, speed_run=str(a.speed_run), speed_modes=a.speed_modes, speed_order=a.speed_order, speed_shrink=a.speed_shrink, speed_shrink_mode=a.speed_shrink_mode, speed_floor=a.speed_floor, water_gate=dict(coh=a.water_min_coh, inl=a.water_min_inl, inlr=a.water_min_inlr), override_clips=a.override_clips, override_est=a.override_est, smooth_s=a.smooth_s, clips=recs))
    from collections import Counter; print('choices:', dict(Counter(r['choice'] for r in recs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
