#!/usr/bin/env python3
"""Infer timestamped trajectories from a public video/calibration package; never reads references."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from public_dataset import (asset_path, load_manifest, load_timestamps, select_clips,
                            sha256, validate_prediction, write_json)
from tracking import describe_environment, track


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, type=Path, help='public release_manifest.json')
    ap.add_argument('--split', default='val', help='train, val, test, comma-separated, or all')
    ap.add_argument('--clips', help='comma-separated anonymous IDs; overrides --split')
    ap.add_argument('--output', required=True, type=Path, help='new predictions run directory')
    ap.add_argument('--target-fps', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dpvo-buffer-size', type=int,
                    help='DPVO keyframe capacity; defaults to the upstream configuration')
    args = ap.parse_args(argv)
    if not np.isfinite(args.target_fps) or args.target_fps <= 0:
        ap.error('--target-fps must be positive and finite')
    if args.dpvo_buffer_size is not None and args.dpvo_buffer_size <= 0:
        ap.error('--dpvo-buffer-size must be positive')
    try:
        manifest = load_manifest(args.dataset)
        selected = select_clips(manifest, args.split, args.clips)
    except (OSError, ValueError, KeyError) as exc:
        ap.error(str(exc))
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('--output must be new or empty; choose a new directory for each run')
    dpvo, environment = describe_environment(args.dpvo_buffer_size)
    pred_dir = args.output / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    root = args.dataset.resolve().parent
    records = []
    run = dict(dataset_sha256=sha256(args.dataset), target_fps=args.target_fps,
               seed=args.seed, dpvo_buffer_size=args.dpvo_buffer_size, dpvo=dpvo,
               environment=environment, clips=records)
    print(f'INFERENCE_START: {len(selected)} clips, target_fps={args.target_fps}, '
          f'seed={args.seed}, dpvo_buffer_size={args.dpvo_buffer_size or "upstream default"}',
          flush=True)
    print(f"DPVO: revision={dpvo['revision']}, buffer_size={dpvo['buffer_size']}, "
          f"checkpoint_sha256={dpvo['checkpoint_sha256']}", flush=True)
    for number, clip in enumerate(selected, 1):
        key = clip['key']
        record = dict(key=key, split=clip['split'])
        started = time.perf_counter()
        print(f'[{number}/{len(selected)}] {key}: starting inference', flush=True)
        try:
            video = asset_path(root, clip['video'])
            if not video.is_file():
                record['status'] = 'missing_video'
            else:
                frame_times = load_timestamps(root, clip)
                poses, indices, pre = track(str(video), clip['camera_mode'], args.target_fps,
                                           args.seed, calibration=clip['calibration'],
                                           frame_timestamps=frame_times,
                                           dpvo_buffer_size=args.dpvo_buffer_size)
                prediction = dict(key=key,
                    frame_indices=[pre.decoded_index(int(i)) for i in indices],
                    times=[float(pre.frame_time(int(i))) for i in indices],
                    positions=np.asarray(poses)[:, :3].tolist())
                validate_prediction(prediction, key, frame_times)
                write_json(pred_dir / f'{key}.json', prediction)
                record.update(status='predicted', n_poses=len(indices),
                              native_fps=pre.native_fps, stride=pre.stride,
                              effective_fps=pre.effective_fps)
        except Exception as exc:
            record.update(status='tracking_failed', reason=str(exc)[:300], error_log=None)
            log = args.output / 'logs' / f'{key}.log'
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(traceback.format_exc(), encoding='utf-8')
            except Exception as log_exc:
                # A failed log write must not abort the remaining clips.
                print(f'{key}: {exc}\nCould not write traceback to {log}: {log_exc}',
                      flush=True)
            else:
                record['error_log'] = log.relative_to(args.output).as_posix()
                print(f'{key}: {exc}\nFull traceback: {log}', flush=True)
        record['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        records.append(record)
        write_json(args.output / 'inference_manifest.json', run)
        print(f"[{number}/{len(selected)}] {key}: {record['status']} "
              f"({record['elapsed_seconds']:.1f}s)", flush=True)
    predicted = sum(r['status'] == 'predicted' for r in records)
    print(f'INFERENCE_COMPLETE: {len(records)} attempted, {predicted} predicted')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
