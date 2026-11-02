#!/usr/bin/env python3
"""Validate and pack trajectory predictions without reading any reference labels."""
import argparse
import json
from pathlib import Path

from public_dataset import load_manifest, load_timestamps, read_json, select_clips, validate_prediction

MAX_BYTES = 64 * 1024 * 1024


def pack(manifest_path, predictions, output, split='test'):
    manifest_path, predictions, output = map(Path, (manifest_path, predictions, output))
    clips = select_clips(load_manifest(manifest_path), split=split)
    if (predictions / 'predictions').is_dir():
        predictions = predictions / 'predictions'
    bundle = {}
    for clip in clips:
        key = clip['key']
        pred = read_json(predictions / f'{key}.json')
        validate_prediction(pred, key, load_timestamps(manifest_path.parent, clip))
        bundle[key] = {field: pred[field] for field in ('key', 'frame_indices', 'times', 'positions')}
    encoded = (json.dumps(bundle, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(encoded) > MAX_BYTES:
        raise ValueError('submission exceeds the 64 MB server limit')
    # Do not overwrite a previous submission or write a partial bundle on validation failure.
    with output.open('xb') as fh:
        fh.write(encoded)
    return len(bundle), len(encoded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path, help='release_manifest.json')
    p.add_argument('--predictions', required=True, type=Path, help='Run directory or predictions/ directory')
    p.add_argument('--output', required=True, type=Path, help='New .json submission file')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = p.parse_args()
    try:
        count, size = pack(args.dataset, args.predictions, args.output, args.split)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        p.exit(1, f'Cannot pack submission: {exc}\n')
    print(f'Packed {count} clips ({size:,} bytes) into {args.output}.')
    print('Format checked. The server separately checks scoreability and >=95% supported coverage on every clip.')


if __name__ == '__main__':
    main()
