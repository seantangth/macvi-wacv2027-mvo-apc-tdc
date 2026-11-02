#!/usr/bin/env bash
# Sourced by every stage script. Runs from the repository root; all code lives under 3_src (same layout as the development tree).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/3_src:$ROOT/3_src/starter_upstream/starter${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python3}"
T="${T:-1_data/raw/lookout-vo-test-v0.1/release_manifest.json}"     # released test manifest (20 clips, no reference)
WORK="${WORK:-$ROOT/work}"                                            # regenerated intermediates
# test clips by camera mode (from the released manifest)
VIS11=clip_093,clip_094,clip_095,clip_096,clip_098,clip_099,clip_104,clip_107,clip_108,clip_109,clip_111
VIS5=clip_093,clip_094,clip_099,clip_107,clip_111      # flir-VIS harbour clips that take the water-flow speed (fixed list, see README)
WIDE8=clip_092,clip_097,clip_101,clip_103,clip_105,clip_106,clip_110,clip_112
WIDE7=clip_092,clip_097,clip_101,clip_103,clip_105,clip_106,clip_110   # wide clips that take the ensemble water-flow speed (112 keeps its DPVO profile)
OVERRIDE=clip_096,clip_104,clip_111,clip_103            # heading fallback to windowed VGGT alone (see README, section "Per-clip fallbacks")
