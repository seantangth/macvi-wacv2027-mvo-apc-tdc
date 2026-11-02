#!/usr/bin/env bash
# Reproduce the JSON bundle of the final entry (leaderboard submission #19363, drift 12.57 %).
#   bash reproduce/reproduce_all.sh A0   final assembly only, from cached intermediates (no video, no GPU, ~1 min)       -> byte-identical expected
#   bash reproduce/reproduce_all.sh A    CPU stages from the released test videos (tracker, speed models, headings, policy) + cached GPU-stage outputs (~15 min CPU)
#   bash reproduce/reproduce_all.sh B    everything from the released test videos, GPU stages re-run on one GPU (reproduce/gpu_box.sh)
# Output: $WORK/final/final_bundle.json, compared with expected/ at the end (compare_bundles.py).
set -euo pipefail
. "$(dirname "$0")/common.sh"
MODE="${1:-A}"; CACHE="$ROOT/cache"; mkdir -p "$WORK"
case "$MODE" in
  A0) bash reproduce/00_get_test_data.sh   # manifest + timestamps (the final assembly needs no video)
      mkdir -p "$WORK/hyaw" "$WORK/speed" && cp -R "$CACHE/stage60/hyaw/." "$WORK/hyaw/" && cp -R "$CACHE/stage60/speed/." "$WORK/speed/" && mkdir -p "$WORK/dpvo112" && cp -R "$CACHE/stage60/dpvo112/." "$WORK/dpvo112/" ;;
  A|B)
    bash reproduce/00_get_test_data.sh
    if [ "$MODE" = A ]; then mkdir -p "$WORK/vggt" "$WORK/flow" "$WORK/dpvo112"; cp -R "$CACHE/gpu/vggt/." "$WORK/vggt/"; cp "$CACHE/gpu/flow/"*.npz "$WORK/flow/"; cp -R "$CACHE/gpu/dpvo112/." "$WORK/dpvo112/"
    else
      WORK="$WORK" bash reproduce/gpu_box.sh all          # VGGT windows + focal probe + RAFT flows + DPVO (clip_112)
      [ -x "$ROOT/.venvV/bin/python" ] && export PY="$ROOT/.venvV/bin/python"   # the venv built by gpu_box.sh also has numpy/opencv/torch for the CPU stages
      mkdir -p "$WORK/vggt"; $PY 3_src/mvo_tools/make_gain_table.py --probe "$WORK/vggt/focal_probe_test.json" --out "$WORK/vggt/gain_test.json" > /dev/null
    fi
    [ -f "$WORK/vggt/gain_test.json" ] || $PY 3_src/mvo_tools/make_gain_table.py --probe "$WORK/vggt/focal_probe_test.json" --out "$WORK/vggt/gain_test.json" > /dev/null
    bash reproduce/20_tracker.sh all                       # CPU: tracking + solve  (~10 min + 1 min)
    FLOW="$WORK/flow" DIAG="$WORK/tracker/solve/diag" bash reproduce/31_speed_models.sh
    WIN="$WORK/vggt/windows" TRK="$WORK/tracker/solve" GAIN="$WORK/vggt/gain_test.json" bash reproduce/50_headings.sh ;;
  *) echo "usage: reproduce_all.sh A0|A|B"; exit 2 ;;
esac
bash reproduce/60_assemble_pack.sh
$PY reproduce/compare_bundles.py "$WORK/final/final_bundle.json" expected/MVO_sub_v028_ensspeedwide-k07-vis5-k10_20261028.json || true
