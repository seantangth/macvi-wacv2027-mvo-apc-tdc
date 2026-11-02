#!/usr/bin/env bash
# Stage 00 (CPU, optional): per-clip static-structure masks (own-vessel bow, rails, radar dome, burned-in overlays) of the 20 test videos.
# The 20 masks used for the final entry are shipped in artifacts/masks_v4/ (regenerating them with this command gives byte-identical PNGs).
. "$(dirname "$0")/common.sh"
OUTM="${OUTM:-$WORK/masks_v4}"; mkdir -p "$OUTM"
$PY 3_src/mvo_tools/static_mask.py --dataset $T --out "$OUTM" --split test > "$OUTM/static_mask.log" 2>&1; ls "$OUTM"/clip_*.png | wc -l
