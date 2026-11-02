#!/usr/bin/env bash
# Stage 50 (CPU, deterministic): VGGT windows + horizon-tracker headings + focal-probe gains -> the four heading runs used by the final policy:
#   rotw (IR clip_100) | f73w (8 wide) | fused_g (flir-VIS) | vggt_g (fallback clips 096/104/111/103)       -> $H/<name>/predictions/clip_*.json
# Inputs: WIN = dir with VGGT window npz (stage 10), TRK = tracker solve output with {rotw,sel}/predictions (stage 20), GAIN = gain_test.json (stage 11)
. "$(dirname "$0")/common.sh"
WIN="${WIN:-$WORK/vggt/windows}"; TRK="${TRK:-$WORK/tracker/solve}"; GAIN="${GAIN:-$WORK/vggt/gain_test.json}"; H="${H:-$WORK/hyaw}"
rm -rf "${H:?}"; mkdir -p "$H"; B="$H/_build"; mkdir -p "$B"
cp -R "$TRK/rotw" "$H/rotw"; cp -R "$TRK/sel" "$H/sel"
$PY 3_src/mvo_tools/windows_to_heading.py --windows "$WIN" --out "$B/vggt_neg" --sign -1 --min-clip 92 > "$B/windows_to_heading.log"
$PY 3_src/mvo_tools/fuse_headings.py --dataset $T --split test --runs "$B/vggt_neg" "$H/sel" --weights 0.7 0.3 --out "$B/fused" > /dev/null
$PY 3_src/mvo_tools/scale_heading.py --dataset $T --split test --run "$B/fused" --out "$H/fused_g" --gain-json "$GAIN" > /dev/null
$PY 3_src/mvo_tools/scale_heading.py --dataset $T --split test --run "$B/vggt_neg" --out "$H/vggt_g" --gain-json "$GAIN" > /dev/null
# wide: 0.7 tracker (rotw) + 0.3 windowed VGGT course (sign-aligned to rotw, rotation scale 0.809 fitted on the 46 training wide clips)
$PY 3_src/mvo_tools/align_heading_sign.py --ref "$H/rotw" --run "$B/vggt_neg" --out "$B/v48_al" --clips $WIDE8 > "$B/align.log"
$PY 3_src/mvo_tools/scale_heading.py --dataset $T --split test --run "$B/v48_al" --out "$B/v48g" --gain 0.809 --clips $WIDE8 > /dev/null
$PY 3_src/mvo_tools/fuse_headings.py --dataset $T --split test --clips $WIDE8 --runs "$H/rotw" "$B/v48g" --weights 0.7 0.3 --out "$H/f73w" > /dev/null
for d in rotw sel fused_g vggt_g f73w; do echo "$d: $(ls "$H/$d/predictions" | wc -l) clips"; done
