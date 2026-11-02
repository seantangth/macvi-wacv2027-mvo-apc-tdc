#!/usr/bin/env bash
# Stage 20 (CPU, classical, deterministic): horizon-band KLT feature tracking of the 20 test videos, then window-wise rotation solve.
#   track : needs the test videos; 12 fps, 960-px width, static masks; flir-VIS clips use a wide band (0.45 H above / 0.10 H below the horizon, 800 features),
#           all other clips the default band (0.25 H above / 0.015 H below, 600 features).   ~10 min on a 10-core laptop
#   solve : --win-s 5.0 --thr-px-win 3.0 --sel-ratio 0.9, the RANSAC random stream is shared inside a shard, therefore the shard layout below
#           (3 shards per camera group, clips in manifest order) is part of the recipe.  ~1 min.   Output: $TRK/solve/{rot,rotw,essw,kd0w,sel,diag}
. "$(dirname "$0")/common.sh"
TRK="${TRK:-$WORK/tracker}"; M="${MASKS:-$ROOT/artifacts/masks_v4}"; OTH=clip_092,clip_097,clip_100,clip_101,clip_103,clip_105,clip_106,clip_110,clip_112
STAGE="${1:-all}"; mkdir -p "$TRK/vis/logs" "$TRK/oth/logs" "$TRK/solve/logs"
if [ "$STAGE" = all ] || [ "$STAGE" = track ]; then
  for i in 0 1 2 3 4; do
    $PY 3_src/mvo_tools/horizon_yaw_vo.py --dataset $T --split test --clips $VIS11 --out "$TRK/vis" --stage track --mask-dir "$M" --above 0.45 --below 0.10 --max-feat 800 --shard $i/5 > "$TRK/vis/logs/track$i.log" 2>&1 &
    $PY 3_src/mvo_tools/horizon_yaw_vo.py --dataset $T --split test --clips $OTH   --out "$TRK/oth" --stage track --mask-dir "$M" --shard $i/5 > "$TRK/oth/logs/track$i.log" 2>&1 &
  done; wait
fi
if [ "$STAGE" = all ] || [ "$STAGE" = solve ]; then
  for i in 0 1 2; do
    $PY 3_src/mvo_tools/horizon_yaw_vo.py --dataset $T --split test --clips $VIS11 --out "$TRK/vis" --solve-out "$TRK/solve" --stage solve --win-s 5.0 --thr-px-win 3.0 --sel-ratio 0.9 --shard $i/3 > "$TRK/solve/logs/solve_vis$i.log" 2>&1 &
    $PY 3_src/mvo_tools/horizon_yaw_vo.py --dataset $T --split test --clips $OTH   --out "$TRK/oth" --solve-out "$TRK/solve" --stage solve --win-s 5.0 --thr-px-win 3.0 --sel-ratio 0.9 --shard $i/3 > "$TRK/solve/logs/solve_oth$i.log" 2>&1 &
  done; wait
  for d in rot rotw essw kd0w sel; do echo "$d: $(ls "$TRK/solve/$d/predictions" | wc -l) clips"; done
fi
