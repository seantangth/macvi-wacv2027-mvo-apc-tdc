#!/usr/bin/env bash
# Stage 60 (CPU, deterministic): intermediates -> final JSON bundle of the final entry (leaderboard #19363).
# Inputs (dirs; defaults point to $WORK, override with env):
#   H       heading runs: <H>/{rotw,f73w,fused_g,vggt_g}/predictions/clip_*.json
#   W_VIS   water-flow speed profile of the 5 harbour flir-VIS clips (raft_speed_proxy.py output)
#   W_WIDE  ensemble speed profile (ridge + gated net) of the wide clips
#   DP      DPVO run (clip_112 only is used)
# Output: $OUT/final_bundle.json (+ sha256) ; per-clip predictions in $OUT/assembled
. "$(dirname "$0")/common.sh"
H="${H:-$WORK/hyaw}"; W_VIS="${W_VIS:-$WORK/speed/vis5}"; W_WIDE="${W_WIDE:-$WORK/speed/wide_ens}"; DP="${DP:-$WORK/dpvo112}"; OUT="${OUT:-$WORK/final}"
rm -rf "${OUT:?}"; mkdir -p "$OUT"
policy() { # $1 out  $2 speed-run  $3 speed-modes  $4 k  $5 forced clips
  $PY 3_src/mvo_tools/policy_v004.py --dataset $T --split test --hyaw "$H" --dpvo "$DP" --out "$1" \
    --vis fused_g --wide f73w --guard-ratio 0 --max-rawspd 0.05 --smooth-mode angle --smooth-h 2 --override-clips $OVERRIDE --override-est vggt_g \
    --speed-run "$2" --speed-modes "$3" --speed-order dpvo,water --speed-smooth 45 --speed-shrink "$4" --speed-shrink-mode log --speed-floor 0.15 \
    --water-min-coh 9 --water-min-inl 999999 --water-min-inlr 9 --water-force-clips "$5" > "$1.log"; }
policy "$OUT/run_vis"  "$W_VIS"  flir-VIS 1.0 "$VIS5"     # 11 flir-VIS (5 harbour clips: water-flow speed, log blend k=1.0), IR clip, wide clip_112 (DPVO speed)
policy "$OUT/run_wide" "$W_WIDE" wide     0.7 "$WIDE7"    # 7 wide clips: ensemble speed, log blend k=0.7
mkdir -p "$OUT/assembled/predictions"
for f in "$OUT"/run_vis/predictions/clip_*.json; do c=$(basename "$f" .json); src="$OUT/run_vis"
  case ",$WIDE7," in *,$c,*) src="$OUT/run_wide";; esac; cp "$src/predictions/$c.json" "$OUT/assembled/predictions/"; done
$PY 3_src/mvo_tools/coverage_gate.py --dataset $T --predictions "$OUT/assembled" --split test | tail -1
$PY 3_src/starter_upstream/starter/pack_submission.py --dataset $T --predictions "$OUT/assembled" --output "$OUT/final_bundle.json" --split test | tail -1
$PY - "$OUT/final_bundle.json" <<'PY'
import hashlib, sys; h = hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(); open(sys.argv[1] + '.sha256', 'w').write(f'{h}  final_bundle.json\n'); print('sha256', h)
PY
