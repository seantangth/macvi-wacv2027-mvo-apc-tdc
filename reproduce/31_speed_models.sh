#!/usr/bin/env bash
# Stage 31 (CPU): RAFT band statistics of the test clips -> speed profiles used by the final policy.
#   W_VIS  : water-flow rule proxy (nearest 3 water bands, min flow 10 px/s) of the 5 harbour flir-VIS clips          (raft_speed_proxy.py)
#   W_WIDE : ensemble of ridge (E029) and gated CNN (E036) profiles of the 7 wide clips                              (ensemble_speed.py)
# Inputs: FLOW = dir with the RAFT band statistics clip_*.npz of the test clips (stage 30), DIAG = tracker diag dir (stage 20; horizon row per clip),
#         MODELS = weights/ (ridge JSON + CNN weights).
. "$(dirname "$0")/common.sh"
FLOW="${FLOW:-$WORK/flow}"; DIAG="${DIAG:-$WORK/tracker/solve/diag}"; MODELS="${MODELS:-$ROOT/weights}"; S="${S:-$WORK/speed}"
rm -rf "${S:?}"; mkdir -p "$S"
# 1) RAFT rule proxy (all test clips that have flow) -> also the gate input of the CNN and the ridge feature
$PY 3_src/mvo_tools/raft_speed_proxy.py --flow-dir "$FLOW" --dataset $T --split test --out "$S/proxy" --diag-dir "$DIAG" --nearest 3 --min-flow 10 > "$S/proxy.log" 2>&1
mkdir -p "$S/vis5/predictions"; for c in ${VIS5//,/ }; do cp "$S/proxy/predictions/$c.json" "$S/vis5/predictions/"; done; cp "$S/proxy/inference_manifest.json" "$S/vis5/"
# 2) ridge regression (E029), fitted on the 46 training wide clips (weights/ridge_train46.json)
$PY 3_src/mvo_tools/learned_speed.py predict --model "$MODELS/ridge_train46.json" --flow-dir "$FLOW" --dataset $T --split test --hmani "$S/proxy/inference_manifest.json" --proxy-manifest "$S/proxy/inference_manifest.json" --out "$S/ridge" > "$S/ridge.log" 2>&1
# 3) gated CNN (E036): inference only, weights = 5 seeds trained on the 68 training clips (weights/cnn_gate/seed*.pt)
$PY 3_src/mvo_tools/speednet.py --flow-dirs "$FLOW" --hmani "$S/proxy/inference_manifest.json" --proxy-dirs "$S/proxy" --out-root "$S/cnn" --mode gate --seeds 5 --load-weights "$MODELS/cnn_gate" > "$S/cnn.log" 2>&1
# 4) ensemble
$PY 3_src/mvo_tools/ensemble_speed.py --ridge "$S/ridge" --net "$S/cnn/test" --out "$S/wide_ens" | tail -1
echo "vis5: $(ls "$S/vis5/predictions" | wc -l) clips, wide_ens: $(ls "$S/wide_ens/predictions" | wc -l) clips"
