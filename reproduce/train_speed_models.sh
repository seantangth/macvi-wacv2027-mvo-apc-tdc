#!/usr/bin/env bash
# OPTIONAL (not needed to reproduce the bundle; the trained models are shipped in artifacts/speed_models):
# retrain the two learned relative-speed models on the TRAIN split only (GPS reference used as training label, as allowed by the rules).
# Inputs: the dev package (1_data/raw/release), RAFT band statistics of the dev clips (stage 30 applied to the dev clips; 95 clips, ~1.5 h on one A100),
#         RAFT rule proxies of the same clips (raft_speed_proxy.py, recipe of stage 31) and the horizon fraction of each clip (tracker diag or manifest).
# Both commands were re-run on the development machine from the stored band statistics and reproduce the shipped model (ridge: byte-identical JSON;
# CNN: byte-identical test/val/train speed profiles).
. "$(dirname "$0")/common.sh"
FLOW_DIRS="${FLOW_DIRS:?dirs with clip_*.npz band statistics of the dev clips}"; HMANI="${HMANI:?inference_manifest.json files of the rule proxies (carry the horizon of each clip)}"
PROXY_ALL="${PROXY_ALL:?run dir(s) with the rule-proxy predictions of all dev clips}"; OUT="${OUT:-$WORK/train_speed}"; D=1_data/raw/release/release_manifest.json; mkdir -p "$OUT"
TRAIN_WIDE46="${TRAIN_WIDE46:-clip_002,clip_003,clip_004,clip_006,clip_010,clip_011,clip_013,clip_014,clip_017,clip_021,clip_023,clip_026,clip_027,clip_029,clip_030,clip_031,clip_033,clip_034,clip_035,clip_036,clip_041,clip_043,clip_044,clip_047,clip_049,clip_051,clip_054,clip_058,clip_060,clip_061,clip_062,clip_063,clip_065,clip_067,clip_068,clip_070,clip_071,clip_073,clip_074,clip_078,clip_079,clip_081,clip_083,clip_085,clip_087,clip_091}"
# (1) ridge regression on log band flows (46 training wide clips; 45 s smoothed GPS speed as label; alpha 100; symmetric left/right features)
$PY 3_src/mvo_tools/learned_speed.py --sym fit --flow-dir "${FLOW_DIRS%% *}" --dataset $D --split train --clips "$TRAIN_WIDE46" --hmani $HMANI --out "$OUT/ridge_train46.json"
# (2) gated dilated 1-D CNN (68 training clips of all camera types, 200 epochs x 5 seeds, CPU); weights -> $OUT/cnn_gate/seed*.pt
$PY 3_src/mvo_tools/speednet.py --flow-dirs $FLOW_DIRS --hmani $HMANI --out-root "$OUT/cnn" --epochs 200 --seeds 5 --mode gate --proxy-dirs $PROXY_ALL --save-weights "$OUT/cnn_gate"
