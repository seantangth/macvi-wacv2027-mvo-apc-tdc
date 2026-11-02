#!/usr/bin/env bash
# GPU stages of the reproduction (one NVIDIA GPU, CUDA 12.x driver, Ubuntu 22.04; developed on A100-40GB, verified on H100-80GB).
# Run from the repository root:   bash reproduce/gpu_box.sh all         (or one of: env data vggt probe flow dpvo)
#   env   : python venv .venvV (torch 2.5.1+cu121, VGGT) and .venvD (DPVO 859bbbf + checkpoint)
#   data  : released test package (macvi.org) -> data/lookout-vo-test-v0.1  (SHA-256 checked)
#   vggt  : windowed VGGT-1B (8 fps, 518 px, 48-frame windows, stride 24) on the 20 test clips  -> $WORK/vggt/windows/clip_*.npz   (stage 10)
#   probe : VGGT focal probe of the 11 flir-VIS clips                                          -> $WORK/vggt/focal_probe_test.json (stage 11)
#   flow  : RAFT-large (torchvision C_T_SKHT_V2) water-band statistics (10 fps, 640 px) on the 12 clips that take a water-flow speed -> $WORK/flow (stage 30)
#   dpvo  : DPVO (official baseline, seed 0, 8 fps) on clip_112 only                           -> $WORK/dpvo112                    (stage 40)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; WORK="${WORK:-$ROOT/work}"; mkdir -p "$WORK/logs"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1 MPLBACKEND=Agg PYTHONPATH="$ROOT/3_src:$ROOT/3_src/starter_upstream/starter"   # MPLBACKEND: DPVO imports pyplot; Jupyter/Colab shells export an inline backend that fails outside a notebook
D=1_data/raw; TESTDIR=$D/lookout-vo-test-v0.1; T=$TESTDIR/release_manifest.json; PYV=$ROOT/.venvV/bin/python; PYD=$ROOT/.venvD/bin/python
VIS11="clip_093 clip_094 clip_095 clip_096 clip_098 clip_099 clip_104 clip_107 clip_108 clip_109 clip_111"
OTH9="clip_100 clip_092 clip_097 clip_101 clip_103 clip_105 clip_106 clip_110 clip_112"
FLOW12="clip_093 clip_094 clip_099 clip_107 clip_111 clip_092 clip_097 clip_101 clip_103 clip_105 clip_106 clip_110"
log() { echo "== $* $(date -Is)"; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO=sudo   # Colab runs as root without sudo
do_env() {
  log "[env]"
  $SUDO apt-get update -qq >/dev/null 2>&1; $SUDO apt-get install -y -qq ffmpeg libgl1 build-essential unzip git >/dev/null 2>&1 || true   # Python 3.10 comes from uv (no apt package needed)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"
  [ -x "$PYV" ] || uv venv --python 3.10 .venvV >/dev/null || return 1
  uv pip install --python "$PYV" -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || return 1
  uv pip install --python "$PYV" -q "numpy<2" opencv-python-headless==4.10.0.84 pillow huggingface_hub einops safetensors tqdm scipy || return 1
  uv pip install --python "$PYV" -q "git+https://github.com/facebookresearch/vggt.git" || return 1
  "$PYV" - <<'PY' || return 1
import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
from vggt.models.vggt import VGGT; m = VGGT.from_pretrained('facebook/VGGT-1B'); print('VGGT-1B OK', sum(p.numel() for p in m.parameters()) // 1_000_000, 'M params')
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; r = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False); print('RAFT-large OK', sum(p.numel() for p in r.parameters()) // 1_000_000, 'M params')
PY
  uv pip freeze --python "$PYV" > "$WORK/logs/pip_freeze_venvV.txt" 2>/dev/null
  { echo "# sha256 of the external checkpoints used by this run ($(date -Is))"
    for f in "$HOME"/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors; do [ -e "$f" ] && echo "$(sha256sum "$(readlink -f "$f")" | cut -c1-64)  VGGT-1B model.safetensors (Hub revision $(basename "$(dirname "$f")"))"; done
    for f in "$HOME"/.cache/torch/hub/checkpoints/raft_large_*.pth; do [ -e "$f" ] && echo "$(sha256sum "$f" | cut -c1-64)  torchvision $(basename "$f")"; done; } > "$WORK/logs/external_weights_sha256.txt" 2>/dev/null
  cat "$WORK/logs/external_weights_sha256.txt"; return 0
}
cuda12_home() {   # prints a CUDA_HOME whose nvcc has the same major version as the PyTorch wheels (12); returns 1 if there is none
  local d v
  for d in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12*; do
    [ -n "$d" ] && [ -x "$d/bin/nvcc" ] || continue
    v=$("$d/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    [ "$v" = 12 ] && { echo "$d"; return 0; }
  done; return 1
}
do_dpvo_env() {   # DPVO baseline environment (needs nvcc 12.x, the major version of the cu121 PyTorch wheels); mirrors the official QUICKSTART pins
  log "[dpvo env]"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"
  [ -x "$PYD" ] || uv venv --python 3.10 --seed .venvD >/dev/null || return 1
  "$PYD" -m pip install -q --upgrade pip wheel setuptools ninja
  "$PYD" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || return 1
  "$PYD" -m pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html || "$PYD" -m pip install -q torch-scatter
  "$PYD" -m pip install -q numpy==1.26.4 opencv-python==4.10.0.84 numba tqdm einops pypose kornia plyfile evo yacs gdown || return 1
  [ -d DPVO ] || git clone -q --recursive https://github.com/princeton-vl/DPVO.git
  git -C DPVO checkout -q 859bbbfdac6c6185f345003b3c473901fcd13ace && git -C DPVO submodule update --init --recursive -q
  [ -d DPVO/thirdparty/eigen-3.4.0 ] || (mkdir -p DPVO/thirdparty && cd DPVO/thirdparty && curl -fsSL -o eigen-3.4.0.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip -q eigen-3.4.0.zip && rm eigen-3.4.0.zip)
  CH=$(cuda12_home) || {   # e.g. Colab ships CUDA 13: add the CUDA 12.6 toolkit pieces from the NVIDIA apt repository (root or sudo needed)
    echo "no nvcc 12.x found: installing the CUDA 12.6 toolkit pieces (apt)"
    $SUDO apt-get install -y -qq --no-install-recommends cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-cccl-12-6 libcublas-dev-12-6 libcusparse-dev-12-6 libcusolver-dev-12-6 >/dev/null 2>&1
    CH=$(cuda12_home); } || true
  [ -n "${CH:-}" ] || { echo "DPVO needs nvcc 12.x (PyTorch is built for CUDA 12.1) and none was found: DPVO cannot be built here"; return 1; }
  export CUDA_HOME="$CH" PATH="$CH/bin:$PATH"; echo "DPVO build with $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/CUDA \1/p') in $CUDA_HOME"
  export TORCH_CUDA_ARCH_LIST="$("$PYD" -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))")"
  "$PYD" -m pip install --no-build-isolation ./DPVO > "$WORK/logs/dpvo_build.log" 2>&1 || { tail -30 "$WORK/logs/dpvo_build.log"; return 1; }
  "$PYD" -m pip install -q numpy==1.26.4 opencv-python==4.10.0.84
  if [ ! -f DPVO/dpvo.pth ]; then "$PYD" -m gdown 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX -O DPVO/models.zip && (cd DPVO && unzip -o -q models.zip dpvo.pth); fi
  echo "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12  DPVO/dpvo.pth" | sha256sum -c - || return 1
  "$PYD" -m pip freeze > "$WORK/logs/pip_freeze_venvD.txt" 2>/dev/null; return 0
}
do_data() {
  log "[data]"; mkdir -p "$D" && cd "$D" || return 1; f=lookout-vo-test-v0.1.tar
  [ -f "$f" ] || curl -fL -C - -O "https://macvi.org/downloads/lookout/v0.1/$f" || { cd "$ROOT"; return 1; }
  curl -fsSL -O "https://macvi.org/downloads/lookout/v0.1/$f.sha256" && sha256sum -c "$f.sha256" || { cd "$ROOT"; return 1; }
  [ -d lookout-vo-test-v0.1 ] || tar -xf "$f"; cd "$ROOT"; ls $TESTDIR/videos | wc -l
}
frames() {  # clip fps side outdir
  [ -f "$4/$1/frames_manifest.json" ] || "$PYV" 3_src/mvo_tools/extract_frames.py --dataset $T --split test --clips "$1" --fps "$2" --out "$4" --max-side "$3" > /dev/null 2>&1
  [ -f "$4/$1/frames_manifest.json" ]
}
do_vggt() {
  log "[vggt windows]"; mkdir -p "$WORK/vggt/windows" "$WORK/vggt/diag" "$WORK/vggt/raw/predictions" "$WORK/vggt/heading/predictions"
  for k in $VIS11 $OTH9; do
    [ -f "$WORK/vggt/windows/$k.npz" ] && { echo "$k: done"; continue; }
    frames $k 8 518 "$WORK/frames8" || { echo "$k: frames FAILED"; return 1; }
    "$PYV" 3_src/mvo_tools/vggt_window_vo.py --frames "$WORK/frames8/$k" --timing $TESTDIR/timing/$k.json --out "$WORK/vggt/raw/predictions/$k.json" \
        --out-heading "$WORK/vggt/heading/predictions/$k.json" --save-windows "$WORK/vggt/windows/$k.npz" --diag "$WORK/vggt/diag/$k.json" --window 48 --stride 24 > "$WORK/vggt/diag/$k.log" 2>&1 || { echo "$k: VGGT FAILED"; tail -3 "$WORK/vggt/diag/$k.log"; return 1; }
    echo "$k: ok"
  done
}
do_probe() { log "[focal probe]"; "$PYV" 3_src/mvo_tools/focal_probe.py --frames-root "$WORK/frames8" --clips "$(echo $VIS11 | tr ' ' ',')" --out "$WORK/vggt/focal_probe_test.json" 2>&1 | grep -v -i warn | tail -13; }
do_flow() {
  log "[raft flow]"; mkdir -p "$WORK/flow/log"
  for k in $FLOW12; do
    [ -f "$WORK/flow/$k.npz" ] && { echo "$k: done"; continue; }
    frames $k 10 640 "$WORK/frames10" || { echo "$k: frames FAILED"; return 1; }
    "$PYV" 3_src/mvo_tools/raft_water_flow.py --frames "$WORK/frames10/$k" --timing $TESTDIR/timing/$k.json --out "$WORK/flow/$k.npz" > "$WORK/flow/log/$k.log" 2>&1 || { echo "$k: RAFT FAILED"; tail -3 "$WORK/flow/log/$k.log"; return 1; }
    tail -1 "$WORK/flow/log/$k.log" | cut -c1-160; rm -rf "$WORK/frames10/$k"
  done
}
do_dpvo() {
  log "[dpvo clip_112]"; export DPVO_DIR="$ROOT/DPVO"
  "$PYD" 3_src/starter_upstream/starter/infer.py --dataset $T --split test --clips clip_112 --target-fps 8 --seed 0 --output "$WORK/dpvo112" 2>&1 | tail -3
}
case "${1:-all}" in
  env) do_env ;; data) do_data ;; vggt) do_vggt ;; probe) do_probe ;; flow) do_flow ;; dpvo) do_dpvo ;; dpvo_env) do_dpvo_env ;;
  all) do_env && do_data && { do_dpvo_env > "$WORK/logs/dpvo_env.log" 2>&1 & DP=$!; } && do_vggt && do_probe && do_flow && { wait $DP; tail -2 "$WORK/logs/dpvo_env.log"; do_dpvo; } ;;
esac
