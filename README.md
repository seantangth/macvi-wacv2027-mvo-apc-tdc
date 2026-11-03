# MaCVi @ WACV 2027 — Maritime Visual Odometry: reproduction repository of team **APC_TDC**

Reproduces the JSON bundle of our **final entry** — macvi.org dashboard submission **#19363**, file `MVO_sub_v028_ensspeedwide-k07-vis5-k10_20261028.json`, leaderboard drift **12.57 %** —
from the released test frames. Method summary: [`METHOD.md`](METHOD.md). Factsheet: sent to the organizers (commit and links below).

| item | value |
|---|---|
| team / uploader | APC_TDC / macvi.org user 1802 (Tze-Hsiang Tang) |
| final entry | submission #19363, Eligible, 20/20 clips scored, coverage 100 % |
| SHA-256 of the bundle | `76bbd5836ee27f19b41bbf70914c2d6eccad597feef46a6848f05fd6141d668e` (copy in `expected/`) |
| code | https://github.com/seantangth/macvi-wacv2027-mvo-apc-tdc (public, MIT) — commit recorded in the factsheet |
| weights / cache | committed in this repository: `weights/` (our speed models, 1.8 MB), `artifacts/masks_v4/`, `cache/` (verified intermediates, 51 MB). `weights/` and `cache/` are also mirrored byte for byte on Hugging Face, https://huggingface.co/seantangth/macvi-wacv2027-mvo-apc-tdc (revision `ccb1af6c345f6379cabb5bccd61258aebb26f40e`; `artifacts/` is not mirrored); the scripts read the copies in this repository. The external checkpoints (VGGT-1B, RAFT-large, DPVO) are downloaded from their official sources and checked by SHA-256 (§3) |

The method is **not** a new network. It is a composition of pretrained models and classical estimators that recovers, for every clip, a *heading* ψ(t) and a *relative speed* s(t) and integrates them
on the water plane, x(t+Δt) = x(t) + s·Δt·[cos ψ, sin ψ]. The leaderboard metric aligns every clip with one global Sim(3) transform, so absolute scale does not matter: only the heading and the
*relative* speed profile do. Inference inputs are only the monocular video, the released (nominal) calibration and the decoded-frame timestamps — no GPS/IMU/AIS/speed/heading of any kind.

## 1. Reproduce (three paths)

```bash
git clone https://github.com/seantangth/macvi-wacv2027-mvo-apc-tdc && cd macvi-wacv2027-mvo-apc-tdc
pip install -r requirements.txt        # numpy 1.26.4, opencv-python-headless 4.10, torch (CPU is enough for A0 and A)
bash reproduce/reproduce_all.sh A0     # or A, or B
```

| path | what runs | hardware / time | result |
|---|---|---|---|
| **A0** | final assembly (policy → splice → official packer) from the *cached* intermediates in `cache/stage60` (+ download of the 1.3 GB test package for the manifest and timestamps) | CPU, ~1 min (16 s on a Colab VM) | **byte-identical** on macOS arm64; on Linux x86 deviation 0.0000 % (largest coordinate difference 8.5·10⁻¹⁴ m, the JSON text differs in the last float digits) |
| **A** | every CPU stage from the test **videos** (tracker, speed models, headings, policy, pack); the four GPU-stage outputs come from `cache/gpu` | CPU, ~10 min on a 10-core laptop (Apple M4; measured 9.6 min from a clean clone), 23 min on a 12-vCPU Colab VM | **byte-identical** on macOS arm64; on Linux x86 mean deviation 0.86 % (max 3.44 %): the OpenCV horizon detector is not bit-reproducible across CPU architectures, see §5 |
| **B** | everything from the test videos, GPU stages (VGGT windows, focal probe, RAFT flow, DPVO) re-run on one GPU by `reproduce/gpu_box.sh` | 1 GPU (A100-40GB): 2 h 10 min on a Colab VM including the environment build | not byte-identical; mean deviation 1.0–1.1 % (max 3.44 %), see §5; printed by `reproduce/compare_bundles.py` |

Each path ends with `reproduce/compare_bundles.py`: SHA-256 of the reproduced bundle next to the expected one and, per clip, the official drift metric of the reproduction scored against the
original trajectory (0 % = identical). The output is `work/final/final_bundle.json`.

## 2. Pipeline and stage scripts

```
video + timestamps + nominal calibration
 ├─ [20] horizon-band KLT tracker (classical, CPU) ─ rotw / sel  (window-wise rotation → yaw)
 ├─ [10] VGGT-1B on 48-frame windows (8 fps, 518 px) ─ yaw      [11] VGGT focal probe → per-clip yaw gain (flir-VIS)
 ├─ [30] RAFT-large water-band flow statistics (10 fps, 640 px) ─ [31] speed profiles: rule proxy | ridge + gated-CNN ensemble
 └─ [40] DPVO baseline, clip_112 only ─ speed profile (used only if it passes a sanity gate)
          ↓ [50] heading fusion per camera group      ↓ [60] policy_v004: heading + speed → trajectory → bundle
```

| stage | script | needs | output | hardware | deterministic (macOS arm64; other platforms: §5) |
|---|---|---|---|---|---|
| 00 masks (optional) | `reproduce/00_masks.sh` | videos | static-structure masks (`artifacts/masks_v4`, shipped) | CPU | yes (byte-identical PNGs) |
| 10 VGGT windows | `reproduce/gpu_box.sh vggt` | videos | `work/vggt/windows/*.npz` | GPU | no (bf16) |
| 11 focal probe → gains | `gpu_box.sh probe`, `3_src/mvo_tools/make_gain_table.py` | frames | `work/vggt/gain_test.json` | GPU + CPU | no / yes |
| 20 tracker | `reproduce/20_tracker.sh` | videos | `work/tracker/solve/{rotw,sel,…}` | CPU | **yes** |
| 30 RAFT flow | `gpu_box.sh flow` | videos | `work/flow/*.npz` (12 clips) | GPU | no |
| 31 speed models | `reproduce/31_speed_models.sh` | flows, tracker diag | `work/speed/{vis5,wide_ens}` | CPU | **yes** |
| 40 DPVO | `gpu_box.sh dpvo` | video | `work/dpvo112` | GPU | no |
| 50 headings | `reproduce/50_headings.sh` | 10, 11, 20 | `work/hyaw/{rotw,f73w,fused_g,vggt_g}` | CPU | **yes** |
| 60 assemble + pack | `reproduce/60_assemble_pack.sh` | 31, 40, 50 | `work/final/final_bundle.json` | CPU | **yes** |

Every "yes" was verified by re-running the stage on the development machine (macOS 15, Apple M4) and comparing the outputs **byte for byte** with the files that produced the submitted bundle (`reproduce/verify_dirs.py`):
20/20 masks, 20/20 track tables, all 100 solve outputs, 20/20 `fused_g`, 20/20 `vggt_g`, 8/8 `f73w`, 17/17 RAFT proxies, 17/17 ridge predictions, 112/112 CNN profiles (retrained with the same seeds),
and the final bundle. Path A was additionally run end to end from a clean copy of this repository on that machine: byte-identical, 0.0000 % deviation on all 20 clips.

### What the policy does per camera group (all values fixed, identical for every clip of the group)

| group (test clips) | heading ψ(t) | relative speed s(t) |
|---|---|---|
| flir-VIS (11) | 0.7 · windowed VGGT-1B yaw (sign-aligned) + 0.3 · tracker `sel`; yaw increments × per-clip gain `clip(0.87 / probe_ratio, 0.6, 1.0)`; 2 s angle smoothing | constant, except 5 harbour clips (093, 094, 099, 107, 111): water-flow proxy, `s = (proxy/median)^1.0`, floor 0.15, 45 s smoothing |
| wide (8) | 0.7 · tracker `rotw` + 0.3 · windowed VGGT-1B yaw (sign-aligned, rotation scale 0.809) | 7 clips: ridge + gated-CNN ensemble, `s = (profile/median)^0.7`, floor 0.15, 45 s smoothing; clip_112: DPVO speed profile (20 s) if it passes the sanity gate, else constant |
| flir-IR (1) | tracker `rotw` | constant |

Per-clip fixed decisions (lists of clips; disclosed in the factsheet and visible in `reproduce/common.sh` and `reproduce/60_assemble_pack.sh`):
* **Heading fallback** `096, 104, 111` (flir-VIS) and `103` (wide): windowed VGGT alone instead of the fusion (tracker diagnostics: total turn of `sel` > 1.5× that of `rotw`, or low horizon-found fraction; 104 has lens droplets).
* **Gain of clip_098** set by hand to 0.78 (the probe rule would give 0.6; the clip has a ~2× optical zoom mid-clip). The other ten flir-VIS gains are a deterministic function of the probe.
* **Harbour clips** `093, 094, 099, 107, 111` take the water-flow speed: clips on which two independent water-speed witnesses agreed (r ≥ 0.6); 094 was added after viewing the video (marina entry).

## 3. External models and checkpoints

| model | use | source | SHA-256 / version | licence |
|---|---|---|---|---|
| VGGT-1B (Wang et al., CVPR 2025) | windowed yaw, focal probe | Hugging Face `facebook/VGGT-1B`, `model.safetensors` | `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` | CC BY-NC 4.0 |
| RAFT-large (Teed & Deng, ECCV 2020) | water-band flow statistics | torchvision 0.20.1 `Raft_Large_Weights.C_T_SKHT_V2` (`raft_large_C_T_SKHT_V2-ff5fadd5.pth`) | `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322` | BSD-3 |
| DPVO (Teed, Lipson, Deng, NeurIPS 2023) | speed profile of clip_112 only | `princeton-vl/DPVO` @ `859bbbf`, `dpvo.pth` | `30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12` | MIT |
| ridge model (ours) | wide speed | `weights/ridge_train46.json` | `57e0057e969d36f5f45dcc467359fa67ba1e6b0be709fe8236ee672a6e2243cf` | MIT |
| gated CNN (ours), seeds 0–4 | wide speed | `weights/cnn_gate/seed*.pt` (377 KB each) | `weights/SHA256SUMS` | MIT |

No external dataset is used. Our own weights are fitted on the **training split only** (`reproduce/train_speed_models.sh`; refit of the ridge model is byte-identical, retraining the CNN with the same seeds reproduces
all 112 speed profiles byte-identically). `weights/external_weights.yaml` lists the public checkpoints.

Hugging Face mirror of `weights/` and `cache/` (public model repository, one upload commit): https://huggingface.co/seantangth/macvi-wacv2027-mvo-apc-tdc at revision `ccb1af6c345f6379cabb5bccd61258aebb26f40e`. Check that it equals the copies here (prints nothing; run on 2026-11-03):

```bash
hf download seantangth/macvi-wacv2027-mvo-apc-tdc --revision ccb1af6c345f6379cabb5bccd61258aebb26f40e --local-dir hf_mirror
diff -r hf_mirror/weights weights && diff -r hf_mirror/cache cache
```

## 4. Scalar constants fitted on the training split

`0.809` = median Theil–Sen slope of GPS course increments vs VGGT yaw increments over the 15 qualifying training wide clips (`3_src/mvo_tools/fit_yaw_gain.py`); `0.87` = typical focal-probe ratio of the 8 training flir-VIS clips
(leave-one-out validated on those clips).

## 5. Determinism, expected deviation and what was verified where

Deviation = official drift metric of the reproduced trajectory scored against the submitted trajectory of the same clip (`reproduce/compare_bundles.py`; 0 % = identical, mean over the 20 clips).

| run | machine | against the submitted bundle (#19363) |
|---|---|---|
| A0 | macOS 15, Apple M4 (development machine) | byte-identical |
| A0 | Linux x86_64, Colab VM (Ubuntu 24.04, 16 s) | 0.0000 % on all 20 clips; largest coordinate difference 8.5·10⁻¹⁴ m (the JSON text differs in the last float digits) |
| A | macOS 15, Apple M4, clean clone, 9.6 min | byte-identical, 0.0000 % |
| A | Linux x86_64, Colab VM (12 vCPU, 23 min) | mean **0.86 %**, max 3.44 % (clip_093) |
| B | Colab VM, A100-SXM4-40GB, 2 h 10 min, commit `3212000`; DPVO build failed there (CUDA 13 toolkit), cached DPVO output used | mean **1.00 %**, max 3.44 % |
| B + DPVO | same VM, clip_112 re-run with the final `gpu_box.sh` (commit `7c5a82f`, 3 min) and the bundle re-assembled | mean **1.09 %**, max 3.44 % (clip_112 1.84 %) |

The leaderboard score of a reproduction cannot be computed locally (the test ground truth is not public). A mean trajectory deviation of ~1 % means that a reproduced bundle is expected to score within about one percentage point of 12.57 %; **path A0 gives the exact bundle** from the intermediates that produced it.

Where the deviations come from (Colab run; `reproduce/compare_intermediates.py` and file-by-file comparisons; logs and a summary in [`verification/colab_a100_20261103/`](verification/colab_a100_20261103/SUMMARY.md)):

1. **VGGT windows and RAFT band statistics reproduce exactly**: 20/20 and 12/12 clips, mean |Δ yaw increment| 0.0000 °/s, mean |Δ flow| 0.0000 px/s. Same GPU model (A100-40GB) and PyTorch 2.5.1+cu121 as the original run; other GPU models or library versions may differ (bf16 autocast).
2. **Focal probe**: the ratio differs on 5 of the 11 flir-VIS clips (104, 107, 108, 109, 111; max |Δ| 0.113, mean 0.019) → different yaw gains → headings of clips 104, 107, 109, 111 deviate by 0.2–1.0 % (109: 1.3 %).
3. **Horizon tracker (OpenCV) is not bit-reproducible across CPU architectures.** On Linux x86 the horizon detector (Canny + probabilistic Hough on frames decoded by OpenCV's bundled FFmpeg) returns a different line than on macOS arm64 in a minority of frames, even with the identical `opencv-python-headless` 4.10.0 build (tested in a clean venv; the Colab GPU environment actually loads OpenCV 4.11.0 because VGGT pulls in `opencv-python`, with the same outcome). The tracked features and window-wise rotations then differ: tracker headings deviate by 0.8 % on average (max 4.0 %), the final bundle by 0.86 % (path A). The macOS outputs are the ones that produced the submitted bundle and are shipped in `cache/stage60` (path A0).
4. **DPVO** (clip_112 only) is documented as non-deterministic by the organisers: the re-run differs from the cached trajectory by 1.56 %, which moves clip_112 from 0.01 % to 1.84 %.
5. The speed models (RAFT rule, ridge, gated CNN) recomputed on Linux differ by ≤ 0.03 %.

Notes for other machines: the DPVO CUDA extensions must be built with an `nvcc` of the same major version as the PyTorch wheels (12.x); `gpu_box.sh` looks for `/usr/local/cuda-12*` and otherwise installs the CUDA 12.6 toolkit pieces from the NVIDIA apt repository (needs root or sudo; Colab ships CUDA 13). It also sets `MPLBACKEND=Agg` (DPVO imports pyplot; Jupyter/Colab shells export an inline backend). If DPVO cannot be built or run, path B warns and uses the cached DPVO output for clip_112. VGGT is installed at the commit of the verification run (`a288dd0`), and `gpu_box.sh env` warns when the downloaded VGGT-1B or RAFT-large checkpoint does not match the SHA-256 of the original run (§3).
Stage times on the Colab VM: test package download + environment 3 min, VGGT windows 81 min, focal probe 1 min, RAFT flow 23 min, DPVO inference 3 min (build ≈ 12 min, overlapped with the VGGT stage), CPU stages (tracker, speed models, headings, pack) 22 min.

## 6. Layout

```
3_src/mvo_tools/        our code (tracker, VGGT windows, RAFT band statistics, speed models, policy, helpers); 3_src/starter_upstream/ official LOOKOUT starter (37ca5c0, MIT)
reproduce/              stage scripts, reproduce_all.sh, compare_bundles.py, verify_dirs.py, train_speed_models.sh
weights/                our trained speed models + external_weights.yaml        artifacts/masks_v4/   the 20 static-structure masks
cache/                  verified intermediate outputs (stage-60 inputs, GPU-stage outputs)      expected/   the submitted bundle and its SHA-256
METHOD.md  repro.yaml  NOTICE.md  LICENSE  requirements.txt  requirements-gpu.txt
```
