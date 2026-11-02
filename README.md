# MaCVi @ WACV 2027 — Maritime Visual Odometry: reproduction repository of team **APC_TDC**

Reproduces the JSON bundle of our **final entry** — macvi.org dashboard submission **#19363**, file `MVO_sub_v028_ensspeedwide-k07-vis5-k10_20261028.json`, leaderboard drift **12.57 %** —
from the released test frames. Method summary: [`METHOD.md`](METHOD.md). Factsheet: sent to the organizers (commit and links below).

| item | value |
|---|---|
| team / uploader | APC_TDC / macvi.org user 1802 (Tze-Hsiang Tang) |
| final entry | submission #19363, Eligible, 20/20 clips scored, coverage 100 % |
| SHA-256 of the bundle | `76bbd5836ee27f19b41bbf70914c2d6eccad597feef46a6848f05fd6141d668e` (copy in `expected/`) |
| code | https://github.com/seantangth/macvi-wacv2027-mvo-apc-tdc (public, MIT) — commit recorded in the factsheet |
| weights / cache mirror | https://huggingface.co/seantangth/macvi-wacv2027-mvo-apc-tdc (`weights/` and `cache/`, identical SHA-256) |

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
| **A0** | final assembly (policy → splice → official packer) from the *cached* intermediates in `cache/stage60` (+ download of the 1.3 GB test package for the manifest and timestamps) | CPU, ~1 min | **byte-identical** bundle |
| **A** | every CPU stage from the test **videos** (tracker, speed models, headings, policy, pack); the four GPU-stage outputs come from `cache/gpu` | CPU, ~10 min on a 10-core laptop (Apple M4; measured 9.6 min from a clean clone) | **byte-identical** bundle |
| **B** | everything from the test videos, GPU stages (VGGT windows, focal probe, RAFT flow, DPVO) re-run on one GPU by `reproduce/gpu_box.sh` | 1 GPU (A100-40GB), see §5 | near-identical; deviation printed by `reproduce/compare_bundles.py` |

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

| stage | script | needs | output | hardware | deterministic |
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

Every "yes" was verified by re-running the stage on the development machine and comparing the outputs **byte for byte** with the files that produced the submitted bundle (`reproduce/verify_dirs.py`):
20/20 masks, 20/20 track tables, all 100 solve outputs, 20/20 `fused_g`, 20/20 `vggt_g`, 8/8 `f73w`, 17/17 RAFT proxies, 17/17 ridge predictions, 112/112 CNN profiles (retrained with the same seeds),
and the final bundle. Path A was additionally run end to end from a clean copy of this repository: byte-identical, 0.0000 % deviation on all 20 clips.

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

## 4. Scalar constants fitted on the training split

`0.809` = median Theil–Sen slope of GPS course increments vs VGGT yaw increments over the 15 qualifying training wide clips (`3_src/mvo_tools/fit_yaw_gain.py`); `0.87` = typical focal-probe ratio of the 8 training flir-VIS clips
(leave-one-out validated on those clips).

## 5. Determinism and expected deviation

* Stages 20, 31, 50, 60 are deterministic on CPU (the tracker's RANSAC stream depends on the shard layout, which is therefore part of `20_tracker.sh`). Verified on macOS arm64 / Python 3.12 / NumPy 1.26.4 / OpenCV 4.10 / PyTorch 2.14.1.
  Other CPUs or library versions can change the last digits of the JSON; `compare_bundles.py` quantifies that.
* Stages 10/11/30/40 use GPU kernels (bf16 autocast in VGGT, atomics in DPVO): not byte-identical between runs or GPU models. `cache/gpu` contains the outputs that produced the submitted bundle.
* DPVO (clip_112 only) is documented non-deterministic by the organisers; if a re-run fails the sanity gate, clip_112 falls back to constant speed.
* **GPU re-run (path B) result: TBD** (filled in by the verification run).

## 6. Layout

```
3_src/mvo_tools/        our code (tracker, VGGT windows, RAFT band statistics, speed models, policy, helpers); 3_src/starter_upstream/ official LOOKOUT starter (37ca5c0, MIT)
reproduce/              stage scripts, reproduce_all.sh, compare_bundles.py, verify_dirs.py, train_speed_models.sh
weights/                our trained speed models + external_weights.yaml        artifacts/masks_v4/   the 20 static-structure masks
cache/                  verified intermediate outputs (stage-60 inputs, GPU-stage outputs)      expected/   the submitted bundle and its SHA-256
METHOD.md  repro.yaml  NOTICE.md  LICENSE  requirements.txt  requirements-gpu.txt
```
