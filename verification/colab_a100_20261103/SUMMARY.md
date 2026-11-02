# Verification run on a Colab A100 (3 November 2026, Taipei time)

Purpose: re-run the repository as an organiser would on a machine that is not the development machine, and measure how close the reproduction is to the submitted bundle (#19363).

**Machine** (`machine.txt`): Google Colab VM, Ubuntu 24.04.4 LTS, NVIDIA A100-SXM4-40GB (driver 580.82.07), Intel Xeon @ 2.20 GHz, 12 vCPU, 83 GB RAM.
**Software**: environments built by `reproduce/gpu_box.sh` (Python 3.10.22 via uv, torch 2.5.1+cu121, torchvision 0.20.1, VGGT commit `a288dd0`, `pip_freeze_venvV.txt`); checkpoint hashes in `external_weights_sha256.txt`.
**Code**: commit `3212000` for the environment, VGGT, focal probe, RAFT and the CPU stages; the DPVO stage was repaired during the run and re-verified with commit `7c5a82f`.

## Timeline of the run (Taipei time, 03.11.2026)

| time | stage |
|---|---|
| 00:43:52 | start (`reproduce_all.sh B`), test package download, environment |
| 00:46:42 – 02:07:57 | VGGT windows, 20 clips (81 min) |
| 02:07:57 – 02:08:37 | focal probe, 11 flir-VIS clips (40 s) |
| 02:08:37 – 02:32:00 | RAFT-large band statistics, 12 clips (23 min) |
| 02:32:00 | DPVO clip_112: `tracking_failed` (the DPVO build had failed, see below) → cached DPVO output used |
| 02:32 – 02:54 | CPU stages: tracker, speed models, headings, policy, pack, compare (22 min) |
| 02:54:25 | path B finished: `rc=0 seconds=7830`; path A (cached GPU outputs, CPU stages on this Linux VM) started |
| 03:17:13 | path A finished: `rc=0 seconds=1368` |

## Results (deviation = official drift metric of the reproduction scored against the submitted trajectory, mean over the 20 clips; per-clip values in `job_pathB_then_pathA.log`)

| run | mean | max |
|---|---|---|
| A0 on Linux (`pathA0_linux.log`, 16 s) | 0.0000 % (largest coordinate difference 8.5e-14 m) | 0.0000 % |
| A on Linux (CPU stages recomputed, cached GPU outputs) | 0.8556 % | 3.4433 % (clip_093) |
| B (VGGT windows, probe, RAFT re-run; cached DPVO output) | 1.0028 % | 3.4433 % (clip_093) |
| B + DPVO re-run (`pathB_with_dpvo_rerun_compare.log`) | 1.0940 % | 3.4433 % (clip_093); clip_112 1.8375 % |

## Stage-by-stage comparison with the cached outputs (`compare_intermediates.txt`)

* VGGT windows: 20/20 clips identical (mean |Δ yaw increment| 0.0000 °/s, max |Δ cumulative yaw| 0.000°).
* RAFT band statistics: 12/12 clips identical (mean |Δ| 0.0000 px/s, correlation 1.00000).
* Focal probe: 6 of 11 clips identical; ratios differ on clips 104 (1.0572→1.0204), 107 (1.1919→1.2034), 108 (0.8401→0.8002), 109 (1.0027→0.8900), 111 (0.9769→0.9854); max |Δ| 0.1126, mean 0.0190.
* DPVO clip_112: re-run vs cached trajectory 1.555 % (official drift after scaling the path to 2 km; the first version of `compare_intermediates.py` printed `nan` for DPVO because its units are arbitrary, fixed in `c86a5a6`).
* Tracker (CPU): heading `rotw` of the Linux run vs the macOS originals 0.81 % mean, 4.02 % max; the track tables differ in the horizon detections (`hz_*`) and therefore in the tracked features (`xy`, `tid`) while frame timestamps, intrinsics, grids are identical (`tracker_opencv410_experiment.log`: a clean venv with `opencv-python-headless==4.10.0.84`, the version of the development machine, gives the same differences as the OpenCV 4.11.0 build that VGGT's dependencies install into the GPU environment).
* Speed models recomputed on Linux: ≤ 0.03 % deviation.

## Problems found during the run and what was changed

1. **DPVO build failed** (`dpvo_env_first_attempt_FAILED.log`): Colab ships CUDA toolkit 13.0, the PyTorch wheels are built for CUDA 12.1, PyTorch refuses to build extensions across major versions. `gpu_box.sh` now looks for an `nvcc` 12.x (`/usr/local/cuda-12*`), else installs the CUDA 12.6 toolkit pieces from the NVIDIA apt repository, and keeps the build log (`dpvo_build.log`).
2. **DPVO import failed** (`dpvo_attempt1_matplotlib_backend_error.log`): Colab/Jupyter shells export `MPLBACKEND=module://matplotlib_inline.backend_inline`, which is invalid outside a notebook; `gpu_box.sh` now sets `MPLBACKEND=Agg`.
3. With both fixes, DPVO built (18 s incremental build) and predicted clip_112 in 179 s (`dpvo_attempt2_final_gpu_box.log`).
4. `compare_intermediates.py` DPVO comparison fixed (see above); `.gitignore` now ignores `1_data/`.
