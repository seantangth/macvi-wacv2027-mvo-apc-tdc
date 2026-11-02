# Method (team APC_TDC, MaCVi @ WACV 2027 Maritime Visual Odometry, final entry #19363)

**Approach.** For every clip we estimate a planar *heading* ψ(t) and a *relative speed* s(t) from the monocular video and integrate them at the released frame timestamps,
x(t+Δt) = x(t) + s·Δt·[cos ψ, sin ψ]. The leaderboard metric aligns each clip with one Sim(3) transform, so absolute scale is irrelevant and we never estimate it.
* **Heading**: classical horizon-band KLT tracker (window-wise rotation → yaw) and windowed VGGT-1B (rotation only; VGGT's translation/scale is discarded), fused per camera group
  (flir-VIS: 0.7 VGGT + 0.3 tracker × per-clip gain from a VGGT focal-length probe; wide: 0.7 tracker + 0.3 VGGT × 0.809; flir-IR: tracker).
* **Speed**: RAFT-large optical flow of the water surface below the horizon → rule proxy (5 harbour flir-VIS clips), or a ridge regressor + a small gated 1-D CNN (7 wide clips);
  DPVO speed profile for clip_112 only; constant speed elsewhere. Profiles enter as `max((p/median p)^k, 0.15)` with k = 1.0 (harbour flir-VIS) and 0.7 (wide), 45 s smoothing.

**Training data.** Only the *training split* of the released development package (GPS reference used as label): ridge model (46 training wide clips), gated CNN (68 training clips),
the VGGT rotation scale 0.809 (training wide clips) and the flir-VIS gain constant 0.87 (8 training flir-VIS clips). The validation split and 17 further development clips were used for
validation and for choosing k / floor / smoothing. No validation or test video was used for training, fine-tuning or self-supervised adaptation.

**All external data / pretrained models.** No external dataset. Pretrained checkpoints used at inference:
VGGT-1B (`facebook/VGGT-1B`, CC BY-NC 4.0), RAFT-large (torchvision `C_T_SKHT_V2`, BSD-3), DPVO (`dpvo.pth`, TartanAir, MIT; clip_112 only). See `weights/external_weights.yaml`.

**Anything non-standard at inference** (fixed lists, identical in every run; they were set after inspecting the test clips, see the factsheet):
* heading fallback to windowed VGGT alone for clip_096, clip_104, clip_111 (flir-VIS) and clip_103 (wide);
* yaw gain of clip_098 set by hand to 0.78 (mid-clip optical zoom; the probe rule would give 0.6); the other ten flir-VIS gains follow the probe rule;
* five harbour flir-VIS clips (093, 094, 099, 107, 111) take the water-flow speed;
* clip_112 uses its DPVO speed profile only if the DPVO trajectory passes a sanity gate, otherwise constant speed.

**Leaderboard-derived choices.** No numeric setting was tuned on leaderboard scores. Scores decided which components were kept (water-flow speed on wide clips, ensemble speed, harbour k = 1)
and which of our uploads is the final entry (#19363 rather than #19434). Details in the factsheet, Section 2.

**Determinism.** The CPU stages (tracker, speed models, fusion, policy, packing) are deterministic and were verified byte-identical against the files that produced the submitted bundle.
The GPU stages (VGGT windows, focal probe, RAFT flow, DPVO) are not byte-reproducible across runs/GPUs; `cache/gpu` holds the outputs that produced the submitted bundle.
Training code: `reproduce/train_speed_models.sh`.
