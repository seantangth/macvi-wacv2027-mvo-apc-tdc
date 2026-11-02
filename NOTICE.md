# Third-party material and attribution

* **Data.** The released LOOKOUT Visual Odometry challenge data (MaCVi @ WACV 2027; CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/) are *not* redistributed here.
  The files in `cache/` and `artifacts/` are numerical outputs derived from the test clips (VGGT window poses, RAFT band statistics, DPVO trajectory of clip_112, static-structure masks, trajectories);
  they are shared under the same terms: attribution to the LOOKOUT Visual Odometry Challenge (MaCVi @ WACV 2027), non-commercial use, changes indicated (derived/processed).
* **Starter pack** (`3_src/starter_upstream/`): LOOKOUT-AI/lookout-vo-challenge @ 37ca5c0, MIT (see `3_src/starter_upstream/LICENSE`).
* **VGGT-1B** (Wang et al., CVPR 2025; weights CC BY-NC 4.0; not redistributed, downloaded from Hugging Face), **RAFT-large** (Teed & Deng, ECCV 2020; torchvision, BSD-3),
  **DPVO** (Teed, Lipson, Deng, NeurIPS 2023; MIT; `princeton-vl/DPVO` @ 859bbbf, TartanAir-trained checkpoint; not redistributed).
* Our code is MIT (see `LICENSE`); our trained speed-model weights in `weights/` are released under the same MIT licence.
