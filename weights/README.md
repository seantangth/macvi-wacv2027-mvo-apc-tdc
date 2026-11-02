# weights/

Our own trained components (committed here, SHA-256 in `SHA256SUMS`):

| file | what | trained on |
|---|---|---|
| `ridge_train46.json` | ridge regression, relative water speed of wide clips | 46 training wide clips (train split only) |
| `cnn_gate/seed{0..4}.pt` | gated dilated 1-D CNN (93 k parameters each), 5 seeds averaged | 68 training clips (train split only) |

Public pretrained checkpoints (VGGT-1B, RAFT-large, DPVO) are *linked*, not copied: see `external_weights.yaml`.
