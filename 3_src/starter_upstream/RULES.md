# Maritime Visual Odometry — v0.1 development rules

## Task and allowed inputs

Estimate a camera trajectory from **monocular video, camera calibration, and video frame
timestamps**. GPS, IMU, AIS, speed, heading, attitude and other clip-specific navigation
signals are not inference inputs. The train/validation reference files are evaluation
labels, separate from the inputs. Test clips and references are withheld from this
development release.

Train on the provided training split. External pretraining is allowed and must be
disclosed. Use validation references for development/evaluation; do not train on the
validation or test splits, including self-supervised training on their videos. These
rules apply to both development and held-out evaluation. The workshop website is
the source of truth for dates and submission availability.

## Data and predictions

The public manifest uses anonymous `clip_NNN` IDs. It contains video/timing/reference
paths and full calibration, without source device identifiers or storage paths. The
coordinate and prediction formats are specified in `QUICKSTART.md`.

Reference trajectories contain only `key`, `times`, and local three-dimensional
`positions` in metres. East and north come from the cleaned navigation reference;
up is zero because this is a horizontal position reference. The first reference position
is zero and no geographic origin is distributed. Reference times use the same seconds
clock as the released video PTS, including the organizer's reviewed clock offset.

Predictions need not share reference axes, origin or scale. Submit a self-consistent
timestamped trajectory and its decoded source-frame indices. Orientation is not scored.

## Metric

For each clip, interpolate the prediction onto reference times in their overlapping
window. Fit one global similarity transform (rotation, translation and scale). For
each reference path length in `{100,200,...,800}` metres and each start index sampled
at stride 10 on that reference window, find the first endpoint reaching that length.
The segment must span at least three reference-index intervals. Compute

`100 * ||(E[j]-E[i]) - (G[j]-G[i])|| / segment_length`.

Drift is the mean over eligible segments. There is no per-segment alignment. This is
a **position-only, globally scale-aligned relative-displacement metric** inspired by
KITTI; it is not the full KITTI pose metric and does not measure recovered metric scale.
ATE is also reported. `starter/metrics.py` is the implementation.

## Coverage and failure reporting

Choose the expected split or explicit clip list independently of prediction files.
The evaluator emits one record for every expected clip:

| Status | Meaning |
|---|---|
| `scored` | Valid trajectory with an eligible drift metric |
| `missing_prediction` | No prediction file for an expected clip |
| `invalid_prediction` | Invalid JSON, dimensions, indices or timestamps |
| `non_finite_estimate` | Non-finite position values |
| `no_overlap` | Fewer than ten reference samples in the shared time window |
| `degenerate_estimate` | Global alignment cannot be computed |
| `no_eligible_segment` | No eligible 100–800 m segment |
| `missing_gt` | A manifest declares withheld test references |
| `invalid_dataset` | Required dataset timing/reference file is missing or invalid |

Inference records tracking/decode failures separately and does not emit a prediction for
them. The evaluator consequently records those expected clips as missing predictions.
Unexpected prediction IDs are listed in the summary and excluded from scoring.

The summary reports expected/scored/unscored counts and mean/median drift **over scored
clips only**. Per-clip temporal coverage and maximum prediction-time gap are recorded.
Shortened trajectories are evaluated over their shared window, so coverage must accompany
scores. Development scores are not leaderboard entries. A low drift over a small
selected subset must not be presented as full-dataset performance.

## Held-out competition submissions

Submit predictions, not a model. Run your method on the released test videos and
use `starter/pack_submission.py` to combine the 20 clip predictions into one JSON
bundle (64 MiB maximum). The participant test package contains no test reference
trajectories. The organizer server evaluates against private references.

An entry is eligible only if **all 20 clips score** and **each clip has at least
95% temporal coverage**. For this gate, a reference timestamp is covered by a
predicted sample at that time, or by interpolation between predicted samples
no more than **1 second** apart. Larger gaps are not supported coverage. A small
number of endpoints cannot qualify as full coverage. Scoring still uses the
full overlapping window; gaps do not remove difficult segments from the score.
The server report lists per-clip status and coverage without reference positions.
This gap-aware gate is separate from the local development evaluator's overlap
coverage diagnostic; only the server report determines competition eligibility.

Create or join a Maritime Visual Odometry team on the MaCVi Teams & paper page before submitting.
The team owner uploads for the team; author metadata and OpenReview details may
be completed later, by the published metadata deadline. The limit is one evaluated
submission per team and uploader per UTC day (reset at 00:00 UTC). Pending jobs
reserve the slot. Format failures release it; scored but incomplete/ineligible
entries consume it. Deleting a submission does not restore the allowance.

Select one eligible submission as the team's **final entry** in the dashboard.
Only this selected entry is publicly ranked, by ascending mean drift across all
20 clips. You can change the selection until the submission deadline. Team and
uploader attribution are recorded at upload time and are not rewritten by later
membership or name changes. Keep your code, model settings and inference logs
for organizer verification; they are not part of the initial JSON upload.

## Release scope and terms

The development release supports local evaluation. Check the
[challenge page](https://macvi.org/workshop/macvi27/challenges/maritime-visual-odometry) for the
held-out download, dates and whether online submissions are open. Consult the supplied
`LICENSE.txt` for the approved release terms and retain upstream notices for dependencies.
