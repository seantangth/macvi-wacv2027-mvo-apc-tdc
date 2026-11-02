#!/usr/bin/env python3
"""Preprocessing for the DPVO baseline.

Everything the baseline does to a clip between "raw mp4 on disk" and "frames fed to the
tracker" lives here, so a participant can read one file and know exactly what was done.

The public inference wrapper uses calibration, active image rectangles and timestamps
from the released manifest. Preset-based detection below supports standalone inspection
when a released calibration is not supplied. Videos are distributed separately.

Steps, in order:

  1. Active-region detection  - thermal (`flir-IR`) clips are pillarboxed: the 640x512
                                sensor is scaled to 1080 rows and padded into a 1920-wide
                                container. Detected per clip, never hard-coded.
  2. Crop to active region    - removes the pad, and with it the burned-in FLIR logo and
                                gauge glyphs that sit on the pad.
  3. Intrinsics               - derived for the detected active width, principal point at
                                the centre of the active region, expressed in cropped
                                pixel coordinates.
  4. Temporal stride          - decimate to a common ~8 Hz so clips of different native
                                fps are compared on equal terms.
  5. Half-resolution downscale- INTER_AREA, intrinsics scaled by 0.5 (DPVO's own convention,
                                reproduced here so the crop composes correctly with it).
  6. Multiple-of-16 crop      - the tracker requires it; the principal point is shifted to
                                match rather than left at the pre-crop centre.

Deliberately NOT done: bow/sky masking, contrast normalisation, distortion correction.
Those are open problems for challenge participants, not part of the reference baseline.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, asdict
import cv2
import numpy as np

__all__ = ["ActiveRegion", "Intrinsics", "ClipPreprocessor",
           "detect_active_region", "intrinsics_for_preset", "PRESET_HFOV_DEG"]

# --------------------------------------------------------------------------------------
# Nominal camera geometry
# --------------------------------------------------------------------------------------
# Single source of truth: configs/camera_intrinsics.json. Nothing about camera geometry is
# duplicated in Python — edit the JSON, not this file.
#
# PROVENANCE: every value there is NOMINAL and FOV-DERIVED, not a measured calibration. The
# data contract requires that distinction be explicit, so it is carried through to the output
# of `intrinsics_for_preset` as `Intrinsics.provenance` and into each clip's manifest.
_LOCAL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'configs', 'camera_intrinsics.json')
CONFIG_PATH = os.environ.get(
    "MARINE_CAMERA_CONFIG",
    _LOCAL_CONFIG if os.path.isfile(_LOCAL_CONFIG) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "camera_intrinsics.json"),
)


def _load_config(path: str = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return json.load(fh)


_CFG = _load_config()
_PRESETS = _CFG["presets"]

# Nominal horizontal field of view per preset; released per-clip calibration takes precedence.
PRESET_HFOV_DEG = {k: v["hfov_deg"] for k, v in _PRESETS.items()}
# Vertical FOV, only where it is given AND differs from the horizontal geometry
# (thermal: 24 deg H / 18 deg V). fy is then derived from VFOV over the active ROWS,
# independent of the horizontal pad width.
PRESET_VFOV_DEG = {k: v["vfov_deg"] for k, v in _PRESETS.items() if v.get("vfov_deg")}

# Presets whose image is padded into a wider container and must be cropped before the FOV can
# be turned into a focal length.
PILLARBOXED_PRESETS = {k for k, v in _PRESETS.items() if v.get("pillarboxed")}

# Where fy is not simply equal to fx (non-square effective pixels after the camera's own
# scaling), the config states it as a number and it wins over the square-pixel assumption.
PRESET_FY_OVERRIDE = {
    k: v["fy"] for k, v in _PRESETS.items()
    if isinstance(v.get("fy"), (int, float)) and isinstance(v.get("fx"), (int, float))
    and v["fy"] != v["fx"]
}

DEFAULT_PRESET = "wide"


@dataclass
class ActiveRegion:
    """The rectangle of a frame that carries real imagery."""
    x0: int
    x1: int
    y0: int
    y1: int
    frame_w: int
    frame_h: int
    pillarboxed: bool
    pad_level: float | None = None      # mean grey value of the pad, when there is one
    method: str = "temporal-variance"

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0:self.y1 + 1, self.x0:self.x1 + 1]


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    preset: str
    provenance: str                     # "nominal-fov-derived" | "measured" | "estimated"
    hfov_deg: float | None = None

    def scaled(self, s: float) -> "Intrinsics":
        return Intrinsics(self.fx * s, self.fy * s, self.cx * s, self.cy * s,
                          int(round(self.width * s)), int(round(self.height * s)),
                          self.preset, self.provenance, self.hfov_deg)

    def as_calib_row(self) -> str:
        """DPVO calibration file format: fx fy cx cy [k1 k2 p1 p2]."""
        return f"{self.fx} {self.fy} {self.cx} {self.cy}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# 1. Active-region detection
# --------------------------------------------------------------------------------------
def detect_active_region(video_path: str, n_samples: int = 24,
                         row_band: tuple[float, float] = (0.30, 0.70),
                         t_std_thresh: float = 0.75) -> ActiveRegion:
    """Find the columns of a clip that carry live imagery.

    A padding bar is constant over time. Real imagery is not. So the discriminator is the
    *temporal* standard deviation of each column, not its brightness.

    Brightness thresholding does not work here and was the source of a real error: on these
    clips the pad is rendered at grey level 15-20, not 0, so a `pixel > 8` test classifies
    the pad as image and reports no pillarboxing at all.

    Only the middle rows are examined (`row_band`). The FLIR logo is burned into the top of
    the frame and a gauge glyph into the bottom left, both of them sitting *on* the pad. A
    full-height test sees those as live columns and pushes the detected edges outward into
    the pad - measured 73..1834 instead of the true 256..1632.
    """
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 10:
        cap.release()
        raise ValueError(f"clip too short to analyse: {video_path}")
    frames = []
    for idx in np.linspace(n * 0.05, n * 0.95, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, im = cap.read()
        if ok:
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 5:
        raise ValueError(f"could not sample enough frames: {video_path}")

    stack = np.stack(frames)                       # (T, H, W)
    _, H, W = stack.shape
    r0, r1 = int(H * row_band[0]), int(H * row_band[1])
    col_t_std = stack[:, r0:r1, :].std(axis=0).mean(axis=0)
    live = col_t_std > t_std_thresh

    # Longest contiguous run of live columns. A run, not min/max of all live columns, so a
    # stray bright glyph outside the image cannot widen the result.
    best_len, best = 0, (0, W - 1)
    i = 0
    while i < W:
        if live[i]:
            j = i
            while j + 1 < W and live[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best = j - i + 1, (i, j)
            i = j + 1
        else:
            i += 1
    x0, x1 = best
    pillarboxed = (x0 > 0) or (x1 < W - 1)
    pad_level = None
    if x0 > 0:
        pad_level = round(float(stack[:, r0:r1, :x0].mean()), 2)
    return ActiveRegion(x0=int(x0), x1=int(x1), y0=0, y1=H - 1,
                        frame_w=W, frame_h=H, pillarboxed=bool(pillarboxed),
                        pad_level=pad_level)


# --------------------------------------------------------------------------------------
# 3. Intrinsics
# --------------------------------------------------------------------------------------
def _focal_from_hfov(width_px: float, hfov_deg: float) -> float:
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def intrinsics_for_preset(preset: str, region: ActiveRegion) -> Intrinsics:
    """Focal length and principal point for the *cropped* frame.

    The FOV in `camera_profiles` describes the optics, so it spans the real image, not the
    padded container. Applying it to the container width is the mistake that produced
    fx=4516 for `flir-IR`; over the measured active width of ~1377 px the same 24 deg gives
    fx ~= 3239.

    A horizontal crop does not change fx or fy - it only moves the principal point. cx is
    therefore the centre of the active region expressed in cropped coordinates, which for a
    symmetric sensor is simply width/2.
    """
    hfov = PRESET_HFOV_DEG.get(preset)
    if hfov is None:
        hfov = PRESET_HFOV_DEG["wide"]
        preset_used = "wide"
    else:
        preset_used = preset
    fx = _focal_from_hfov(region.width, hfov)
    vfov = PRESET_VFOV_DEG.get(preset_used)
    if vfov is not None:
        fy = _focal_from_hfov(region.height, vfov)   # from VFOV over active rows; pad width irrelevant
    else:
        fy = PRESET_FY_OVERRIDE.get(preset_used, fx)
    return Intrinsics(fx=fx, fy=fy,
                      cx=region.width / 2.0, cy=region.height / 2.0,
                      width=region.width, height=region.height,
                      preset=preset_used, provenance="nominal-fov-derived", hfov_deg=hfov)


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------
class ClipPreprocessor:
    """Owns every transform applied to one clip, and the intrinsics that go with it.

    Construct once per clip, then either iterate `frames()` or hand `stride`, `calib_path`
    and `crop_rect` to a stream that does its own decoding.
    """

    def __init__(self, video_path: str, preset: str, target_fps: float = 8.0,
                 half_res: bool = True, multiple_of: int = 16,
                 region: ActiveRegion | None = None, calibration: dict | None = None,
                 frame_timestamps=None):
        if not np.isfinite(target_fps) or target_fps <= 0:
            raise ValueError('target_fps must be positive and finite')
        self.video_path = video_path
        self.preset = preset
        self.target_fps = target_fps
        self.half_res = half_res
        self.multiple_of = multiple_of

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('cannot open video')
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_fps = cap.get(cv2.CAP_PROP_FPS)
        self._pts = []      # actual PTS (s) of each yielded frame, filled by frames()
        self._decoded_index = []   # decoded frame index of each yielded frame
        # Same information keyed by tracker index. frames() may run in a DIFFERENT process
        # from the one that later scores the trajectory, so these maps are what a parent
        # process installs via adopt_frame_times(); they take precedence over the lists.
        self._pts_by_index: dict = {}
        self._decoded_by_index: dict = {}
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.frame_timestamps = frame_timestamps
        if frame_timestamps is not None:
            # Container frame counts can differ from the actual decoded stream.
            # Check the complete decoded stream in frames_with_time instead.
            self.n_frames = len(frame_timestamps)
        if calibration is not None:
            from public_dataset import validate_calibration
            validate_calibration(calibration)
            if [frame_w, frame_h] != [calibration['container_width'], calibration['container_height']]:
                raise ValueError('video dimensions differ from the released calibration')
            x, y, w, h = calibration['active_rect']
            region = ActiveRegion(x, x+w-1, y, y+h-1, frame_w, frame_h,
                                  w != frame_w or h != frame_h, method='released-calibration')

        # 1. active region - detect only for presets that can be padded, so we do not pay
        #    24 seeks on every wide clip.
        if region is not None:
            self.region = region
        elif preset in PILLARBOXED_PRESETS:
            self.region = detect_active_region(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            self.region = ActiveRegion(0, w - 1, 0, h - 1, w, h, False, method="assumed-full-frame")

        # 3. intrinsics for the cropped frame
        self.intrinsics_full = (Intrinsics(
            calibration['fx'], calibration['fy'], calibration['cx'], calibration['cy'],
            calibration['width'], calibration['height'], preset, calibration['provenance'])
            if calibration is not None else intrinsics_for_preset(preset, self.region))

        # 4. temporal stride
        self.stride = max(1, int(round((self.native_fps or target_fps) / target_fps)))

        # 5/6. resolution chain: half-res then trim to a multiple of 16
        intr = self.intrinsics_full.scaled(0.5) if half_res else self.intrinsics_full
        w = intr.width - intr.width % multiple_of
        h = intr.height - intr.height % multiple_of
        # the trim removes rows/cols from the bottom/right only, so the principal point keeps
        # its absolute position and must NOT be recentred on the trimmed frame
        self.intrinsics = Intrinsics(intr.fx, intr.fy, intr.cx, intr.cy, w, h,
                                     intr.preset, intr.provenance, intr.hfov_deg)

    # -- accessors ---------------------------------------------------------------------
    @property
    def effective_fps(self) -> float:
        return (self.native_fps or self.target_fps) / self.stride

    @property
    def has_real_frame_times(self) -> bool:
        """True when frame_time() reports measured PTS rather than the average-fps estimate.

        Check this in the parent process before scoring. Decoding happens in a separate
        process, so an instance that was never iterated and never had timestamps installed
        will answer frame_time() from the fps estimate without complaining.
        """
        return bool(self._pts_by_index) or bool(self._pts)

    def adopt_frame_times(self, pts_by_index: dict, decoded_by_index: dict | None = None) -> None:
        """Install presentation timestamps captured while iterating this clip elsewhere.

        frames() records PTS onto the instance doing the decoding. When decoding runs in a
        child process, those values never reach the parent's instance, and frame_time()
        silently answers from the average-fps estimate instead - which is exactly the defect
        this method exists to close. The parent carries (index -> pts) across the queue and
        installs it here before scoring.
        """
        if not pts_by_index:
            return
        self._pts_by_index = {int(k): float(v) for k, v in pts_by_index.items()}
        if decoded_by_index:
            self._decoded_by_index = {int(k): int(v) for k, v in decoded_by_index.items()}

    def frame_time(self, tracker_index: int) -> float:
        """Actual presentation timestamp (s) of the n-th frame the tracker saw.

        Uses the decoded frame's real PTS, NOT an average-fps reconstruction - variable-rate
        video and the stride offset otherwise mislabel frames (a frame at PTS 2.2 s was being
        called 3.233 s). Prefers timestamps installed by adopt_frame_times (the cross-process
        path), then those recorded by a local frames() iteration, and only falls back to the
        fps estimate when this instance has neither. Use has_real_frame_times to tell which.
        """
        idx = int(tracker_index)
        if idx in self._pts_by_index:
            return float(self._pts_by_index[idx])
        if self._pts and 0 <= idx < len(self._pts):
            return float(self._pts[idx])
        return idx * self.stride / (self.native_fps or self.target_fps)

    def decoded_index(self, tracker_index: int):
        """Index of the decoded video frame behind the n-th frame the tracker saw, or None."""
        idx = int(tracker_index)
        if idx in self._decoded_by_index:
            return self._decoded_by_index[idx]
        if self._decoded_index and 0 <= idx < len(self._decoded_index):
            return self._decoded_index[idx]
        return None

    def write_calib(self, path: str, with_distortion: bool = False) -> str:
        row = self.intrinsics.as_calib_row()
        if with_distortion:
            row += " 0 0 0 0"
        with open(path, "w") as fh:
            fh.write(row)
        return path

    def _load_mask(self):
        """MVO addition: optional static-structure mask (bow, radar dome, logos). Enabled by env MVO_MASK_DIR pointing to
        a directory of <clip_key>.png (255 = masked) in encoded-video pixels. Masked pixels are painted with the frame
        mean so that gradient-biased patch selection avoids them. Loaded lazily per process (spawn-safe)."""
        if getattr(self, '_mask_loaded', False):
            return self._mask
        self._mask_loaded = True; self._mask = None
        mdir = os.environ.get('MVO_MASK_DIR')
        if mdir:
            key = os.path.splitext(os.path.basename(self.video_path))[0]
            mp = os.path.join(mdir, key + '.png')
            if os.path.isfile(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape[0] == self.region.frame_h and m.shape[1] == self.region.frame_w:
                    m = self.region.crop(m)
                    if self.half_res:
                        m = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                    h, w = m.shape[:2]
                    self._mask = m[:h - h % self.multiple_of, :w - w % self.multiple_of] > 127
        return self._mask

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply steps 2, 5 and 6 to a single decoded frame (plus the optional MVO static mask)."""
        img = self.region.crop(image)
        if self.half_res:
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        img = img[:h - h % self.multiple_of, :w - w % self.multiple_of]
        # MVO addition (E013): optional CLAHE on the L channel, enabled by env MVO_CLAHE=<clipLimit> (e.g. 2.0);
        # MVO_CLAHE_PRESETS=<comma list> restricts it to those camera presets (default: all). Read per process (spawn-safe).
        cl = os.environ.get('MVO_CLAHE')
        if cl:
            presets = os.environ.get('MVO_CLAHE_PRESETS', '')
            if not presets or getattr(self, 'preset', None) in presets.split(','):
                if not hasattr(self, '_clahe'):
                    self._clahe = cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(8, 8))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = self._clahe.apply(lab[:, :, 0]); img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        m = self._load_mask()
        if m is not None and m.shape == img.shape[:2] and m.any():
            img = img.copy(); img[m] = img.reshape(-1, img.shape[-1]).mean(axis=0).astype(img.dtype)
        return img

    def frames(self):
        """Yield `(tracker_index, preprocessed_frame, intrinsics_array)`.

        Thin wrapper over frames_with_time for callers that do not carry timestamps. If you
        are decoding in one process and scoring in another, use frames_with_time and pass the
        PTS across - see adopt_frame_times.
        """
        for index, image, intr, _pts, _decoded in self.frames_with_time():
            yield index, image, intr

    def frames_with_time(self):
        """Yield `(tracker_index, frame, intrinsics_array, pts_seconds, decoded_index)`.

        The PTS and decoded index are yielded, not merely recorded on self, so a decoder
        running in a child process can hand them to the parent that will do the scoring.
        """
        cap = cv2.VideoCapture(self.video_path)
        t, ok = 0, True
        self._pts, self._decoded_index = [], []
        intr = np.array([self.intrinsics.fx, self.intrinsics.fy,
                         self.intrinsics.cx, self.intrinsics.cy], dtype=np.float32)
        decoded_count = 0
        try:
            while True:
                for _ in range(self.stride):
                    ok, image = cap.read()
                    if not ok:
                        break
                    decoded_count += 1
                    if self.frame_timestamps is not None and decoded_count > self.n_frames:
                        raise ValueError('decoded video has more frames than released timestamps')
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                idx = decoded_count - 1
                if self.frame_timestamps is not None:
                    pts = float(self.frame_timestamps[idx])
                elif not np.isfinite(pts) or (self._pts and pts <= self._pts[-1]):
                    raise ValueError('decoder did not provide increasing presentation timestamps')
                self._pts.append(pts)
                self._decoded_index.append(idx)
                self._pts_by_index[t] = float(pts)
                self._decoded_by_index[t] = int(idx)
                yield t, self.transform(image), intr, float(pts), int(idx)
                t += 1
            if self.frame_timestamps is not None and decoded_count != self.n_frames:
                raise ValueError('decoded video has fewer frames than released timestamps')
        finally:
            cap.release()

    def manifest(self) -> dict:
        """Everything the data contract asks us to record for this clip."""
        return {
            "video": os.path.basename(self.video_path),
            "camera_mode": self.preset,
            "image_size": [self.region.frame_w, self.region.frame_h],
            "active_image_rect": {"x0": self.region.x0, "y0": self.region.y0,
                                  "x1": self.region.x1, "y1": self.region.y1,
                                  "width": self.region.width, "height": self.region.height},
            "pillarboxed": self.region.pillarboxed,
            "pad_level": self.region.pad_level,
            "active_region_method": self.region.method,
            "intrinsics_active": self.intrinsics_full.to_dict(),
            "intrinsics_as_fed_to_tracker": self.intrinsics.to_dict(),
            "distortion_model": "none",
            "calibration_provenance": self.intrinsics.provenance,
            "transforms": [
                {"op": "crop", "rect": [self.region.x0, self.region.y0,
                                        self.region.x1, self.region.y1]},
                {"op": "temporal_stride", "stride": self.stride,
                 "native_fps": self.native_fps, "effective_fps": round(self.effective_fps, 3)},
                {"op": "resize", "factor": 0.5 if self.half_res else 1.0,
                 "interpolation": "INTER_AREA"},
                {"op": "trim_to_multiple", "multiple": self.multiple_of,
                 "final_size": [self.intrinsics.width, self.intrinsics.height]},
            ],
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the preprocessing applied to a clip.")
    ap.add_argument("video")
    ap.add_argument("--preset", default="wide")
    ap.add_argument("--target-fps", type=float, default=8.0)
    a = ap.parse_args()
    pp = ClipPreprocessor(a.video, a.preset, target_fps=a.target_fps)
    print(json.dumps(pp.manifest(), indent=2))
