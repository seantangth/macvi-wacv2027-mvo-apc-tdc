"""DPVO tracking core shared by private experiments and the public starter."""
from __future__ import annotations
import os
import platform
import subprocess
import sys
import multiprocessing as mp
from queue import Empty
import numpy as np
from preprocessing import ClipPreprocessor
from public_dataset import sha256

def _resolve_dpvo_dir():
    if os.environ.get('DPVO_DIR'):
        return os.environ['DPVO_DIR']
    if os.environ.get('MACVI_ROOT'):
        return os.path.join(os.environ['MACVI_ROOT'], 'DPVO')
    return None

DPVO_DIR = _resolve_dpvo_dir()

def _stream_worker(queue, video_path: str, preset: str, target_fps: float,
                   calibration=None, frame_timestamps=None):
    """Decode + preprocess in a separate process so the GPU is not waiting on ffmpeg.

    The queue carries each frame's real PTS and decoded index alongside the image. It has to:
    this function builds its OWN ClipPreprocessor, so any timestamp it recorded on that
    object would die with the process and leave the parent scoring against an average-fps
    clock.
    """
    try:
        pre = ClipPreprocessor(video_path, preset, target_fps=target_fps,
                               calibration=calibration, frame_timestamps=frame_timestamps)
        for index, image, intrinsics, pts, decoded in pre.frames_with_time():
            queue.put((index, image, intrinsics, pts, decoded))
        queue.put((-1, None, None, -1.0, -1))
    except Exception as exc:
        queue.put((-2, str(exc), None, -1.0, -1))


class ClipStream:
    """Decodes a clip in a child process and RETAINS the timestamps that child captured.

    Iterating yields `(tracker_index, image, intrinsics)` exactly as the tracker wants, while
    `frame_times` / `decoded_indices` accumulate the values carried across the queue. Install
    them on the scoring-side preprocessor with adopt_frame_times once tracking is done.

    Uses 'spawn', not the default 'fork': a forked child inherits the parent's CUDA context
    and holds ~200 MB of GPU memory for its whole life. Leaking one of those per clip is what
    exhausted a 24 GB card partway through an earlier run and surfaced as a bogus
    "CUDA out of memory". The decoder needs no GPU at all, so it starts clean.
    """

    def __init__(self, video_path: str, preset: str, target_fps: float,
                 calibration=None, frame_timestamps=None):
        ctx = mp.get_context('spawn')
        self._queue = ctx.Queue(maxsize=8)
        self._worker = ctx.Process(target=_stream_worker,
                                   args=(self._queue, video_path, preset, target_fps,
                                         calibration, frame_timestamps),
                                   daemon=True)
        self.frame_times: dict = {}
        self.decoded_indices: dict = {}
        self._worker.start()

    def __iter__(self):
        while True:
            try:
                index, image, intrinsics, pts, decoded = self._queue.get(timeout=5)
            except Empty:
                if not self._worker.is_alive():
                    raise RuntimeError('video decoder exited without completing the stream')
                continue
            if index == -2:
                raise RuntimeError(f'video decoder failed: {image}')
            if index < 0:
                break
            self.frame_times[int(index)] = float(pts)
            self.decoded_indices[int(index)] = int(decoded)
            yield index, image, intrinsics

    def close(self):
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=5)
        self._queue.close()


def _dpvo_config(dpvo_buffer_size=None):
    """DPVO configuration exactly as track() runs it: upstream defaults, default.yaml, override."""
    if not DPVO_DIR:
        raise ValueError('Set DPVO_DIR to the pinned DPVO checkout; see QUICKSTART.md')
    if dpvo_buffer_size is not None and dpvo_buffer_size <= 0:
        raise ValueError('dpvo_buffer_size must be positive')
    sys.path.insert(0, DPVO_DIR)
    from dpvo.config import cfg
    # Keep a diagnostic override local to this run instead of mutating DPVO's singleton.
    cfg = cfg.clone()
    cfg.merge_from_file(os.path.join(DPVO_DIR, 'config', 'default.yaml'))
    if dpvo_buffer_size is not None:
        cfg.BUFFER_SIZE = dpvo_buffer_size
    return cfg


def _git_revision(path):
    """Commit of the checkout rooted at path; 'unknown' otherwise (not a parent repo's HEAD)."""
    try:
        top, revision = subprocess.run(
            ['git', '-C', path, 'rev-parse', '--show-toplevel', 'HEAD'], capture_output=True,
            text=True, timeout=10, check=True).stdout.split()
    except (OSError, ValueError, subprocess.SubprocessError):
        return 'unknown'
    return revision if os.path.realpath(top) == os.path.realpath(path) else 'unknown'


def _module_version(name):
    try:
        return str(__import__(name).__version__)
    except Exception:
        return None


def describe_environment(dpvo_buffer_size=None):
    """Record the DPVO revision, checkpoint, resolved settings and library versions of a run.

    Every probe is best effort: a missing DPVO checkout or library is recorded as unknown/null
    instead of stopping the run, because the per-clip records already capture that failure.
    """
    dpvo = dict(dir=DPVO_DIR, revision='unknown', checkpoint_sha256=None,
                buffer_size=dpvo_buffer_size, config=None)
    if DPVO_DIR:
        dpvo['revision'] = _git_revision(DPVO_DIR)
        try:
            dpvo['checkpoint_sha256'] = sha256(os.path.join(DPVO_DIR, 'dpvo.pth'))
        except OSError:
            pass
        try:
            cfg = _dpvo_config(dpvo_buffer_size)
            dpvo['buffer_size'] = int(cfg.BUFFER_SIZE)
            dpvo['config'] = {key: cfg[key] for key in sorted(cfg)}
        except Exception:
            pass
    env = dict(python=platform.python_version(), numpy=np.__version__,
               opencv=_module_version('cv2'), torch=None, torch_cuda=None, gpu=None)
    try:
        import torch
        env.update(torch=str(torch.__version__), torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            env['gpu'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return dpvo, env


def track(video_path: str, preset: str, target_fps: float, seed: int,
          calibration=None, frame_timestamps=None, dpvo_buffer_size=None):
    """Run DPVO over one clip. Returns (poses, frame_indices, preprocessor)."""
    cfg = _dpvo_config(dpvo_buffer_size)
    import torch
    from dpvo.dpvo import DPVO

    torch.manual_seed(seed)
    np.random.seed(seed)

    pre = ClipPreprocessor(video_path, preset, target_fps=target_fps,
                           calibration=calibration, frame_timestamps=frame_timestamps)
    stream = ClipStream(video_path, preset, target_fps, calibration, frame_timestamps)

    slam = None
    try:
        with torch.no_grad():
            for index, image, intrinsics in stream:
                image_t = torch.from_numpy(image).permute(2, 0, 1).cuda()
                intr_t = torch.from_numpy(np.asarray(intrinsics, np.float32)).cuda()
                if slam is None:
                    _, height, width = image_t.shape
                    slam = DPVO(cfg, os.path.join(DPVO_DIR, 'dpvo.pth'),
                                ht=height, wd=width, viz=False)
                slam(index, image_t, intr_t)
        if slam is None:
            raise RuntimeError('no frames decoded')
        poses, indices = slam.terminate()
        # Move the child's timestamps onto the instance the caller will score against.
        # Without this the caller holds a preprocessor that was never iterated, and
        # frame_time() answers from the average-fps estimate instead of the real PTS.
        pre.adopt_frame_times(stream.frame_times, stream.decoded_indices)
        if not pre.has_real_frame_times:
            raise RuntimeError('no presentation timestamps reached the scoring process')
        return np.asarray(poses), np.asarray(indices), pre
    finally:
        # Always reclaim the decoder and the GPU, including on the failure path.
        stream.close()
        del slam
        torch.cuda.empty_cache()
