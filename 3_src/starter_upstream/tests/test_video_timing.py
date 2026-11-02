"""Exercise real MP4 decoding despite inaccurate container frame metadata."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from preprocessing import ClipPreprocessor

class VideoTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = str(Path(self.temp.name) / 'clip.mp4')
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(14):
            writer.write(np.full((48,64,3), i*10, dtype=np.uint8))
        writer.release()

    def test_metadata_disagreement_preserves_decoded_indices_and_times(self):
        capture = cv2.VideoCapture
        for delta in [-1, 1]:
            class MetadataOffset:
                def __init__(self, path): self.cap = capture(path)
                def get(self, field):
                    value = self.cap.get(field)
                    return value+delta if field == cv2.CAP_PROP_FRAME_COUNT else value
                def __getattr__(self, name): return getattr(self.cap, name)
            times = np.arange(14)/30
            with self.subTest(delta=delta), patch('preprocessing.cv2.VideoCapture', MetadataOffset):
                pre = ClipPreprocessor(self.video, 'wide', frame_timestamps=times)
                frames = list(pre.frames_with_time())
            self.assertEqual([f[4] for f in frames], [3,7,11])
            np.testing.assert_array_equal([f[3] for f in frames], times[[3,7,11]])
            self.assertEqual(pre.n_frames,14)

    def test_missing_decoded_frame_rejects_even_unselected_tail(self):
        pre = ClipPreprocessor(self.video, 'wide', frame_timestamps=np.arange(15)/30)
        with self.assertRaisesRegex(ValueError, 'fewer frames'):
            list(pre.frames_with_time())

    def test_extra_decoded_frame_rejects_even_unselected_tail(self):
        pre = ClipPreprocessor(self.video, 'wide', frame_timestamps=np.arange(13)/30)
        with self.assertRaisesRegex(ValueError, 'more frames'):
            list(pre.frames_with_time())

    def test_target_above_native_fps_keeps_every_frame_without_upsampling(self):
        times = np.arange(14)/30
        pre = ClipPreprocessor(self.video, 'wide', target_fps=80,
                               frame_timestamps=times)
        frames = list(pre.frames_with_time())
        self.assertEqual(pre.stride, 1)
        self.assertAlmostEqual(pre.effective_fps, 30)
        self.assertEqual([frame[4] for frame in frames], list(range(14)))
