import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
import infer
import tracking


class InferenceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'release_manifest.json'
        self.dataset.write_text('{}', encoding='utf-8')
        self.output = self.root / 'run'
        self.clips = [dict(key=f'clip_{i:03}', split='val', video=f'{i}.mp4',
                           camera_mode='wide', calibration={}) for i in range(2)]
        for clip in self.clips:
            (self.root / clip['video']).touch()
        self.pre = SimpleNamespace(decoded_index=int, frame_time=float,
                                   native_fps=30.0, stride=4, effective_fps=7.5)
        self.success = (np.zeros((2, 7)), np.arange(2), self.pre)
        self.environment = ({'revision': 'abc123', 'buffer_size': 4096,
                             'checkpoint_sha256': 'f' * 64}, {'python': '3.10.12'})

    def run_inference(self, outcomes, extra=(), environment=None):
        stdout = io.StringIO()
        probe = (patch.object(infer, 'describe_environment', return_value=self.environment)
                 if environment is None else contextlib.nullcontext())
        with patch.object(infer, 'load_manifest', return_value={'clips': self.clips}), \
             patch.object(infer, 'load_timestamps', return_value=np.array([0., 1.])), \
             patch.object(infer, 'track', side_effect=outcomes) as tracker, \
             probe, contextlib.redirect_stdout(stdout):
            result = infer.main(['--dataset', str(self.dataset), '--output',
                                 str(self.output), *extra])
        self.assertEqual(result, 0)
        manifest = json.loads((self.output / 'inference_manifest.json').read_text())
        return manifest, stdout.getvalue(), tracker

    def test_full_error_log_survives_reason_truncation_and_next_clip_runs(self):
        error = 'x' * 400 + ' important trailing detail'
        run, stdout, tracker = self.run_inference(
            [RuntimeError(error), self.success], ['--dpvo-buffer-size', '8192'])
        first, second = run['clips']
        self.assertEqual(first['status'], 'tracking_failed')
        self.assertEqual(len(first['reason']), 300)
        log = (self.output / first['error_log']).read_text()
        self.assertIn('Traceback', log)
        self.assertIn(error, log)
        self.assertFalse((self.output / 'predictions/clip_000.json').exists())
        self.assertEqual(second['status'], 'predicted')
        self.assertEqual((second['native_fps'], second['stride'], second['effective_fps']),
                         (30., 4, 7.5))
        self.assertTrue(all(r['elapsed_seconds'] >= 0 for r in run['clips']))
        self.assertEqual(run['target_fps'], 8.)
        self.assertEqual(run['dpvo_buffer_size'], 8192)
        self.assertEqual(tracker.call_args.kwargs['dpvo_buffer_size'], 8192)
        self.assertIn('[1/2] clip_000: starting inference', stdout)
        self.assertIn('[2/2] clip_001: predicted', stdout)

    def test_non_finite_prediction_remains_failed_and_is_not_written(self):
        poses = np.zeros((2, 7))
        poses[1, 0] = np.nan
        run, _, _ = self.run_inference([(poses, np.arange(2), self.pre), self.success])
        self.assertEqual(run['clips'][0]['status'], 'tracking_failed')
        self.assertIn('non-finite', run['clips'][0]['reason'])
        self.assertIn('FloatingPointError',
                      (self.output / run['clips'][0]['error_log']).read_text())
        self.assertFalse((self.output / 'predictions/clip_000.json').exists())

    def test_requested_fps_and_seed_are_preserved(self):
        run, stdout, tracker = self.run_inference(
            [self.success, self.success], ['--target-fps', '80', '--seed', '1'])
        self.assertEqual((run['target_fps'], run['seed'], run['dpvo_buffer_size']),
                         (80., 1, None))
        self.assertEqual(tracker.call_args.args[2:4], (80., 1))
        self.assertIn('target_fps=80.0', stdout)

    def test_missing_video_still_records_progress_and_duration(self):
        (self.root / self.clips[0]['video']).unlink()
        run, stdout, tracker = self.run_inference([self.success])
        self.assertEqual(run['clips'][0]['status'], 'missing_video')
        self.assertIn('elapsed_seconds', run['clips'][0])
        self.assertIn('clip_000: missing_video', stdout)
        self.assertEqual(tracker.call_count, 1)

    def test_unwritable_error_log_does_not_abort_the_run(self):
        original = Path.write_text

        def write_text(path, *args, **kwargs):
            if path.suffix == '.log':
                raise OSError('disk full')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'write_text', autospec=True, side_effect=write_text):
            run, stdout, tracker = self.run_inference([RuntimeError('boom'), self.success])
        first, second = run['clips']
        self.assertEqual((first['status'], first['reason'], first['error_log']),
                         ('tracking_failed', 'boom', None))
        self.assertEqual(second['status'], 'predicted')
        self.assertEqual(tracker.call_count, 2)
        self.assertIn('Could not write traceback', stdout)

    def test_manifest_records_dpvo_revision_checkpoint_settings_and_versions(self):
        dpvo_dir = self.root / 'DPVO'
        (dpvo_dir / 'config').mkdir(parents=True)
        (dpvo_dir / 'config/default.yaml').write_text('PATCHES_PER_FRAME: 96\n')
        (dpvo_dir / 'dpvo.pth').write_bytes(b'checkpoint')
        git = ['git', '-C', str(dpvo_dir), '-c', 'user.name=t', '-c', 'user.email=t@t',
               '-c', 'commit.gpgsign=false']
        subprocess.run(['git', 'init', '-q', str(dpvo_dir)], check=True)
        subprocess.run(git + ['add', '.'], check=True)
        subprocess.run(git + ['commit', '-q', '-m', 'pin'], check=True)
        revision = subprocess.run(git + ['rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()

        class Config(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

            def clone(self):
                return Config(self)

            def merge_from_file(self, path):
                self['PATCHES_PER_FRAME'] = 96 if Path(path).is_file() else None

        shared = Config(BUFFER_SIZE=4096, PATCHES_PER_FRAME=80)
        torch = SimpleNamespace(__version__='2.5.1+cu121', version=SimpleNamespace(cuda='12.1'),
                                cuda=SimpleNamespace(is_available=lambda: True,
                                                     get_device_name=lambda _: 'Test GPU'))
        modules = {'torch': torch, 'cv2': SimpleNamespace(__version__='4.10.0'),
                   'dpvo.config': SimpleNamespace(cfg=shared)}
        for override, expected in ((None, 4096), (8192, 8192)):
            with self.subTest(override=override), patch.dict(sys.modules, modules), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch.object(tracking, 'DPVO_DIR', str(dpvo_dir)):
                output = self.root / f'run-{override}'
                self.output = output
                extra = () if override is None else ('--dpvo-buffer-size', str(override))
                run, stdout, _ = self.run_inference([self.success] * 2, extra, environment=True)
                self.assertEqual(run['dpvo_buffer_size'], override)
                self.assertEqual(run['dpvo']['buffer_size'], expected)
                self.assertEqual(run['dpvo']['config']['BUFFER_SIZE'], expected)
                self.assertEqual(run['dpvo']['config']['PATCHES_PER_FRAME'], 96)
                self.assertEqual(run['dpvo']['revision'], revision)
                self.assertEqual(run['dpvo']['checkpoint_sha256'],
                                 hashlib.sha256(b'checkpoint').hexdigest())
                self.assertEqual(run['environment'], dict(
                    python=platform.python_version(), numpy=np.__version__, opencv='4.10.0',
                    torch='2.5.1+cu121', torch_cuda='12.1', gpu='Test GPU'))
                self.assertIn(f'buffer_size={expected}', stdout)
                self.assertIn(f'revision={revision}', stdout)
        self.assertEqual(shared['BUFFER_SIZE'], 4096)

    def test_environment_without_dpvo_checkout_is_recorded_as_unknown(self):
        # A copied DPVO directory inside another repository must not report that repo's HEAD.
        subprocess.run(['git', 'init', '-q', str(self.root / 'outer')], check=True)
        plain = self.root / 'outer/DPVO'
        plain.mkdir()
        for dpvo_dir, override in ((None, None), (str(plain), 8192)):
            with self.subTest(dpvo_dir=dpvo_dir), \
                 patch.object(tracking, 'DPVO_DIR', dpvo_dir), \
                 patch.object(tracking, '_dpvo_config', side_effect=ImportError('no dpvo')):
                dpvo, env = tracking.describe_environment(override)
            self.assertEqual((dpvo['revision'], dpvo['checkpoint_sha256'], dpvo['buffer_size']),
                             ('unknown', None, override))
            self.assertEqual(env['python'], platform.python_version())

    def test_invalid_buffer_rejected_before_creating_output(self):
        for value in ('0', '-1', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as failure:
                infer.main(['--dataset', str(self.dataset), '--output', str(self.output),
                            '--dpvo-buffer-size', value])
            self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.output.exists())


class TrackingConfigurationTests(unittest.TestCase):
    def test_buffer_override_is_applied_after_yaml_without_mutating_shared_config(self):
        shared = Mock(BUFFER_SIZE=4096)
        configured = Mock(BUFFER_SIZE=4096)
        shared.clone.return_value = configured
        configured.merge_from_file.side_effect = lambda _: setattr(configured, 'BUFFER_SIZE', 4096)
        torch = SimpleNamespace(manual_seed=Mock(), no_grad=contextlib.nullcontext,
                                cuda=SimpleNamespace(empty_cache=Mock()))
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter(()))
        modules = {'torch': torch, 'dpvo.config': SimpleNamespace(cfg=shared),
                   'dpvo.dpvo': SimpleNamespace(DPVO=Mock())}
        with patch.dict(sys.modules, modules), \
             patch.object(sys, 'path', list(sys.path)), \
             patch.object(tracking, 'DPVO_DIR', 'fake-dpvo'), \
             patch.object(tracking, 'ClipPreprocessor'), \
             patch.object(tracking, 'ClipStream', return_value=stream):
            with self.assertRaisesRegex(RuntimeError, 'no frames decoded'):
                tracking.track('unused.mp4', 'wide', 8, 1, dpvo_buffer_size=8192)
        self.assertEqual(shared.BUFFER_SIZE, 4096)
        self.assertEqual(configured.BUFFER_SIZE, 8192)
        configured.merge_from_file.assert_called_once()
        stream.close.assert_called_once()
        torch.cuda.empty_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()
