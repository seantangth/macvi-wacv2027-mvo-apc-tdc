import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from pack_submission import pack


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'synthetic'
        subprocess.run([sys.executable, str(ROOT / 'examples/make_synthetic_example.py'),
                        '--output', str(self.root)], check=True, capture_output=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_packs_without_references_and_strips_extra_metadata(self):
        source = self.root / 'run/predictions/clip_000.json'
        pred = json.loads(source.read_text())
        pred['private_extra'] = 'must not be uploaded'
        source.write_text(json.dumps(pred))
        for ref in (self.root / 'references').glob('*.json'):
            ref.unlink()
        output = self.root / 'submission.json'
        count, _ = pack(self.root / 'release_manifest.json', self.root / 'run', output, 'val')
        self.assertGreater(count, 0)
        self.assertNotIn('private_extra', json.loads(output.read_text())['clip_000'])
        with self.assertRaises(FileExistsError):
            pack(self.root / 'release_manifest.json', self.root / 'run', output, 'val')

    def test_missing_prediction_leaves_no_output(self):
        (self.root / 'run/predictions/clip_000.json').unlink()
        output = self.root / 'submission.json'
        with self.assertRaises(FileNotFoundError):
            pack(self.root / 'release_manifest.json', self.root / 'run', output, 'val')
        self.assertFalse(output.exists())

    def test_bad_timestamps_are_rejected(self):
        source = self.root / 'run/predictions/clip_000.json'
        pred = json.loads(source.read_text())
        pred['times'][0] += 0.1
        source.write_text(json.dumps(pred))
        with self.assertRaises(ValueError):
            pack(self.root / 'release_manifest.json', self.root / 'run', self.root / 'submission.json', 'val')


if __name__ == '__main__':
    unittest.main()
