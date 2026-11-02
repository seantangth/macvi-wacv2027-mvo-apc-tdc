import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SyntheticExampleTests(unittest.TestCase):
    def test_readme_workflow_and_missing_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'example'
            subprocess.run([sys.executable, str(ROOT/'examples/make_synthetic_example.py'),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            command = [sys.executable, str(ROOT/'starter/evaluate.py'),
                       '--dataset', str(output/'release_manifest.json'),
                       '--predictions', str(output/'run'), '--split', 'val']
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output/'run/summary.json').read_text())
            self.assertEqual((summary['expected'], summary['scored'], summary['unscored']), (1, 1, 0))
            self.assertGreater(summary['median_drift_pct_scored'], 0)
            self.assertLess(summary['median_drift_pct_scored'], 5)
            (output/'run/predictions/clip_000.json').unlink()
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output/'run/summary.json').read_text())
            self.assertEqual(summary['by_status'], {'missing_prediction': 1})
            self.assertIsNone(summary['median_drift_pct_scored'])

    def test_generator_does_not_overwrite_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp)/'keep.txt'
            marker.write_text('existing content')
            result = subprocess.run([sys.executable, str(ROOT/'examples/make_synthetic_example.py'),
                                     '--output', temp], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), 'existing content')


if __name__ == '__main__':
    unittest.main()
