"""The convenience command must keep GPU smoke gates and preserve existing jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geo_opsd import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.exp = self.output / 'experiment'
        self.value = {'experiment_dir': str(self.exp), 'run_dir': str(self.exp / 'run'),
                      'smoke_dir': str(self.exp / 'smoke'), 'gpus': [0, 1, 2, 3, 4, 5]}
        self.calls = []
        self.settings = {'paths': {'output': str(self.output)}, 'experiment': {'name': 'experiment'}}
        for name, replacement in [('load_settings', lambda _: self.settings),
                                  ('prepared', lambda _: self.value), ('fingerprint', lambda _: 'fixed'),
                                  ('execute', self.execute)]:
            p = patch.object(workflow, name, replacement)
            p.start()
            self.addCleanup(p.stop)

    def marker(self, name, **fields):
        self.exp.mkdir(exist_ok=True)
        (self.exp / name).write_text(json.dumps({'passed': True, 'fingerprint': 'fixed', **fields}))

    def execute(self, script, config, *extra):
        self.calls.append((script, extra))
        if script == 'prepare.py':
            self.marker('PREPARED.json')
        elif script == 'preflight.py':
            self.marker('CPU_PREFLIGHT_PASSED.json')
            self.marker('CODE_FROZEN.json')
        elif script == 'smoke_check.py':
            self.marker('SMOKE_PASSED.json')

    def test_full_workflow_preserves_order_and_reuses_training_gpu(self):
        workflow.run('config')
        self.assertEqual(self.calls, [('prepare.py', ()), ('preflight.py', ()),
                                    ('launch.py', ('--mode', 'smoke')),
                                    ('smoke_check.py', ('--gpu', '0')),
                                    ('launch.py', ('--mode', 'train'))])

    def test_prepare_only_never_launches_gpu_work(self):
        workflow.run('config', prepare_only=True)
        self.assertEqual([call[0] for call in self.calls], ['prepare.py', 'preflight.py'])

    def test_failed_smoke_never_starts_formal_training(self):
        def fail(script, config, *extra):
            if script == 'launch.py':
                raise subprocess.CalledProcessError(1, script)
            self.execute(script, config, *extra)
        with patch.object(workflow, 'execute', fail), self.assertRaises(subprocess.CalledProcessError):
            workflow.run('config')
        self.assertFalse(any('--mode' in args and 'train' in args for _, args in self.calls))

    def test_existing_failed_or_active_smoke_is_not_restarted(self):
        self.marker('PREPARED.json')
        self.marker('CPU_PREFLIGHT_PASSED.json')
        self.marker('CODE_FROZEN.json')
        smoke = Path(self.value['smoke_dir'])
        smoke.mkdir()
        (smoke / 'process.json').write_text('{}')
        with patch.object(workflow, 'finished', side_effect=AssertionError('active/failed')), \
             self.assertRaises(AssertionError):
            workflow.run('config')
        self.assertEqual(self.calls, [])

    def test_changed_fingerprint_cannot_bypass_checks(self):
        self.marker('PREPARED.json')
        self.marker('CPU_PREFLIGHT_PASSED.json', fingerprint='changed')
        with self.assertRaisesRegex(AssertionError, 'changed'):
            workflow.run('config')
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
