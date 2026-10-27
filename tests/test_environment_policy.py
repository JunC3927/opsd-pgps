"""Compatibility boundaries: allow tooling changes, reject numerical/runtime drift."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('environment_policy', ROOT / 'environment/verify_environment.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class EnvironmentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads((ROOT / 'environment/expected_versions.json').read_text())
        self.actual = dict(self.baseline['packages'], pip=self.baseline['bootstrap_pip'])

    def test_compatible_accepts_and_records_tooling_changes(self):
        self.actual.update(wandb='0.29.0', tensorboard='2.21.0', tqdm='4.68.0',
                           click='8.4.0', pip='26.0', **{'example-extra-dependency': '1.0'})
        del self.actual['fastapi-cloud-cli']  # Unneeded historical CLI is no longer a root requirement.
        result = policy.audit_versions(self.actual, self.baseline, 'compatible')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['package_mismatches'], {})
        self.assertIn('wandb', result['baseline_differences'])
        self.assertIn('fastapi-cloud-cli', result['baseline_differences'])
        self.assertIn('example-extra-dependency', result['extra_packages'])

    def test_compatible_rejects_changed_or_missing_core(self):
        for package in ('torch', 'vllm', 'transformers', 'tokenizers', 'flash-attn',
                        'ray', 'numpy', 'pillow', 'qwen-vl-utils', 'nvidia-nccl-cu12'):
            for value in ('0.0.0', None):
                actual = dict(self.actual)
                if value is None:
                    del actual[package]
                else:
                    actual[package] = value
                with self.subTest(package=package, version=value):
                    result = policy.audit_versions(actual, self.baseline, 'compatible')
                    self.assertTrue(result['errors'])
                    self.assertIn(package, result['package_mismatches'])

    def test_historical_keeps_original_strict_policy(self):
        self.assertEqual(policy.audit_versions(self.actual, self.baseline, 'historical')['errors'], [])
        self.actual.update(tqdm='4.68.0', pip='26.0', **{'new-extra': '1.0'})
        result = policy.audit_versions(self.actual, self.baseline, 'historical')
        self.assertEqual(len(result['errors']), 3)

    def test_resolver_requirements_retain_every_core_pin(self):
        pins = policy.core_versions(self.baseline)
        requirements = (ROOT / 'environment/requirements.compatible.txt').read_text().splitlines()
        for name, version in pins.items():
            if name != 'flash-attn':  # Installed only from the historical hash-verified wheel.
                self.assertIn(name + '==' + version, requirements)
        self.assertIn('-c constraints.core.txt', requirements)
        self.assertFalse(any(line.startswith('flash-attn') for line in requirements))

    def test_python_patch_is_flexible_but_abi_is_not(self):
        self.assertTrue(policy.python_matches('3.12.10', '3.12'))
        self.assertTrue(policy.python_matches('3.12.14', '3.12.14'))
        self.assertFalse(policy.python_matches('3.13.0', '3.12'))
        self.assertFalse(policy.python_matches('3.12.10', '3.12.14'))


if __name__ == '__main__':
    unittest.main()
