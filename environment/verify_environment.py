#!/usr/bin/env python3
"""Audit fixed core or historical freeze. GPU discovery requires the explicit --cuda flag."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def core_versions(expected):
    pins = {}
    for line in (ROOT / 'constraints.core.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, version = line.strip().split('==')
        name = normalize(name)
        if expected['packages'].get(name) != version:
            raise ValueError('Core constraint differs from historical baseline: ' + name)
        pins[name] = version
    return pins


def audit_versions(actual, expected, profile):
    required = expected['packages'] if profile == 'historical' else core_versions(expected)
    differences = {key: {'expected': version, 'actual': actual.get(key)}
                   for key, version in expected['packages'].items() if actual.get(key) != version}
    mismatches = {key: {'expected': version, 'actual': actual.get(key)}
                  for key, version in required.items() if actual.get(key) != version}
    extras = {key: version for key, version in actual.items()
              if key not in expected['packages'] and key not in expected['allowed_extra_packages']}
    errors, warnings = [], []
    if mismatches:
        errors.append('Historical package pin mismatch' if profile == 'historical' else 'Training/inference core pin mismatch')
    if profile == 'historical':
        if extras:
            errors.append('Unexpected extra packages in historical mode')
        if actual.get('pip') != expected['bootstrap_pip']:
            errors.append('pip bootstrap version differs from the recorded old overlay')
    elif differences or extras or actual.get('pip') != expected['bootstrap_pip']:
        warnings.append('Auxiliary versions/dependency set differ from the historical freeze; recorded for provenance')
    return {'profile': profile, 'required_pin_count': len(required), 'package_mismatches': mismatches,
            'baseline_differences': differences, 'extra_packages': extras,
            'pip_version': actual.get('pip'), 'errors': errors, 'warnings': warnings}


def python_matches(actual, requested):
    return actual.split('.')[:2] == ['3', '12'] and (requested == '3.12' or actual == requested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--profile', choices=('compatible', 'historical'), default='compatible')
    parser.add_argument('--expected-python', default=None)
    parser.add_argument('--imports', action='store_true')
    parser.add_argument('--cuda', action='store_true', help='Explicitly inspect the visible CUDA devices')
    parser.add_argument('--expect-gpu-regex', default=None, help='For example H100; only with --cuda')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.expected_python = args.expected_python or ('3.12.14' if args.profile == 'historical' else '3.12')
    if args.expect_gpu_regex and not args.cuda:
        parser.error('--expect-gpu-regex requires --cuda')
    expected = json.loads((ROOT / 'expected_versions.json').read_text())
    errors, warnings = [], []
    report = {'python': platform.python_version(), 'prefix': sys.prefix,
              'platform': platform.platform(), 'machine': platform.machine(),
              'baseline_python': expected['baseline_python'],
              'requested_python': args.expected_python,
              'cuda_probe_requested': args.cuda, 'training_smoke_completed': False}
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        errors.append('The historical native wheels require Linux x86_64')
    libc_name, libc_version = platform.libc_ver()
    report['libc'] = {'name': libc_name, 'version': libc_version}
    # The historical torch wheel is tagged manylinux_2_28_x86_64.
    if libc_name != 'glibc' or tuple(int(x) for x in libc_version.split('.')[:2]) < (2, 28):
        errors.append('The historical torch wheel needs glibc >= 2.28')
    if not python_matches(platform.python_version(), args.expected_python):
        errors.append('Python differs from the requested CPython 3.12 version / ABI')
    if platform.python_version() != expected['baseline_python']:
        warnings.append('Python patch differs from the historical version; CPython 3.12 ABI is still required')
    if args.prefix and args.prefix.resolve() != Path(sys.prefix).resolve():
        errors.append('Wrong Python interpreter prefix')
    distributions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        if name:
            distributions.setdefault(normalize(name), []).append(distribution)
    actual = {key: values[0].version for key, values in distributions.items()}
    audit = audit_versions(actual, expected, args.profile)
    errors.extend(audit.pop('errors'))
    warnings.extend(audit.pop('warnings'))
    report.update(audit)
    report['duplicate_distributions'] = {key: [d.version for d in values]
                                         for key, values in distributions.items() if len(values) > 1}
    if report['duplicate_distributions']:
        errors.append('Duplicate package metadata; environment may be ambiguous')
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.imports or args.cuda:
        report['imports'] = {}
        for name in ['torch', 'flash_attn', 'transformers', 'vllm', 'ray', 'datasets',
                     'hydra', 'peft', 'tensordict', 'wandb', 'tensorboard']:
            try:
                module = importlib.import_module(name)
                location = Path(module.__file__).resolve()
                confined = location.is_relative_to(Path(sys.prefix).resolve())
                report['imports'][name] = {'path': str(location), 'inside_prefix': confined}
                if not confined:
                    errors.append('Import outside this environment: ' + name)
            except Exception as exc:
                report['imports'][name] = {'error': repr(exc)}
                errors.append('Import failure: ' + name)
        if 'torch' in sys.modules:
            import torch
            report['torch'] = {'runtime_version': torch.__version__,
                               'cuda_runtime': torch.version.cuda,
                               'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI}
            if torch.version.cuda != '12.8' or not torch._C._GLIBCXX_USE_CXX11_ABI:
                errors.append('Torch CUDA runtime / ABI differs from the historical native wheel')
            if args.cuda:
                report['cuda_available'] = torch.cuda.is_available()
                report['visible_gpu_count'] = torch.cuda.device_count()
                report['gpus'] = []
                if not report['cuda_available']:
                    errors.append('No visible CUDA device; inspect driver and CUDA_VISIBLE_DEVICES')
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    report['gpus'].append({'visible_index': i, 'name': props.name,
                                           'capability': [props.major, props.minor],
                                           'total_memory_bytes': props.total_memory})
                    if args.expect_gpu_regex and not re.search(args.expect_gpu_regex, props.name):
                        errors.append('Unexpected GPU model: ' + props.name)
                # Read-only driver inventory, never captures process/env credentials.
                try:
                    query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                            '--format=csv,noheader'], capture_output=True, text=True,
                                           timeout=20, check=True)
                    report['driver_inventory'] = query.stdout.strip().splitlines()
                except Exception as exc:
                    warnings.append('Driver inventory unavailable: ' + type(exc).__name__)
                warnings.append('CUDA discovery is not a FA2 kernel or distributed training test')
    report['errors'] = errors
    report['warnings'] = warnings
    report['passed'] = not errors
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
