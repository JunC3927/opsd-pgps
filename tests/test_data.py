"""CPU tests with real Arrow round trips and synthetic, non-PGPS fixtures.

Run in the declared environment: python -m unittest discover -s tests -p test_data.py -v
No model, GPU, network, or API is used. Missing pyarrow is a test failure, not a
skip: the data migration cannot be validated without its actual storage library.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geo_opsd import data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class DataMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.images = self.source / 'images'
        self.images.mkdir()
        self.image_content = {'a.png': b'png-byte-fixture-A\x00\xff', 'b.png': b'png-byte-fixture-B\x00'}
        for name, value in self.image_content.items():
            (self.images / name).write_bytes(value)
        self.official = {
            'prob_0': {'text': 'Find x.  Keep  spacing.', 'diagram': 'a.png',
                       'parsing_stru_seqs': ['line A B C'], 'parsing_sem_seqs': ['AB = 2', 'BC = 3'],
                       'answer': 'PRIVATE_GOLD_5', 'choices': ['A. PRIVATE_CHOICE_5']},
            'prob_1': {'text': 'Find the angle.\nUse the diagram.', 'diagram': 'b.png',
                       'parsing_stru_seqs': ['line D E'], 'parsing_sem_seqs': ['DE \\perp EF'],
                       'answer': 'PRIVATE_GOLD_90', 'choices': ['B. PRIVATE_CHOICE_90']},
            'prob_2': {'text': 'Find y.', 'diagram': 'a.png', 'parsing_stru_seqs': [],
                       'parsing_sem_seqs': ['AB = 2'], 'answer': 'PRIVATE_GOLD_2', 'choices': []},
        }
        self.test = {'test_0': {'text': 'Excluded test question'}}
        self.train_json, self.test_json = self.source / 'train.json', self.source / 'test.json'
        dump(self.train_json, self.official)
        dump(self.test_json, self.test)
        self.rows = []
        for qid, question in self.official.items():
            student = '<image>' + question['text']
            teacher = (student + '\n\n' + data.INTRO + '\n\nStructural relations:\n'
                       + '\n'.join(question['parsing_stru_seqs']) + '\n\nSemantic relations:\n'
                       + '\n'.join(question['parsing_sem_seqs']))
            self.rows.append({
                'data_source': 'synthetic-only', 'ability': 'geometry',
                'prompt': [{'role': 'user', 'content': student}],
                'teacher_prompt': [{'role': 'user', 'content': teacher}],
                'images': [{'path': str(self.images / question['diagram'])}],
                'teacher_images': [{'path': str(self.images / question['diagram'])}],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'index': qid, 'diagram_id': question['diagram'], 'sampler_padding': False},
                'unchanged_nested': {'list': [1, 2], 'nullable': None},
            })
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.rows[-1]['extra_info']['sampler_padding'] = True
        self.parquet = self.source / 'train.parquet'
        self._write_rows()
        self.image_manifest = self.source / 'training_images.json'
        dump(self.image_manifest, {'files': [{'path': 'data/training_images/' + name,
                                            'sha256': sha(self.images / name), 'bytes': len(content)}
                                           for name, content in self.image_content.items()]})
        self.pins = {
            'SOURCE_PARQUET_SHA': sha(self.parquet), 'OFFICIAL_TRAIN_SHA': sha(self.train_json),
            'OFFICIAL_TEST_SHA': sha(self.test_json),
            'IMAGE_CONTENT_SHA': data._canonical_sha({name: sha(self.images / name) for name in self.image_content}),
            'TRAIN_ROWS': 4, 'TRAIN_QUESTIONS': 3, 'TEST_QUESTIONS': 1, 'IMAGE_COUNT': 2,
            'IMAGE_BYTES': sum(map(len, self.image_content.values())), 'SMOKE_ROWS': 2,
        }
        # Only synthetic fixture identity/count constants are substituted. Arrow
        # readers, writers, hashes, validation, and copying are real code paths.
        self.patcher = patch.multiple(data, **self.pins)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _write_rows(self):
        table = pa.Table.from_pylist(self.rows).replace_schema_metadata({b'fixture': b'preserve this metadata'})
        pq.write_table(table, self.parquet)

    def _export(self, name='bundle'):
        target = self.base / name
        report = data.export_bundle(source_parquet=self.parquet, official_train=self.train_json,
                                    official_test=self.test_json, image_manifest=self.image_manifest,
                                    output=target)
        self.assertTrue(report['passed'])
        return target

    def test_export_and_prepare_preserve_source_schema_sequence_and_fields(self):
        bundle = self._export()
        source_bytes = self.parquet.read_bytes()
        self.assertEqual((bundle / 'source/train.parquet').read_bytes(), source_bytes)
        self.assertNotIn(str(self.source), (bundle / 'manifest.json').read_text())
        all_hashes_before = {p.relative_to(bundle).as_posix(): sha(p) for p in bundle.rglob('*') if p.is_file()}
        suffix = 'Use the reference privately.\n\nDo not cite it!'
        report = data.prepare_data(bundle, self.base / 'prepared', suffix)
        self.assertEqual((report['rows'], report['unique_questions'], report['smoke_rows']), (4, 3, 2))
        self.assertTrue(report['only_teacher_suffix_changed'])
        self.assertTrue(report['all_other_columns_and_row_order_identical'])
        actual = pq.read_table(report['train_parquet'])
        self.assertTrue(actual.schema.equals(pq.read_table(self.parquet).schema, check_metadata=True))
        expected = copy.deepcopy(self.rows)
        for row in expected:
            for field in ('images', 'teacher_images'):
                row[field][0]['path'] = str(bundle / 'training_images' / Path(row[field][0]['path']).name)
            row['teacher_prompt'][0]['content'] += '\n\n' + suffix
        self.assertEqual(actual.to_pylist(), expected)
        self.assertEqual(pq.read_table(report['smoke_parquet']).to_pylist(), expected[:2])
        self.assertEqual(json.loads(Path(report['audit_path']).read_text()), report)
        self.assertEqual(all_hashes_before, {p.relative_to(bundle).as_posix(): sha(p) for p in bundle.rglob('*') if p.is_file()})
        self.assertEqual(self.parquet.read_bytes(), source_bytes)
        for row in actual.to_pylist():
            for field in ('prompt', 'teacher_prompt'):
                self.assertNotIn('PRIVATE_GOLD', row[field][0]['content'])
                self.assertNotIn('PRIVATE_CHOICE', row[field][0]['content'])

    def test_relocated_bundle_does_not_need_any_original_image_paths(self):
        bundle = self._export()
        moved = self.base / 'new-machine' / 'bundle'
        moved.parent.mkdir()
        bundle.rename(moved)
        shutil.rmtree(self.source)
        report = data.prepare_data(moved, self.base / 'new-machine' / 'run-data', '')
        actual = pq.read_table(report['train_parquet']).to_pylist()
        for before, after in zip(self.rows, actual):
            self.assertEqual(before['teacher_prompt'], after['teacher_prompt'])
            self.assertEqual(before['prompt'], after['prompt'])
            self.assertTrue(Path(after['images'][0]['path']).is_file())
        self.assertFalse(report['suffix_appended_once'])

    def test_changed_image_and_rewritten_manifest_both_rejected(self):
        bundle = self._export()
        image = bundle / 'training_images/a.png'
        image.write_bytes(b'X' * image.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'image SHA'):
            data.verify_bundle(bundle)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        manifest['images']['training_images/a.png'] = sha(image)
        manifest['image_content_sha256'] = data._canonical_sha({Path(k).name: v for k, v in manifest['images'].items()})
        dump(bundle / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'original training bytes'):
            data.verify_bundle(bundle)

    def test_source_identity_is_fixed(self):
        self.parquet.write_bytes(self.parquet.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Unexpected original source'):
            self._export()
        self.assertFalse((self.base / 'bundle').exists())

    def test_semantic_audit_rejects_prompt_answer_parse_and_padding_changes(self):
        original = copy.deepcopy(self.rows)
        changes = [
            ('extra student instruction', lambda r: r[1]['prompt'][0].update(content=r[1]['prompt'][0]['content'] + '\nThink step by step')),
            ('teacher parsing change', lambda r: r[1]['teacher_prompt'][0].update(content=r[1]['teacher_prompt'][0]['content'] + '\nEF=9')),
            ('gold leak', lambda r: r[1]['reward_model'].update(ground_truth='90')),
            ('padding moved', lambda r: r.reverse()),
            ('nonidentical duplicate', lambda r: r[-1].update(ability='different')),
            ('mismatched teacher image', lambda r: r[1]['teacher_images'][0].update(path=str(self.images / 'a.png'))),
        ]
        for label, change in changes:
            with self.subTest(label=label):
                self.rows = copy.deepcopy(original)
                change(self.rows)
                self._write_rows()
                with patch.object(data, 'SOURCE_PARQUET_SHA', sha(self.parquet)), self.assertRaises(ValueError):
                    self._export()
                self.assertFalse((self.base / 'bundle').exists())

    def test_same_basename_with_different_bytes_is_rejected(self):
        other = self.base / 'different-source'
        other.mkdir()
        (other / 'a.png').write_bytes(b'wrong bytes')
        for field in ('images', 'teacher_images'):
            self.rows[2][field][0]['path'] = str(other / 'a.png')
        self._write_rows()
        with patch.object(data, 'SOURCE_PARQUET_SHA', sha(self.parquet)), self.assertRaisesRegex(ValueError, 'changed training image'):
            self._export()

    def test_symlink_and_extra_image_are_rejected(self):
        bundle = self._export()
        target = bundle / 'training_images/a.png'
        target.unlink()
        target.symlink_to(self.images / 'a.png')
        with self.assertRaisesRegex(ValueError, 'ordinary files'):
            data.verify_bundle(bundle)
        target.unlink()
        target.write_bytes(self.image_content['a.png'])
        (bundle / 'training_images/extra.png').write_bytes(b'unexpected')
        with self.assertRaisesRegex(ValueError, 'Unexpected files'):
            data.verify_bundle(bundle)

    def test_traversal_and_existing_output_rejected_without_overwrite(self):
        bundle = self._export()
        for path in ('../train.json', '/train.json', 'training_images/../train.json', './train.json', 'train\\json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                data._safe_file(bundle, path)
        target = self.base / 'prepared'
        report = data.prepare_data(bundle, target, 'Exact suffix!')
        before = {p.name: sha(p) for p in target.iterdir()}
        with self.assertRaisesRegex(ValueError, 'already exists'):
            data.prepare_data(bundle, target, 'A different suffix!')
        self.assertEqual(before, {p.name: sha(p) for p in target.iterdir()})
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self._export()
        self.assertEqual(report['train_parquet_sha256'], sha(target / 'train.parquet'))

    def test_suffix_file_newline_is_not_silently_stripped(self):
        bundle = self._export()
        for suffix in ('constraint\n', '\nconstraint', 'constraint\r\n', '   '):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                data.prepare_data(bundle, self.base / 'prepared', suffix)
        self.assertFalse((self.base / 'prepared').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
