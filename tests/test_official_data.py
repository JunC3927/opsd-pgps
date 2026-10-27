"""Real PNG/Parquet checks for loading an ordinary official download directly."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geo_opsd import official_data as data


class OfficialDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.download = self.root / 'download'
        self.images = self.download / 'Diagram_Visual'
        self.images.mkdir(parents=True)
        split = self.download / 'PGPS9K'
        split.mkdir()
        self.train = split / 'train.json'
        self.test = split / 'test.json'
        Image.new('RGB', (232, 203), 'red').save(self.images / 'img_0.png')
        Image.new('RGBA', (67, 51), (0, 0, 255, 128)).save(self.images / 'img_1.png')
        self.questions = {
            qid: {'text': 'Find x.  Keep spacing.\nNew line.', 'diagram': 'img_' + str(i % 2) + '.png',
                  'parsing_stru_seqs': ['line A B C', 'line A D'], 'parsing_sem_seqs': ['AB = 2'],
                  'choices': ['PRIVATE_OPTION'], 'answer': 'PRIVATE_GOLD', 'expression': 'PRIVATE_EXPRESSION'}
            for i, qid in enumerate(['prob_10', 'prob_0', 'prob_2'])}
        self.write()
        self.test.write_text(json.dumps({'prob_99': {'answer': 'TEST_ONLY'}}))
        self.patches = patch.multiple(data, TRAIN_QUESTIONS=3, TRAIN_ROWS=4, SMOKE_ROWS=3)
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def write(self):
        self.train.write_text(json.dumps(self.questions))

    def test_direct_download_preserves_images_and_only_teacher_gets_parsing(self):
        before = {p: p.read_bytes() for p in self.images.iterdir()}
        report = data.prepare_official(self.download, self.root / 'cache', 'PRIVATE_GUIDANCE')
        rows = pq.read_table(report['train_parquet']).to_pylist()
        self.assertEqual([r['extra_info']['index'] for r in rows], ['prob_0', 'prob_2', 'prob_10', 'prob_0'])
        for row in rows:
            question = self.questions[row['extra_info']['index']]
            self.assertEqual(row['prompt'], [{'role': 'user', 'content': '<image>' + question['text']}])
            teacher = row['teacher_prompt'][0]['content']
            self.assertIn('line A B C\nline A D', teacher)
            self.assertTrue(teacher.endswith('\n\nPRIVATE_GUIDANCE'))
            self.assertEqual(row['images'], row['teacher_images'])
            self.assertEqual(row['images'][0]['path'], str(self.images / question['diagram']))
            self.assertEqual(row['reward_model']['ground_truth'], '')
            for secret in ('PRIVATE_GOLD', 'PRIVATE_OPTION', 'PRIVATE_EXPRESSION', 'TEST_ONLY'):
                self.assertNotIn(secret, json.dumps(row))
        self.assertTrue(rows[-1]['extra_info']['sampler_padding'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.images.iterdir()})
        self.assertEqual(report['images']['img_0.png']['size'], [232, 203])
        self.assertEqual(report['images']['img_1.png']['mode'], 'RGBA')
        self.assertFalse(report['source_images_copied'])
        self.assertEqual(sorted(p.name for p in (self.root / 'cache').iterdir()), ['audit.json', 'smoke.parquet', 'train.parquet'])
        data.verify_sources(report)

    def test_flat_and_nested_split_layouts(self):
        self.assertEqual(data.locate(self.download)[0], self.train)
        self.assertEqual(data.locate(self.train.parent)[1], self.images)
        flat = self.download / 'train.json'
        self.train.rename(flat)
        self.assertEqual(data.locate(self.download)[0], flat)

    def test_no_fallback_to_wrong_diagram_folder(self):
        self.images.rename(self.download / 'Diagram')
        with self.assertRaisesRegex(ValueError, 'Diagram_Visual'):
            data.load_rows(self.download, '')

    def test_missing_corrupt_or_escaping_image_is_rejected(self):
        original = copy.deepcopy(self.questions)
        for name in ('missing.png', '../train.json'):
            self.questions['prob_0']['diagram'] = name
            self.write()
            with self.assertRaises(ValueError):
                data.load_rows(self.download, '')
        self.questions = original
        self.write()
        (self.images / 'img_0.png').write_bytes(b'broken PNG')
        with self.assertRaises(OSError):
            data.load_rows(self.download, '')

    def test_test_overlap_and_missing_relations_are_rejected(self):
        self.test.write_text(json.dumps({'prob_0': {}}))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data.load_rows(self.download, '')
        self.test.unlink()
        del self.questions['prob_0']['parsing_stru_seqs']
        self.write()
        with self.assertRaises(KeyError):
            data.load_rows(self.download, '')

    def test_changed_source_is_detected_and_existing_cache_is_preserved(self):
        report = data.prepare_official(self.download, self.root / 'cache', '')
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            data.prepare_official(self.download, self.root / 'cache', '')
        self.questions['prob_0']['text'] = 'Changed question'
        self.write()
        with self.assertRaisesRegex(ValueError, 'JSON changed'):
            data.verify_sources(report)
        self.questions['prob_0']['text'] = 'Find x.  Keep spacing.\nNew line.'
        self.write()
        Image.new('RGB', (232, 203), 'green').save(self.images / 'img_0.png')
        with self.assertRaisesRegex(ValueError, 'image changed'):
            data.verify_sources(report)


if __name__ == '__main__':
    unittest.main()
