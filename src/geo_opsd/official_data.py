"""Read the official PGPS9K download; images stay in place without conversion."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

from .data import (TRAIN_QUESTIONS, TRAIN_ROWS, SMOKE_ROWS, _canonical_sha,
                   _new_temporary, _publish, _question_prompts, _require, _sha, _write)


def locate(data_root):
    root = Path(data_root).expanduser().resolve()
    candidates = [root / 'train.json', root / 'PGPS9K/train.json', root / 'PGPS9K/PGPS9K/train.json']
    train = next((p for p in candidates if p.is_file()), None)
    _require(train is not None, 'No PGPS9K train.json found under ' + str(root))
    folders = [train.parent / 'Diagram_Visual', train.parent.parent / 'Diagram_Visual', root / 'Diagram_Visual']
    images = next((p.resolve() for p in folders if p.is_dir()), None)
    _require(images is not None, 'Missing Diagram_Visual beside PGPS9K/train.json; extract the official images first')
    test = train.with_name('test.json')
    return train.resolve(), images, test.resolve() if test.is_file() else None


def load_rows(data_root, suffix):
    from PIL import Image
    _require(isinstance(suffix, str) and (not suffix or suffix.strip()), 'Invalid teacher suffix')
    _require(not suffix.startswith(('\n', '\r')) and not suffix.endswith(('\n', '\r'))
             and '\r' not in suffix, 'Teacher suffix must not include a trailing file newline')
    train, image_dir, test = locate(data_root)
    train_bytes = train.read_bytes()
    official = json.loads(train_bytes)
    _require(isinstance(official, dict) and len(official) == TRAIN_QUESTIONS,
             'Expected the complete official PGPS9K training split (8021 questions)')
    _require(all(isinstance(qid, str) and re.fullmatch(r'prob_\d+', qid) for qid in official), 'Invalid PGPS9K question ID')
    test_bytes = test.read_bytes() if test else None
    if test:
        test_rows = json.loads(test_bytes)
        _require(isinstance(test_rows, dict) and set(official).isdisjoint(test_rows), 'Training/test IDs overlap')
    ids = sorted(official, key=lambda qid: (int(qid.split('_')[1]), qid))
    rows, image_records = [], {}
    for qid in ids:
        question = official[qid]
        _require(isinstance(question, dict), 'Invalid question: ' + qid)
        name = question['diagram']
        _require(isinstance(name, str) and name not in ('', '.', '..')
                 and Path(name).name == name and '\\' not in name, 'Invalid image filename: ' + qid)
        image = image_dir / name
        _require(image.is_file(), 'Missing training image: ' + str(image))
        if name not in image_records:
            before = image.stat()
            with Image.open(image) as opened:
                size, mode = list(opened.size), opened.mode
                opened.verify()
            digest = _sha(image)
            after = image.stat()
            _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                     'Image changed while reading: ' + name)
            image_records[name] = {'path': str(image), 'sha256': digest, 'bytes': after.st_size,
                                   'mtime_ns': after.st_mtime_ns, 'size': size, 'mode': mode}
        student, teacher = _question_prompts(question)
        if suffix:
            teacher += '\n\n' + suffix
        rows.append({'data_source': 'pgps9k_clight_port',
                     'prompt': [{'role': 'user', 'content': student}],
                     'teacher_prompt': [{'role': 'user', 'content': teacher}],
                     'images': [{'path': str(image)}], 'teacher_images': [{'path': str(image)}],
                     'ability': 'geometry', 'reward_model': {'style': 'rule', 'ground_truth': ''},
                     'extra_info': {'index': qid, 'diagram_id': name, 'sampler_padding': False}})
    padding = copy.deepcopy(rows[0])
    padding['extra_info']['sampler_padding'] = True
    rows.append(padding)
    _require(len(rows) == TRAIN_ROWS, 'Unexpected padded row count')
    report = {'passed': True, 'data_mode': 'official', 'rows': len(rows),
              'unique_questions': len(official), 'padding_id': ids[0], 'padding_positions': [len(ids)],
              'row_order_policy': 'numeric PGPS problem ID, then one first-row padding copy',
              'row_order_sha256': _canonical_sha([r['extra_info']['index'] for r in rows]),
              'source_train': str(train), 'source_train_sha256': hashlib.sha256(train_bytes).hexdigest(),
              'source_test': str(test) if test else None,
              'source_test_sha256': hashlib.sha256(test_bytes).hexdigest() if test else None,
              'test_ids_excluded': bool(test), 'image_directory': str(image_dir),
              'image_count': len(image_records), 'images': image_records,
              'image_content_sha256': _canonical_sha({name: entry['sha256'] for name, entry in image_records.items()}),
              'source_images_copied': False, 'source_images_resized': False,
              'image_policy': 'Official Diagram_Visual; RGB conversion and model processor at runtime',
              'uses_historical_training_image_bytes': False, 'original_prompt_and_parsing_verified': True,
              'raw_parsing_unmodified': True, 'answers_options_or_system_added_to_input': False,
              'teacher_suffix': suffix, 'teacher_suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
              'suffix_appended_once': bool(suffix), 'training_started': False}
    return rows, report


def prepare_official(data_root, output_dir, suffix):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows, report = load_rows(data_root, suffix)
    table = pa.Table.from_pylist(rows)
    _require(table.to_pylist() == rows, 'Arrow conversion changed training inputs')
    output, temporary = _new_temporary(output_dir)
    try:
        pq.write_table(table, temporary / 'train.parquet')
        pq.write_table(table.slice(0, SMOKE_ROWS), temporary / 'smoke.parquet')
        stored = pq.read_table(temporary / 'train.parquet')
        _require(stored.to_pylist() == rows, 'Training cache round trip failed')
        # Parquet normalizes nested list field names; compare the two persisted
        # tables, while the check above guarantees the original values survived.
        _require(pq.read_table(temporary / 'smoke.parquet').equals(stored.slice(0, SMOKE_ROWS), check_metadata=True),
                 'Smoke cache must be the exact first 18 rows')
        report.update(smoke_rows=SMOKE_ROWS, train_parquet=str(output / 'train.parquet'),
                      smoke_parquet=str(output / 'smoke.parquet'), cache_generated_automatically=True,
                      train_parquet_sha256=_sha(temporary / 'train.parquet'),
                      smoke_parquet_sha256=_sha(temporary / 'smoke.parquet'))
        _write(temporary / 'audit.json', report)
        _publish(temporary, output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_sources(report):
    for key in ('source_train', 'source_test'):
        if report.get(key):
            _require(_sha(report[key]) == report[key + '_sha256'], 'Official JSON changed since preparation: ' + key)
    for name, entry in report['images'].items():
        path = Path(entry['path'])
        _require(path.is_file() and _sha(path) == entry['sha256'], 'Official image changed since preparation: ' + name)
