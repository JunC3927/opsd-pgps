"""Strict, CPU-only export and relocation of the original PGPS training data.

The source parquet remains byte-identical inside the independent data bundle.
Only image locations and the explicitly requested teacher suffix change when a
run parquet is prepared. Nothing in this module launches a model or judge.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

SOURCE_PARQUET_SHA = "9c69e065ec8432f5f66e324d69523c63237d61af9ea07e7d57a43b4e3627d577"
OFFICIAL_TRAIN_SHA = "dcb397f4f55ed5075c4174fbac304ea4719979217cc561023954b0c1692eb377"
OFFICIAL_TEST_SHA = "682fbfe54f02e3baf05f08194fc6b9ebaca3d190ac856f83c75a10f40264b298"
IMAGE_CONTENT_SHA = "5ef7a253285720150932d9a3b94248ad83f6b7f6e1dd01f244c1c10ea75c2a06"
TRAIN_ROWS, TRAIN_QUESTIONS, TEST_QUESTIONS = 8022, 8021, 1000
IMAGE_COUNT, IMAGE_BYTES, SMOKE_ROWS = 3776, 88566205, 18
PADDING_QID = "prob_0"
INTRO = ("The following annotations describe the geometric relationships and labeled values "
         "in the diagram. Use the relevant information to help solve the problem.")
FORMAT = "geo-opsd-pgps-original-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _safe_file(root, relative):
    _require(isinstance(relative, str) and "\\" not in relative, "Invalid bundle path")
    parts = PurePosixPath(relative)
    _require(not parts.is_absolute() and relative not in ("", ".")
             and ".." not in parts.parts and parts.as_posix() == relative, "Unsafe bundle path")
    current = root
    for part in parts.parts:
        current = current / part
        _require(not current.is_symlink(), "Bundle members must be ordinary files/directories")
    _require(current.is_file(), "Missing bundle file: " + relative)
    _require(current.resolve().is_relative_to(root.resolve()), "Bundle path escapes its root")
    return current


def _question_prompts(question):
    text = question["text"]
    structure, semantics = question["parsing_stru_seqs"], question["parsing_sem_seqs"]
    _require(isinstance(text, str) and "<image>" not in text and "<video>" not in text,
             "Original question contains an unexpected multimodal placeholder")
    _require(isinstance(structure, list) and isinstance(semantics, list)
             and all(isinstance(x, str) for x in structure + semantics), "Invalid original parsing")
    student = "<image>" + text
    teacher = (student + "\n\n" + INTRO + "\n\nStructural relations:\n" + "\n".join(structure)
               + "\n\nSemantic relations:\n" + "\n".join(semantics))
    return student, teacher


def _check_rows(rows, official, test):
    _require(isinstance(official, dict) and len(official) == TRAIN_QUESTIONS, "Unexpected official training split")
    _require(isinstance(test, dict) and len(test) == TEST_QUESTIONS, "Unexpected official test split")
    _require(set(official).isdisjoint(test), "Training and test question IDs overlap")
    _require(len(rows) == TRAIN_ROWS, "Wrong training row count")
    ids = [row["extra_info"]["index"] for row in rows]
    counts = Counter(ids)
    _require(set(ids) == set(official), "Training IDs do not match the official training split")
    _require({key: value for key, value in counts.items() if value != 1} == {PADDING_QID: 2},
             "Expected exactly the historical single duplicate")
    padded = [i for i, row in enumerate(rows) if row["extra_info"]["sampler_padding"] is True]
    _require(padded == [len(rows) - 1] and ids[-1] == ids[0] == PADDING_QID,
             "Historical padding location/order changed")
    expected_last = copy.deepcopy(rows[0])
    expected_last["extra_info"]["sampler_padding"] = True
    _require(rows[-1] == expected_last, "Padding row is not the original first row duplicate")
    images = {}
    for row in rows:
        qid = row["extra_info"]["index"]
        question = official[qid]
        student, teacher = _question_prompts(question)
        _require(row["prompt"] == [{"role": "user", "content": student}],
                 qid + ": student input differs from the exact original question")
        _require(row["teacher_prompt"] == [{"role": "user", "content": teacher}],
                 qid + ": teacher input differs from the original unmodified parsing")
        _require(row["reward_model"]["ground_truth"] == "", qid + ": ground_truth must be empty")
        _require(row["images"] == row["teacher_images"] and len(row["images"]) == 1,
                 qid + ": student/teacher images differ")
        _require(isinstance(row["extra_info"]["sampler_padding"], bool), "Padding flag must be boolean")
        raw_path = row["images"][0]["path"]
        _require(isinstance(raw_path, str), "Image path must be a string")
        image = Path(raw_path)
        name = image.name
        _require(name and name == question["diagram"] == row["extra_info"]["diagram_id"],
                 qid + ": image filename and diagram identity disagree")
        images.setdefault(name, set()).add(raw_path)
    _require(len(images) == IMAGE_COUNT, "Unexpected number of referenced training images")
    return {"rows": len(rows), "unique_questions": len(counts), "padding_id": PADDING_QID,
            "padding_positions": padded, "row_order_sha256": _canonical_sha(ids)}, images


def _expected_files():
    return {"source/train.parquet": SOURCE_PARQUET_SHA,
            "train.json": OFFICIAL_TRAIN_SHA, "test.json": OFFICIAL_TEST_SHA}


def _validated_bundle(data_root):
    import pyarrow.parquet as pq
    root = Path(data_root).resolve()
    manifest = _read(_safe_file(root, "manifest.json"))
    _require(manifest.get("format") == FORMAT, "Unsupported or legacy data bundle; export the strict v1 bundle first")
    _require(manifest.get("source_parquet_sha256") == SOURCE_PARQUET_SHA, "Wrong source parquet identity")
    _require(set(manifest["files"]) == set(_expected_files()), "Unexpected bundle core file list")
    for relative, expected in _expected_files().items():
        path = _safe_file(root, relative)
        record = manifest["files"][relative]
        _require(record["sha256"] == expected == _sha(path), "Bundle source SHA mismatch: " + relative)
        _require(record["bytes"] == path.stat().st_size, "Bundle source size mismatch: " + relative)
    table = pq.read_table(root / "source/train.parquet")
    rows = table.to_pylist()
    facts, source_images = _check_rows(rows, _read(root / "train.json"), _read(root / "test.json"))
    for key, value in facts.items():
        _require(manifest.get(key) == value, "Bundle row audit mismatch: " + key)
    image_hashes = manifest["images"]
    expected_relatives = {"training_images/" + name for name in source_images}
    _require(set(image_hashes) == expected_relatives, "Image manifest differs from referenced training images")
    basenames = {PurePosixPath(relative).name: digest for relative, digest in image_hashes.items()}
    _require(_canonical_sha(basenames) == IMAGE_CONTENT_SHA == manifest.get("image_content_sha256"),
             "Image manifest does not describe the original training bytes")
    sizes = {}
    for relative, expected in sorted(image_hashes.items()):
        image = _safe_file(root, relative)
        _require(_sha(image) == expected, "Training image SHA mismatch: " + relative)
        sizes[relative] = image.stat().st_size
    _require(sum(sizes.values()) == IMAGE_BYTES == manifest.get("image_bytes"), "Training image sizes differ")
    actual_images = {p.relative_to(root).as_posix() for p in (root / "training_images").rglob("*")
                     if p.is_file() or p.is_symlink()}
    _require(actual_images == expected_relatives, "Unexpected files in training_images")
    report = dict(facts, passed=True, format=FORMAT, source_parquet_sha256=SOURCE_PARQUET_SHA,
                  image_count=len(image_hashes), image_bytes=sum(sizes.values()),
                  image_content_sha256=IMAGE_CONTENT_SHA, all_image_hashes_verified=True,
                  manifest_sha256=_sha(root / "manifest.json"),
                  original_prompt_and_parsing_verified=True, test_ids_excluded=True)
    return root, table, rows, report


def verify_bundle(data_root: Path) -> dict:
    """Re-read and hash every required file. Does not write to the data bundle."""
    return _validated_bundle(data_root)[3]


def _new_temporary(output):
    output = Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), "Output already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".partial-", dir=output.parent))
    return output, temporary


def _publish(temporary, output):
    _require(not output.exists() and not output.is_symlink(), "Output appeared during preparation; refusing to overwrite")
    # Cooperating exports/preparations also use an exclusive reservation file.
    reservation = output.with_name("." + output.name + ".publish.lock")
    fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _require(not output.exists() and not output.is_symlink(), "Output appeared during preparation")
        temporary.rename(output)
    finally:
        os.close(fd)
        reservation.unlink()


def export_bundle(*, source_parquet: Path, official_train: Path, official_test: Path,
                  image_manifest: Path, output: Path) -> dict:
    """Copy the original parquet and exact training images to a new private bundle.

    image_manifest accepts the historical {files: [{path, bytes, sha256, ...}]}
    training_images.json. Source paths in that manifest are never exported.
    """
    import pyarrow.parquet as pq
    sources = {"source/train.parquet": Path(source_parquet), "train.json": Path(official_train),
               "test.json": Path(official_test)}
    for relative, expected in _expected_files().items():
        _require(_sha(sources[relative]) == expected, "Unexpected original source: " + relative)
    rows = pq.read_table(source_parquet).to_pylist()
    facts, referenced = _check_rows(rows, _read(official_train), _read(official_test))
    entries = _read(image_manifest)["files"]
    wanted = {}
    for entry in entries:
        name = PurePosixPath(entry["path"]).name
        _require(name not in wanted, "Duplicate basename in historical image manifest")
        wanted[name] = entry["sha256"]
    _require(set(wanted) == set(referenced) and _canonical_sha(wanted) == IMAGE_CONTENT_SHA,
             "Image manifest is not the original training image set")
    image_sources = {}
    for name, paths in referenced.items():
        # Same basename from two locations is allowed only if every byte hash agrees.
        for raw_path in paths:
            source = Path(raw_path)
            _require(source.is_file() and _sha(source) == wanted[name],
                     "Missing or changed training image: " + name)
            image_sources.setdefault(name, source)
    output, temporary = _new_temporary(output)
    try:
        (temporary / "source").mkdir()
        (temporary / "training_images").mkdir()
        files = {}
        for relative, source in sources.items():
            target = temporary / relative
            shutil.copyfile(source, target)
            files[relative] = {"sha256": _sha(target), "bytes": target.stat().st_size}
        image_hashes = {}
        image_bytes = 0
        for name, source in sorted(image_sources.items()):
            relative = "training_images/" + name
            target = temporary / relative
            shutil.copyfile(source, target)
            _require(_sha(target) == wanted[name], "Copied training image differs: " + name)
            image_hashes[relative] = wanted[name]
            image_bytes += target.stat().st_size
        manifest = dict(facts, format=FORMAT, source_parquet_sha256=SOURCE_PARQUET_SHA,
                        files=files, images=image_hashes, image_count=len(image_hashes),
                        image_content_sha256=IMAGE_CONTENT_SHA, image_bytes=image_bytes,
                        data_recipe="original_unconstrained", contains_model_weights=False,
                        contains_evaluation_responses=False)
        _write(temporary / "manifest.json", manifest)
        report = verify_bundle(temporary)
        _publish(temporary, output)
        return dict(report, output=str(output), training_started=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_data(data_root: Path, output_dir: Path, suffix: str) -> dict:
    """Prepare a new run's train/smoke parquet; preserve every logical source row.

    suffix is the exact model-input text, without a file-ending LF. Empty means
    the original teacher. No whitespace is stripped or implicitly normalized.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    _require(isinstance(suffix, str) and (not suffix or suffix.strip()), "Invalid teacher suffix")
    _require(not suffix.startswith(("\n", "\r")) and not suffix.endswith(("\n", "\r"))
             and "\r" not in suffix, "Pass the exact suffix without a file-ending newline")
    root, table, original, verified = _validated_bundle(data_root)
    rows = copy.deepcopy(original)
    for before, row in zip(original, rows):
        for field in ("images", "teacher_images"):
            for image in row[field]:
                relative = "training_images/" + Path(image["path"]).name
                image["path"] = str(_safe_file(root, relative))
        row["teacher_prompt"][0]["content"] += ("\n\n" + suffix) if suffix else ""
        expected = copy.deepcopy(before)
        for field in ("images", "teacher_images"):
            expected[field][0]["path"] = str(root / "training_images" / Path(before[field][0]["path"]).name)
        expected["teacher_prompt"][0]["content"] += ("\n\n" + suffix) if suffix else ""
        _require(row == expected, "A field outside image paths/teacher suffix changed")
    converted = pa.Table.from_pylist(rows, schema=table.schema)
    _require(converted.schema.equals(table.schema, check_metadata=True), "Source Arrow schema changed")
    _require(converted.to_pylist() == rows, "Arrow conversion changed row values")
    _require(len(rows) >= SMOKE_ROWS, "Too few rows for smoke")
    output, temporary = _new_temporary(output_dir)
    try:
        pq.write_table(converted, temporary / "train.parquet")
        pq.write_table(converted.slice(0, SMOKE_ROWS), temporary / "smoke.parquet")
        train = pq.read_table(temporary / "train.parquet")
        smoke = pq.read_table(temporary / "smoke.parquet")
        _require(train.to_pylist() == rows and train.schema.equals(table.schema, check_metadata=True),
                 "Written parquet changed row values or column types")
        _require(train.slice(0, SMOKE_ROWS).equals(smoke, check_metadata=True), "Smoke is not the exact prefix")
        report = dict(verified, smoke_rows=SMOKE_ROWS, only_teacher_suffix_changed=True,
                      image_paths_rewritten=True, image_bytes_unchanged=True,
                      student_prompts_unchanged=True, raw_parsing_unmodified=True,
                      all_other_columns_and_row_order_identical=True,
                      answers_options_or_system_added_to_input=False,
                      teacher_suffix=suffix, teacher_suffix_sha256=hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
                      suffix_appended_once=bool(suffix),
                      train_parquet=str(output / "train.parquet"), smoke_parquet=str(output / "smoke.parquet"),
                      train_parquet_sha256=_sha(temporary / "train.parquet"),
                      smoke_parquet_sha256=_sha(temporary / "smoke.parquet"),
                      audit_path=str(output / "audit.json"), source_schema_preserved=True,
                      preparation_only=True, training_started=False)
        _write(temporary / "audit.json", report)
        _publish(temporary, output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
