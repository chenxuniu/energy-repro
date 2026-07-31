from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "pilot_preprocess.py"
)
SPEC = importlib.util.spec_from_file_location("pilot_preprocess", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pilot_preprocess = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot_preprocess)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        del tokenize, add_generation_prompt
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        if text == pilot_preprocess.ASSISTANT_MARKER:
            self.assert_marker_kwargs = kwargs
            return {"input_ids": [8, 9]}
        encodings = {
            "good": {
                "input_ids": [1, 8, 9, 21, 22, 0],
                "attention_mask": [1, 1, 1, 1, 1, 0],
            },
            "truncated-marker": {
                "input_ids": [1, 8, 0, 0, 0, 0],
                "attention_mask": [1, 1, 0, 0, 0, 0],
            },
            "marker-without-response": {
                "input_ids": [1, 8, 9, 0, 0, 0],
                "attention_mask": [1, 1, 1, 0, 0, 0],
            },
        }
        return encodings[text]


class FakeTokenizerClass:
    loaded = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loaded = (path, kwargs)
        return FakeTokenizer()


class FakeDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def add_column(self, name, values):
        rows = []
        for row, value in zip(self.rows, values, strict=True):
            new_row = dict(row)
            new_row[name] = value
            rows.append(new_row)
        return FakeDataset(rows)

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeDataset(rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class FakeDatasetDict(dict):
    saved = None
    format_call = None

    def set_format(self, **kwargs):
        type(self).format_call = kwargs

    def save_to_disk(self, path):
        output = Path(path)
        output.mkdir(exist_ok=True)
        (output / "fake-dataset.txt").write_text("saved", encoding="utf-8")
        type(self).saved = self


class FakeDatasetsModule:
    Dataset = FakeDataset
    DatasetDict = FakeDatasetDict
    source = None
    loaded = None

    class DownloadConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @classmethod
    def load_dataset(cls, path, **kwargs):
        cls.loaded = (path, kwargs)
        return FakeDataset(cls.source)


class PilotValidationTests(unittest.TestCase):
    def _config(self, **overrides):
        pilot = {
            "classification": pilot_preprocess.PILOT_CLASSIFICATION,
            "train_examples": 8,
            "eval_examples": 2,
            "candidate_multiplier": 4,
            "workers": 2,
        }
        pilot.update(overrides)
        return {"energy_repro_pilot": pilot}

    def test_accepts_explicit_bounded_pilot(self):
        spec = pilot_preprocess.validate_pilot_config(self._config())
        self.assertEqual(spec.requested_examples, 10)
        self.assertEqual(spec.candidate_examples, 40)
        self.assertEqual(spec.workers, 2)

    def test_rejects_wrong_classification(self):
        with self.assertRaisesRegex(
            pilot_preprocess.PilotPreprocessError,
            "classification must be exactly",
        ):
            pilot_preprocess.validate_pilot_config(
                self._config(classification="paper-reproduction")
            )

    def test_rejects_bool_as_count(self):
        with self.assertRaisesRegex(
            pilot_preprocess.PilotPreprocessError,
            "positive integer",
        ):
            pilot_preprocess.validate_pilot_config(
                self._config(train_examples=True)
            )

    def test_rejects_unbounded_candidate_multiplier(self):
        with self.assertRaisesRegex(
            pilot_preprocess.PilotPreprocessError,
            f"at most {pilot_preprocess.MAX_CANDIDATE_MULTIPLIER}",
        ):
            pilot_preprocess.validate_pilot_config(
                self._config(
                    candidate_multiplier=(
                        pilot_preprocess.MAX_CANDIDATE_MULTIPLIER + 1
                    )
                )
            )


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "dataset"
        self.tokenizer = self.root / "tokenizer"
        self.output = self.root / "derived" / "pilot"
        self.dataset.mkdir()
        self.tokenizer.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _config(self):
        return {
            "data": {
                "dataset_name": str(self.dataset),
                "tokenizer_name": str(self.tokenizer),
                "dataset_path": str(self.output),
            },
            "energy_repro": {
                "assets": {
                    "dataset": {"path": str(self.dataset)},
                    "tokenizer": {"path": str(self.tokenizer)},
                }
            },
        }

    def test_accepts_matching_pinned_local_assets_and_new_output(self):
        self.assertEqual(
            pilot_preprocess.validate_runtime_paths(self._config()),
            (self.dataset, self.tokenizer, self.output),
        )

    def test_rejects_asset_path_not_bound_by_resolver(self):
        config = self._config()
        config["energy_repro"]["assets"]["dataset"]["path"] = str(
            self.root / "different"
        )
        with self.assertRaisesRegex(
            pilot_preprocess.PilotPreprocessError,
            "does not match the pinned local dataset",
        ):
            pilot_preprocess.validate_runtime_paths(config)

    def test_rejects_existing_output_without_removing_it(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(
            pilot_preprocess.PilotPreprocessError,
            "refusing to overwrite",
        ):
            pilot_preprocess.validate_runtime_paths(self._config())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


class BoundedPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "dataset"
        self.tokenizer = self.root / "tokenizer"
        self.output = self.root / "derived" / "pilot"
        self.dataset.mkdir()
        self.tokenizer.mkdir()
        FakeDatasetsModule.source = [
            {
                "id": f"row-{index}",
                "messages": [{"role": "user", "content": "good"}],
            }
            for index in range(9)
        ]
        FakeDatasetDict.saved = None
        FakeDatasetDict.format_call = None
        FakeDatasetsModule.loaded = None
        FakeTokenizerClass.loaded = None

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _config(self):
        return {
            "experiment": {"seed": 23},
            "data": {
                "dataset_name": str(self.dataset),
                "tokenizer_name": str(self.tokenizer),
                "dataset_path": str(self.output),
                "dataset_split": "train",
                "max_sequence_length": 6,
            },
            "energy_repro": {
                "assets": {
                    "dataset": {"path": str(self.dataset)},
                    "tokenizer": {"path": str(self.tokenizer)},
                }
            },
            "energy_repro_pilot": {
                "classification": pilot_preprocess.PILOT_CLASSIFICATION,
                "train_examples": 2,
                "eval_examples": 1,
                "candidate_multiplier": 2,
                "workers": 2,
            },
        }

    def test_builds_exact_splits_from_only_the_bounded_candidate_pool(self):
        summary, supervised_tokens = pilot_preprocess._run_preprocessing(
            self._config(),
            datasets_module=FakeDatasetsModule,
            tokenizer_class=FakeTokenizerClass,
        )
        self.assertEqual(summary["source_examples"], 9)
        self.assertEqual(summary["candidate_examples"], 6)
        self.assertEqual(
            summary["candidate_selection"],
            "python-random-sample-with-pinned-python",
        )
        self.assertEqual(summary["train_examples"], 2)
        self.assertEqual(summary["eval_examples"], 1)
        self.assertRegex(summary["train_ids_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["eval_ids_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            summary["train_ids_sha256"],
            summary["eval_ids_sha256"],
        )
        self.assertEqual(supervised_tokens, 6)
        self.assertEqual(len(FakeDatasetDict.saved["train"]), 2)
        self.assertEqual(len(FakeDatasetDict.saved["test"]), 1)
        self.assertTrue(self.output.is_dir())
        self.assertEqual(
            FakeDatasetDict.format_call,
            {
                "type": "torch",
                "columns": ["input_ids", "attention_mask", "labels", "id"],
            },
        )
        self.assertEqual(
            FakeTokenizerClass.loaded[1],
            {"local_files_only": True, "trust_remote_code": False},
        )
        self.assertTrue(
            FakeDatasetsModule.loaded[1]["download_config"].kwargs[
                "local_files_only"
            ]
        )


class LabelConstructionTests(unittest.TestCase):
    def test_labels_only_active_tokens_after_complete_marker(self):
        labels = pilot_preprocess.build_response_labels(
            [1, 8, 9, 21, 22, 0],
            [1, 1, 1, 1, 1, 0],
            [8, 9],
        )
        self.assertEqual(labels, [-100, -100, -100, 21, 22, -100])

    def test_incomplete_marker_is_rejected(self):
        self.assertIsNone(
            pilot_preprocess.build_response_labels(
                [1, 8, 0, 0],
                [1, 1, 0, 0],
                [8, 9],
            )
        )

    def test_marker_without_supervised_tokens_is_rejected(self):
        self.assertIsNone(
            pilot_preprocess.build_response_labels(
                [1, 8, 9, 0],
                [1, 1, 1, 0],
                [8, 9],
            )
        )


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def _sample(self, content, identifier="source-id"):
        return {
            "id": identifier,
            "messages": [{"role": "user", "content": content}],
        }

    def test_valid_candidate_has_masked_labels(self):
        record = pilot_preprocess.preprocess_candidate(
            self._sample("good", identifier="17"),
            source_index=5,
            tokenizer=self.tokenizer,
            marker_ids=[8, 9],
            max_sequence_length=6,
        )
        self.assertTrue(record["valid"])
        self.assertEqual(record["labels"], [-100, -100, -100, 21, 22, -100])
        self.assertEqual(record["_source_id"], "17")
        self.assertEqual(record["_source_index"], 5)

    def test_truncated_marker_and_empty_response_are_discarded(self):
        for text in ("truncated-marker", "marker-without-response"):
            record = pilot_preprocess.preprocess_candidate(
                self._sample(text),
                source_index=1,
                tokenizer=self.tokenizer,
                marker_ids=[8, 9],
                max_sequence_length=6,
            )
            self.assertFalse(record["valid"])
            self.assertEqual(
                record["reason"], "no_complete_supervised_response"
            )

    def test_parallel_processing_preserves_input_order(self):
        candidates = [
            (9, self._sample("good")),
            (2, self._sample("truncated-marker")),
            (7, self._sample("good")),
        ]
        records = pilot_preprocess.process_candidates(
            candidates,
            tokenizer=self.tokenizer,
            marker_ids=[8, 9],
            max_sequence_length=6,
            workers=3,
        )
        self.assertEqual(records[0]["_source_index"], 9)
        self.assertFalse(records[1]["valid"])
        self.assertEqual(records[2]["_source_index"], 7)


class StableIdTests(unittest.TestCase):
    @staticmethod
    def _record(source_index, source_id):
        return {
            "_source_index": source_index,
            "_source_id": source_id,
            "input_ids": [1],
            "attention_mask": [1],
            "labels": [1],
        }

    def test_preserves_unique_integer_compatible_ids(self):
        result = pilot_preprocess.assign_stable_ids(
            [self._record(5, 12), self._record(6, "13")]
        )
        self.assertEqual([row["id"] for row in result], [12, 13])

    def test_generates_deterministic_unique_int64_ids(self):
        records = [
            self._record(5, "not-an-int"),
            self._record(6, 7),
            self._record(7, 7),
        ]
        first = pilot_preprocess.assign_stable_ids(records)
        second = pilot_preprocess.assign_stable_ids(records)
        identifiers = [row["id"] for row in first]
        self.assertEqual(identifiers, [row["id"] for row in second])
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(
            all(
                isinstance(identifier, int)
                and 0 <= identifier <= pilot_preprocess.MAX_INT64
                for identifier in identifiers
            )
        )


if __name__ == "__main__":
    unittest.main()
