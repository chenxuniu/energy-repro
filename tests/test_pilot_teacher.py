from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "pilot_teacher.py"
SPEC = importlib.util.spec_from_file_location("pilot_teacher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pilot_teacher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot_teacher)


def base_config(pipeline: str = "kd"):
    value = {
        "pipeline": pipeline,
        "energy_repro_pilot": {
            "classification": pilot_teacher.PILOT_CLASSIFICATION,
            "train_examples": 96,
            "eval_examples": 32,
        },
        "experiment": {"seed": 42, "debug_mode": False},
        "training": {
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
        },
        "distillation": {"top_k_logits": 100},
    }
    if pipeline == "sft":
        value.update({"batch_size": 1})
        value["synthetic_data"] = {
            "max_gen_examples": 64,
            "generation": {
                "decoding_strategy": "greedy",
                "max_new_tokens": 128,
            },
        }
    return value


class ValidationTests(unittest.TestCase):
    def test_accepts_bounded_kd_and_synthetic_sft(self):
        self.assertEqual(pilot_teacher.validate_config(base_config()), "kd")
        self.assertEqual(
            pilot_teacher.validate_config(base_config("sft")),
            "sft",
        )

    def test_rejects_paper_claim_debug_and_large_input(self):
        config = base_config()
        config["energy_repro_pilot"]["classification"] = "paper-reproduction"
        with self.assertRaisesRegex(
            pilot_teacher.PilotTeacherError,
            "classification",
        ):
            pilot_teacher.validate_config(config)

        config = base_config()
        config["experiment"]["debug_mode"] = True
        with self.assertRaisesRegex(pilot_teacher.PilotTeacherError, "debug"):
            pilot_teacher.validate_config(config)

        config = base_config()
        config["energy_repro_pilot"]["train_examples"] = 97
        with self.assertRaisesRegex(pilot_teacher.PilotTeacherError, "128"):
            pilot_teacher.validate_config(config)

    def test_rejects_accumulation_and_unsafe_synthetic_settings(self):
        config = base_config()
        config["training"]["gradient_accumulation_steps"] = 4
        with self.assertRaisesRegex(
            pilot_teacher.PilotTeacherError,
            "gradient_accumulation_steps",
        ):
            pilot_teacher.validate_config(config)

        for field, value, message in (
            ("batch_size", 2, "top-level batch_size"),
            ("decoding_strategy", "sampling", "greedy"),
            ("max_new_tokens", 129, "limited"),
        ):
            config = base_config("sft")
            if field == "batch_size":
                config[field] = value
            else:
                config["synthetic_data"]["generation"][field] = value
            with self.assertRaisesRegex(pilot_teacher.PilotTeacherError, message):
                pilot_teacher.validate_config(config)


class FakeSplit(list):
    pass


class FakeDatasetDict(dict):
    pass


class FakeDatasets:
    DatasetDict = FakeDatasetDict

    @staticmethod
    def load_from_disk(_path):
        return FakeDatasetDict(
            {
                "train": FakeSplit({"id": index} for index in range(3)),
                "test": FakeSplit({"id": index} for index in range(3, 5)),
            }
        )


class DatasetInspectionTests(unittest.TestCase):
    def test_exact_split_sizes_and_order_hashes_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            config = {
                "data": {"dataset_path": str(data)},
                "energy_repro_pilot": {
                    "train_examples": 3,
                    "eval_examples": 2,
                },
            }
            _, summary = pilot_teacher.inspect_dataset(
                config,
                FakeDatasets,
            )
            self.assertEqual(summary["train_examples"], 3)
            self.assertEqual(summary["eval_examples"], 2)
            self.assertRegex(summary["train_ids_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(
                summary["train_ids_sha256"],
                summary["eval_ids_sha256"],
            )

    def test_split_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            config = {
                "data": {"dataset_path": str(data)},
                "energy_repro_pilot": {
                    "train_examples": 4,
                    "eval_examples": 1,
                },
            }
            with self.assertRaisesRegex(
                pilot_teacher.PilotTeacherError,
                "split sizes",
            ):
                pilot_teacher.inspect_dataset(config, FakeDatasets)


if __name__ == "__main__":
    unittest.main()
