#!/usr/bin/env python3
"""Run a bounded, seeded teacher stage for workflow-only pilot recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PILOT_CLASSIFICATION = "workflow-pilot-not-paper-reproduction"
MAX_EXAMPLES = 128


class PilotTeacherError(ValueError):
    """Raised when a teacher stage is outside the pilot safety envelope."""


def _config_get(config: Any, dotted_key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        try:
            value = config.get(dotted_key, default)
        except TypeError:
            value = default
        if value is not default:
            return value
    value: Any = config
    for key in dotted_key.split("."):
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        else:
            return default
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PilotTeacherError(f"{field} must be a positive integer")
    return value


def validate_config(config: Any) -> str:
    pilot = _config_get(config, "energy_repro_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("classification") != PILOT_CLASSIFICATION:
        raise PilotTeacherError(
            "teacher wrapper requires the workflow-pilot-not-paper-reproduction classification"
        )
    requested = _positive_int(pilot.get("train_examples"), "train_examples")
    requested += _positive_int(pilot.get("eval_examples"), "eval_examples")
    if requested > MAX_EXAMPLES:
        raise PilotTeacherError(f"pilot input is limited to {MAX_EXAMPLES} examples")
    if bool(_config_get(config, "experiment.debug_mode", False)):
        raise PilotTeacherError("measured pilots must not use debug mode")
    expected_training = {
        "training.batch_size": 1,
        "training.eval_batch_size": 1,
        "training.gradient_accumulation_steps": 1,
        "training.num_epochs": 1,
    }
    for field, expected in expected_training.items():
        if _config_get(config, field) != expected:
            raise PilotTeacherError(f"{field} must be {expected} for a pilot")

    pipeline = str(_config_get(config, "pipeline", "")).lower()
    if pipeline == "kd":
        if _config_get(config, "distillation.top_k_logits") != 100:
            raise PilotTeacherError(
                "the pinned upstream KD cache hardcodes top_k_logits=100"
            )
    elif pipeline == "sft":
        if _config_get(config, "batch_size") != 1:
            raise PilotTeacherError(
                "top-level batch_size must be 1 to avoid synthetic label corruption"
            )
        if _config_get(config, "synthetic_data.generation.decoding_strategy") != "greedy":
            raise PilotTeacherError("synthetic pilot decoding must be greedy")
        generated = _positive_int(
            _config_get(config, "synthetic_data.max_gen_examples"),
            "synthetic_data.max_gen_examples",
        )
        new_tokens = _positive_int(
            _config_get(config, "synthetic_data.generation.max_new_tokens"),
            "synthetic_data.generation.max_new_tokens",
        )
        if generated > MAX_EXAMPLES or new_tokens > 128:
            raise PilotTeacherError(
                "synthetic pilot is limited to 128 examples and 128 generated tokens"
            )
    else:
        raise PilotTeacherError(f"unsupported pilot pipeline: {pipeline!r}")
    return pipeline


def _local_directory(config: Any, field: str) -> Path:
    value = _config_get(config, field)
    if not isinstance(value, str) or not value:
        raise PilotTeacherError(f"{field} must name a local directory")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise PilotTeacherError(
            f"{field} must be an existing absolute non-symlink directory"
        )
    return path


def _id_hash(split: Any) -> str:
    identifiers = []
    for index, record in enumerate(split):
        identifier = record.get("id", index)
        identifiers.append(str(identifier))
    payload = (
        json.dumps(identifiers, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_dataset(config: Any, datasets_module: Any) -> tuple[Any, dict[str, Any]]:
    dataset_path = _local_directory(config, "data.dataset_path")
    dataset = datasets_module.load_from_disk(str(dataset_path))
    if not isinstance(dataset, datasets_module.DatasetDict):
        raise PilotTeacherError("pilot input must be a DatasetDict")
    if set(dataset) != {"train", "test"}:
        raise PilotTeacherError("pilot input must contain exactly train and test splits")
    train_count = len(dataset["train"])
    eval_count = len(dataset["test"])
    if train_count < 1 or eval_count < 1 or train_count + eval_count > MAX_EXAMPLES:
        raise PilotTeacherError(
            f"pilot input must contain 2..{MAX_EXAMPLES} total examples"
        )
    expected_train = _config_get(config, "energy_repro_pilot.train_examples")
    expected_eval = _config_get(config, "energy_repro_pilot.eval_examples")
    if (train_count, eval_count) != (expected_train, expected_eval):
        raise PilotTeacherError(
            "pilot input split sizes do not match the resolved recipe"
        )
    return dataset, {
        "train_examples": train_count,
        "eval_examples": eval_count,
        "train_ids_sha256": _id_hash(dataset["train"]),
        "eval_ids_sha256": _id_hash(dataset["test"]),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o640)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _sequential_prepare_dataset(train_ds: Any, eval_ds: Any, config: Any):
    from torch.utils.data import DataLoader
    from distill_bench.core.utils import CustomPadCollator

    collator = CustomPadCollator(
        config.max_sequence_length,
        pad_token_id=100277,
    )
    return (
        DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            drop_last=True,
        ),
        DataLoader(
            eval_ds,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            drop_last=False,
        ),
    )


def run_teacher(config: Any) -> dict[str, Any]:
    import datasets
    from distill_bench.core.energy_logger import EnergyTracker
    from distill_bench.core.utils import fix_seed

    pipeline = validate_config(config)
    _local_directory(config, "data.tokenizer_name")
    if pipeline in {"kd", "sft"}:
        _local_directory(config, "model.teacher")
    _, input_summary = inspect_dataset(config, datasets)
    run_dir_value = _config_get(config, "output.run_dir")
    if not isinstance(run_dir_value, str) or not Path(run_dir_value).is_absolute():
        raise PilotTeacherError("output.run_dir must be an absolute path")
    run_dir = Path(run_dir_value)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = _config_get(config, "experiment.seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PilotTeacherError("experiment.seed must be a non-negative integer")
    fix_seed(seed)

    summary: dict[str, Any] = {
        "schema": "energy-repro/pilot-teacher-summary/v1",
        "classification": PILOT_CLASSIFICATION,
        "pipeline": pipeline,
        "seed": seed,
        "input": input_summary,
    }
    if pipeline == "kd":
        from distill_bench.data import logit_caching

        logit_caching.prepare_dataset = _sequential_prepare_dataset
        tracker = EnergyTracker(
            run_dir=str(run_dir),
            experiment_name="pilot_kd_logit_caching",
            config=config,
        )
        tracker.start_stage("pilot_kd_logit_caching")
        total_tokens = logit_caching.cache_teacher_logprobs(
            config,
            energy_tracker=tracker,
        )
        tracker.end_stage(tokens_processed=total_tokens)
        summary.update(
            {
                "deterministic_train_order": True,
                "top_k_logits": 100,
                "tokens_processed": total_tokens,
            }
        )
        tracker.save_summary(additional_metadata={"energy_repro_pilot": summary})
    else:
        from distill_bench.data.synthetic_generation import (
            generate_synthetic_dataset,
        )

        tracker = EnergyTracker(
            run_dir=str(run_dir),
            experiment_name="pilot_synthetic_generation",
            config=config,
        )
        synthetic = generate_synthetic_dataset(
            config,
            energy_tracker=tracker,
            stage_name="pilot_synthetic_generation",
        )
        summary.update(
            {
                "decoding_strategy": "greedy",
                "max_gen_examples": _config_get(
                    config,
                    "synthetic_data.max_gen_examples",
                ),
                "max_new_tokens": _config_get(
                    config,
                    "synthetic_data.generation.max_new_tokens",
                ),
                "output_train_examples": len(synthetic["train"]),
                "output_eval_examples": len(synthetic["test"]),
            }
        )
        tracker.save_summary(additional_metadata={"energy_repro_pilot": summary})

    _atomic_json(run_dir / "pilot-teacher-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "WARNING: WORKFLOW PILOT ONLY — NOT A PAPER REPRODUCTION.",
        flush=True,
    )
    from distill_bench.core.config_loader import load_config

    config = load_config(args.config)
    try:
        result = run_teacher(config)
    except PilotTeacherError as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "classification": PILOT_CLASSIFICATION,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
        )
        return 2
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
