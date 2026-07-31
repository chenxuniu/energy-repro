#!/usr/bin/env python3
"""Create a small, deterministic Tulu dataset for workflow-only pilot runs.

This tool intentionally lives in the reproducibility capsule rather than the
pinned upstream checkout.  It is fail-closed: only an explicitly labelled
pilot config is accepted, all inputs must be local, and an existing output
path is never removed or overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


PILOT_CLASSIFICATION = "workflow-pilot-not-paper-reproduction"
ASSISTANT_MARKER = "<|assistant|>\n"
MAX_CANDIDATE_MULTIPLIER = 32
MAX_WORKERS = 64
MAX_PILOT_EXAMPLES = 128
MAX_CANDIDATE_EXAMPLES = MAX_PILOT_EXAMPLES * MAX_CANDIDATE_MULTIPLIER
MAX_INT64 = (1 << 63) - 1
GENERATED_ID_BASE = 1 << 62
WARNING = (
    "WARNING: WORKFLOW PILOT ONLY — NOT A PAPER REPRODUCTION. "
    "Results from this bounded dataset must not be compared with paper results."
)


class PilotPreprocessError(ValueError):
    """Raised when pilot preprocessing cannot proceed safely."""


class PilotSpec(NamedTuple):
    """Validated, bounded controls supplied by a resolved pilot recipe."""

    train_examples: int
    eval_examples: int
    candidate_multiplier: int
    workers: int

    @property
    def requested_examples(self) -> int:
        return self.train_examples + self.eval_examples

    @property
    def candidate_examples(self) -> int:
        return self.requested_examples * self.candidate_multiplier


def _config_get(config: Any, dotted_key: str, default: Any = None) -> Any:
    """Read a dotted value from an upstream Config object or plain mapping."""

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


def _positive_int(value: Any, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotPreprocessError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise PilotPreprocessError(f"{field} must be at most {maximum}")
    return value


def validate_pilot_config(config: Any) -> PilotSpec:
    """Validate the explicit pilot safety envelope and return typed controls."""

    raw = _config_get(config, "energy_repro_pilot")
    if not isinstance(raw, Mapping):
        raise PilotPreprocessError(
            "resolved config must contain an energy_repro_pilot mapping"
        )
    if raw.get("classification") != PILOT_CLASSIFICATION:
        raise PilotPreprocessError(
            "energy_repro_pilot.classification must be exactly "
            f"{PILOT_CLASSIFICATION!r}"
        )

    train_examples = _positive_int(raw.get("train_examples"), "train_examples")
    eval_examples = _positive_int(raw.get("eval_examples"), "eval_examples")
    candidate_multiplier = _positive_int(
        raw.get("candidate_multiplier"),
        "candidate_multiplier",
        MAX_CANDIDATE_MULTIPLIER,
    )
    workers = _positive_int(raw.get("workers"), "workers", MAX_WORKERS)
    spec = PilotSpec(
        train_examples=train_examples,
        eval_examples=eval_examples,
        candidate_multiplier=candidate_multiplier,
        workers=workers,
    )
    if spec.requested_examples > MAX_PILOT_EXAMPLES:
        raise PilotPreprocessError(
            "pilot train_examples + eval_examples must be at most "
            f"{MAX_PILOT_EXAMPLES}"
        )
    if spec.candidate_examples > MAX_CANDIDATE_EXAMPLES:
        raise PilotPreprocessError(
            f"bounded candidate pool must be at most {MAX_CANDIDATE_EXAMPLES}"
        )
    return spec


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotPreprocessError(f"{field} must be a non-negative integer")
    return value


def _require_local_directory(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PilotPreprocessError(f"{field} must name a pinned local directory")
    path = Path(value)
    if not path.is_absolute():
        raise PilotPreprocessError(f"{field} must be an absolute local path")
    if path.is_symlink() or not path.is_dir():
        raise PilotPreprocessError(
            f"{field} must be an existing, non-symlink local directory: {path}"
        )
    return path


def validate_runtime_paths(config: Any) -> tuple[Path, Path, Path]:
    """Return verified dataset, tokenizer, and new output directories."""

    dataset_name = _require_local_directory(
        _config_get(config, "data.dataset_name", getattr(config, "dataset_name", None)),
        "data.dataset_name",
    )
    tokenizer_name = _require_local_directory(
        _config_get(
            config,
            "data.tokenizer_name",
            getattr(config, "tokenizer_name", None),
        ),
        "data.tokenizer_name",
    )
    pinned_dataset = _config_get(config, "energy_repro.assets.dataset.path")
    pinned_tokenizer = _config_get(config, "energy_repro.assets.tokenizer.path")
    if pinned_dataset != str(dataset_name):
        raise PilotPreprocessError(
            "data.dataset_name does not match the pinned local dataset asset"
        )
    if pinned_tokenizer != str(tokenizer_name):
        raise PilotPreprocessError(
            "data.tokenizer_name does not match the pinned local tokenizer asset"
        )
    output_value = _config_get(
        config,
        "data.dataset_path",
        getattr(config, "dataset_path", None),
    )
    if not isinstance(output_value, str) or not output_value:
        raise PilotPreprocessError("data.dataset_path must name a new local directory")
    output_path = Path(output_value)
    if not output_path.is_absolute():
        raise PilotPreprocessError("data.dataset_path must be an absolute local path")
    if output_path.exists() or output_path.is_symlink():
        raise PilotPreprocessError(
            f"refusing to overwrite existing dataset path: {output_path}"
        )
    if output_path.is_relative_to(dataset_name) or output_path.is_relative_to(
        tokenizer_name
    ):
        raise PilotPreprocessError(
            "data.dataset_path must not be inside a pinned input asset"
        )
    return dataset_name, tokenizer_name, output_path


def _as_int_list(value: Any, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PilotPreprocessError(f"tokenizer returned invalid {field}")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PilotPreprocessError(f"tokenizer returned non-integer {field}")
        result.append(item)
    return result


def find_subsequence(sequence: Sequence[int], needle: Sequence[int]) -> int | None:
    """Return the first complete subsequence offset, or ``None``."""

    if not needle or len(needle) > len(sequence):
        return None
    for start in range(len(sequence) - len(needle) + 1):
        if list(sequence[start : start + len(needle)]) == list(needle):
            return start
    return None


def build_response_labels(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    marker_ids: Sequence[int],
) -> list[int] | None:
    """Mask everything except active tokens following the first full marker."""

    if len(input_ids) != len(attention_mask):
        raise PilotPreprocessError("input_ids and attention_mask lengths differ")
    active_positions = [
        index for index, mask_value in enumerate(attention_mask) if int(mask_value) != 0
    ]
    active_ids = [input_ids[index] for index in active_positions]
    marker_start = find_subsequence(active_ids, marker_ids)
    if marker_start is None:
        return None
    response_start = marker_start + len(marker_ids)
    supervised_positions = active_positions[response_start:]
    if not supervised_positions:
        return None
    labels = [-100] * len(input_ids)
    for position in supervised_positions:
        labels[position] = int(input_ids[position])
    return labels


def _source_integer_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str):
        try:
            candidate = int(value, 10)
        except ValueError:
            return None
    else:
        return None
    if 0 <= candidate <= MAX_INT64:
        return candidate
    return None


def assign_stable_ids(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve unique integer IDs and deterministically replace unusable ones."""

    result: list[dict[str, Any]] = []
    used: set[int] = set()
    for record in records:
        source_index = _require_nonnegative_int(
            record.get("_source_index"), "_source_index"
        )
        identifier = _source_integer_id(record.get("_source_id"))
        if identifier is None or identifier in used:
            identifier = GENERATED_ID_BASE + source_index
            while identifier in used and identifier <= MAX_INT64:
                identifier += 1
            if identifier > MAX_INT64:
                identifier = source_index
                while identifier in used:
                    identifier += 1
                if identifier > MAX_INT64:
                    raise PilotPreprocessError("unable to allocate a unique int64 ID")
        used.add(identifier)
        clean = {
            "id": identifier,
            "input_ids": list(record["input_ids"]),
            "attention_mask": list(record["attention_mask"]),
            "labels": list(record["labels"]),
        }
        result.append(clean)
    return result


def stable_id_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the selected ID order without exposing source examples."""

    payload = (
        json.dumps(
            [int(record["id"]) for record in records],
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocess_candidate(
    sample: Mapping[str, Any],
    *,
    source_index: int,
    tokenizer: Any,
    marker_ids: Sequence[int],
    max_sequence_length: int,
) -> dict[str, Any]:
    """Convert one source row into a model-ready record or rejection reason."""

    messages = sample.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return {"valid": False, "reason": "missing_messages"}
    try:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            chat_text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
            return_attention_mask=True,
        )
        input_ids = _as_int_list(encoded["input_ids"], "input_ids")
        attention_mask = _as_int_list(
            encoded["attention_mask"], "attention_mask"
        )
        if len(input_ids) != max_sequence_length:
            raise PilotPreprocessError(
                "tokenizer did not honor padding='max_length'"
            )
        labels = build_response_labels(input_ids, attention_mask, marker_ids)
    except (KeyError, TypeError, ValueError, PilotPreprocessError):
        return {"valid": False, "reason": "format_or_tokenization_error"}
    if labels is None:
        return {"valid": False, "reason": "no_complete_supervised_response"}
    return {
        "valid": True,
        "_source_index": source_index,
        "_source_id": sample.get("id"),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def process_candidates(
    candidates: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    tokenizer: Any,
    marker_ids: Sequence[int],
    max_sequence_length: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Process candidates concurrently while preserving deterministic order."""

    def process(item: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        source_index, sample = item
        return preprocess_candidate(
            sample,
            source_index=source_index,
            tokenizer=tokenizer,
            marker_ids=marker_ids,
            max_sequence_length=max_sequence_length,
        )

    if workers == 1:
        return [process(item) for item in candidates]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(process, candidates))


def _run_preprocessing(
    config: Any,
    *,
    datasets_module: Any,
    tokenizer_class: Any,
) -> tuple[dict[str, Any], int]:
    spec = validate_pilot_config(config)
    dataset_path, tokenizer_path, output_path = validate_runtime_paths(config)
    seed = _require_nonnegative_int(
        _config_get(config, "experiment.seed", getattr(config, "seed", None)),
        "experiment.seed",
    )
    max_sequence_length = _positive_int(
        _config_get(
            config,
            "data.max_sequence_length",
            getattr(config, "max_sequence_length", None),
        ),
        "data.max_sequence_length",
    )

    tokenizer = tokenizer_class.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    marker_ids = _as_int_list(
        tokenizer(ASSISTANT_MARKER, add_special_tokens=False)["input_ids"],
        "assistant marker input_ids",
    )
    if not marker_ids:
        raise PilotPreprocessError("assistant marker tokenization is empty")

    split = _config_get(
        config,
        "data.dataset_split",
        getattr(config, "dataset_split", None),
    ) or "train"
    download_config = datasets_module.DownloadConfig(local_files_only=True)
    source = datasets_module.load_dataset(
        str(dataset_path),
        split=split,
        download_config=download_config,
    )
    if len(source) == 0:
        raise PilotPreprocessError("source dataset is empty")
    source_examples = len(source)

    candidate_count = min(len(source), spec.candidate_examples)
    # Select only the bounded candidate count. Hugging Face Dataset.shuffle()
    # builds an index permutation for the entire source dataset, which makes a
    # 128-example pilot still scale with every source row.
    candidate_indices = random.Random(seed).sample(
        range(len(source)),
        candidate_count,
    )
    source = source.select(candidate_indices)
    candidates = [
        (source_index, row)
        for source_index, row in zip(candidate_indices, source, strict=True)
    ]
    processed = process_candidates(
        candidates,
        tokenizer=tokenizer,
        marker_ids=marker_ids,
        max_sequence_length=max_sequence_length,
        workers=spec.workers,
    )
    valid = [record for record in processed if record.get("valid") is True]
    if len(valid) < spec.requested_examples:
        rejection_counts: dict[str, int] = {}
        for record in processed:
            if record.get("valid") is not True:
                reason = str(record.get("reason") or "unknown")
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        raise PilotPreprocessError(
            "bounded candidate pool produced too few usable examples: "
            f"needed {spec.requested_examples}, found {len(valid)} from "
            f"{candidate_count}; rejections={json.dumps(rejection_counts, sort_keys=True)}"
        )

    selected = assign_stable_ids(valid[: spec.requested_examples])
    train_records = selected[: spec.train_examples]
    eval_records = selected[spec.train_examples :]
    if len(train_records) != spec.train_examples or len(eval_records) != spec.eval_examples:
        raise PilotPreprocessError("internal error: exact pilot split sizes were not met")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise PilotPreprocessError(
            f"refusing to overwrite existing dataset path: {output_path}"
        )
    final_dataset = datasets_module.DatasetDict(
        {
            "train": datasets_module.Dataset.from_list(train_records),
            "test": datasets_module.Dataset.from_list(eval_records),
        }
    )
    final_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "id"],
    )
    try:
        output_path.mkdir()
    except FileExistsError as exc:
        raise PilotPreprocessError(
            f"refusing to overwrite existing dataset path: {output_path}"
        ) from exc
    final_dataset.save_to_disk(str(output_path))

    supervised_tokens = sum(
        1
        for record in selected
        for token in record["labels"]
        if token != -100
    )
    summary = {
        "schema": "energy-repro/pilot-preprocess-summary/v1",
        "classification": PILOT_CLASSIFICATION,
        "status": "ok",
        "seed": seed,
        "source_examples": source_examples,
        "candidate_examples": candidate_count,
        "candidate_selection": "python-random-sample-with-pinned-python",
        "valid_candidates": len(valid),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "train_ids_sha256": stable_id_hash(train_records),
        "eval_ids_sha256": stable_id_hash(eval_records),
        "max_sequence_length": max_sequence_length,
        "supervised_tokens": supervised_tokens,
        "output_path": str(output_path),
    }
    return summary, supervised_tokens


def _run_with_runtime(config: Any, run_dir: Path) -> dict[str, Any]:
    # Heavy runtime dependencies are deliberately imported only after config
    # validation begins and never when this module's pure helpers are imported.
    import datasets
    from distill_bench.core.energy_logger import EnergyTracker
    from transformers import AutoTokenizer

    tracker = EnergyTracker(
        run_dir=str(run_dir),
        experiment_name="pilot_preprocess",
        config=config,
    )
    tracker.start_stage("pilot_preprocess")
    try:
        summary, supervised_tokens = _run_preprocessing(
            config,
            datasets_module=datasets,
            tokenizer_class=AutoTokenizer,
        )
    except BaseException:
        if tracker.current_stage is not None:
            tracker.end_stage(tokens_processed=0)
        tracker.save_summary(
            additional_metadata={
                "energy_repro_pilot": {
                    "classification": PILOT_CLASSIFICATION,
                    "status": "error",
                }
            }
        )
        raise
    tracker.end_stage(tokens_processed=supervised_tokens)
    tracker.save_summary(
        additional_metadata={"energy_repro_pilot": summary}
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic bounded dataset for a workflow-only Energy pilot."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Recipe-resolved experiment YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(WARNING, flush=True)

    # Loading the pinned upstream package also remains a runtime-only action.
    from distill_bench.core.config_loader import load_config

    config = load_config(args.config)
    run_dir_value = (
        getattr(config, "run_dir", None)
        or _config_get(config, "output.run_dir")
        or getattr(config, "output_dir", None)
    )
    if not isinstance(run_dir_value, str) or not run_dir_value:
        raise PilotPreprocessError("resolved config must define output.run_dir")
    run_dir = Path(run_dir_value)
    if not run_dir.is_absolute():
        raise PilotPreprocessError("output.run_dir must be an absolute local path")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = _run_with_runtime(config, run_dir)
    except PilotPreprocessError as exc:
        print(
            json.dumps(
                {
                    "classification": PILOT_CLASSIFICATION,
                    "status": "error",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
