#!/usr/bin/env python3
"""Create a deterministic, sanitized, public result bundle from one run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


EXPORT_SCHEMA = "energy-repro/public-result/v1"
RECEIPT_SCHEMA = "energy-repro/public-result-receipt/v1"
DEFAULT_MAX_FILE_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024
TERMINAL_STATES = {"succeeded", "failed", "interrupted"}
REDACTED = "<redacted>"
REDACTED_PATH = "<redacted-path>"

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEY = re.compile(
    r"""(?ix)
    ^(?:
        password|passwd|pwd|secret|credential|credentials|cookie|
        authorization|private[_-]?key|access[_-]?key|
        token|api[_-]?key|auth[_-]?token|access[_-]?token|
        refresh[_-]?token|session[_-]?token|hf[_-]?token|github[_-]?token
    )$
    """
)
_IDENTITY_KEY = re.compile(
    r"(?ix)^(?:uuid|gpu[_-]?uuid|hostname|host[_-]?name|nodename|node[_-]?name|fqdn|machine[_-]?name)$"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_KNOWN_TOKEN = re.compile(
    r"(?i)\b(?:github_pat_[A-Za-z0-9_]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|hf_[A-Za-z0-9]{12,}|AKIA[0-9A-Z]{16})\b"
)
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(password|passwd|pwd|secret|credential|cookie|authorization|
       private[_-]?key|access[_-]?key|api[_-]?key|token|
       auth[_-]?token|access[_-]?token|refresh[_-]?token|
       session[_-]?token|hf[_-]?token|github[_-]?token)
    (\s*[:=]\s*)
    (?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)
    """
)
_URL_CREDENTIAL = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")
_GPU_UUID = re.compile(r"(?i)\b(?:GPU|MIG)-[0-9a-f][0-9a-f-]{15,}\b")
_GENERIC_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
_HOST_ASSIGNMENT = re.compile(
    r"(?i)\b(host|hostname|host[_-]?name|nodename|node[_-]?name|fqdn|machine[_-]?name)(\s*[:=]\s*)(?![{\[])[^\s,;]+"
)
_IDENTITY_LINE = re.compile(
    r"""(?imx)
    (?P<prefix>
      (?<![A-Za-z0-9_.-])
      ["']?
      (?:
        (?:[A-Za-z0-9]+[_-])*
        (?:
          host|hostname|host[_-]?(?:name|id)|fqdn|
          node|nodename|node[_-]?(?:name|id)|
          worker|worker[_-]?(?:name|id)|
          server|server[_-]?(?:name|id)|
          instance|instance[_-]?(?:name|id)|
          machine[_-]?(?:name|id)|master[_-]?(?:addr|address)|
          ip|ip[_-]?address|ipv4|ipv6|mac|mac[_-]?address|
          serial|serial[_-]?number
        )
      )
      ["']?
      \s*[:=]\s*
    )
    (?P<value>"[^"\r\n]*"|'[^'\r\n]*'|[^\r\n,;}]+)
    """
)
_SECRET_LINE = re.compile(
    r"""(?imx)
    (?P<prefix>
      (?<![A-Za-z0-9_.-])
      ["']?
      (?:
        (?:[A-Za-z0-9]+[_-])*
        (?:
          password|passwd|pwd|credential|credentials|cookie|authorization|
          auth[_-]?(?:header|token)|
          private[_-]?key|api[_-]?key|access[_-]?key|
          access[_-]?token|refresh[_-]?token|session[_-]?token|
          hf[_-]?token|github[_-]?token|client[_-]?secret|
          secret(?:[_-]access[_-]key)?
        )
      )
      ["']?
      \s*[:=]\s*
    )
    (?P<value>"[^"\r\n]*"|'[^'\r\n]*'|[^\r\n,;}]+)
    """
)
_SHELL_PROMPT = re.compile(r"(?<![\w.-])(?:root|ubuntu|admin|user)@[A-Za-z0-9][A-Za-z0-9.-]*")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6 = re.compile(
    r"(?ix)(?<![0-9a-f:])(?:(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|[0-9a-f:]*::[0-9a-f:]*)(?![0-9a-f:])"
)
_MAC_ADDRESS = re.compile(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b")
_FILE_URI = re.compile(r"(?i)\bfile:///[^\s\"'<>|]+")
_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:\\|\\\\)[^\s\"'<>|]+")
_UNIX_PATH = re.compile(
    r"""(?x)
    (?<![A-Za-z0-9:/])
    /
    (?:[A-Za-z0-9._~@%+=,:-]+/)
    *[A-Za-z0-9._~@%+=,:-]+
    """
)
_HOME_PATH = re.compile(r"(?<![A-Za-z0-9])~(?:/[^\s\"'<>|]+)?")
_METRIC_NAME = re.compile(r"(?i)(?:metric|result|summary|eval|telemetry|pilot)")
_SAFE_RESOLVED_NAME = re.compile(
    r"(?:preprocess|teacher|student|eval|execution-provenance)(?:\.metadata)?\.(?:json|ya?ml|toml|txt)\Z"
)
_PUBLIC_NUMERIC_KEY = re.compile(
    r"""(?ix)
    (?:^|[^a-z0-9])
    (?:
      loss|accuracy|acc|perplexity|ppl|score|reward|energy|joules?|watts?|power|
      duration|elapsed|latency|throughput|utilization|temperature|memory|vram|
      tokens?|steps?|epochs?|iterations?|examples?|samples?|records?|rows?|
      count|rate|mean|average|minimum|maximum|min|max|median|percentile|std|
      variance|flops?|bytes?
    )
    (?:$|[^a-z0-9])
    """
)
_PUBLIC_LABEL_TOKENS = {
    "acc",
    "accuracy",
    "average",
    "bytes",
    "count",
    "cpu",
    "duration",
    "elapsed",
    "energy",
    "epoch",
    "epochs",
    "eval",
    "evaluation",
    "example",
    "examples",
    "flop",
    "flops",
    "gpu",
    "iteration",
    "iterations",
    "joule",
    "joules",
    "latency",
    "loss",
    "max",
    "maximum",
    "mean",
    "median",
    "memory",
    "min",
    "minimum",
    "percentile",
    "perplexity",
    "pilot",
    "power",
    "ppl",
    "preprocess",
    "rate",
    "record",
    "records",
    "reward",
    "row",
    "rows",
    "sample",
    "samples",
    "score",
    "seconds",
    "std",
    "step",
    "steps",
    "student",
    "teacher",
    "temperature",
    "test",
    "throughput",
    "token",
    "tokens",
    "total",
    "train",
    "training",
    "utilization",
    "validation",
    "variance",
    "vram",
    "watt",
    "watts",
}
_BANNED_EXTENSIONS = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}
_LARGE_SEQUENCE_KEYS = {
    "gpu_power_samples",
    "history",
    "power_samples",
    "predictions",
    "readings",
    "records",
    "samples",
    "timeline",
}
_LOG_METRIC_PATTERNS = (
    (
        "average_train_loss",
        re.compile(r"(?i)\baverage\s+train\s+loss\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"),
    ),
    (
        "eval_loss",
        re.compile(r"(?i)\beval(?:uation)?(?:/|\s+)?loss\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"),
    ),
    (
        "train_loss",
        re.compile(r"(?i)\btrain(?:/|\s+)?loss\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"),
    ),
    (
        "loss",
        re.compile(r"(?i)(?<![A-Za-z_])loss\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"),
    ),
    (
        "total_tokens",
        re.compile(r"(?i)\btotal\s+tokens(?:\s+processed)?\s*[:=]\s*([\d,]+)"),
    ),
    (
        "duration_hours",
        re.compile(r"(?i)\btraining\s+completed\s+in\s+(\d+(?:\.\d+)?)\s+hours?"),
    ),
)


class ExportError(RuntimeError):
    """Raised when a public export cannot be produced safely."""


@dataclass(frozen=True)
class SourceFile:
    relative: str
    path: Path
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class VerifiedRun:
    root: Path
    files: Mapping[str, SourceFile]
    manifest: Mapping[str, Any]
    plan: Mapping[str, Any]
    state: Mapping[str, Any]
    manifest_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source: SourceFile) -> tuple[int, int, int, int, int]:
    return (
        source.device,
        source.inode,
        source.size,
        source.mtime_ns,
        source.ctime_ns,
    )


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_source(source: SourceFile) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source.path, flags)
    except OSError as exc:
        raise ExportError(f"Cannot safely open source artifact {source.relative}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or _stat_identity(info) != _source_identity(source):
            raise ExportError(
                f"Source artifact changed after inventory scan: {source.relative}"
            )
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def _read_source_bytes(source: SourceFile, *, max_bytes: int | None = None) -> bytes:
    if max_bytes is not None and source.size > max_bytes:
        raise ExportError(
            f"Source artifact is too large to read safely: {source.relative} ({source.size} bytes)"
        )
    descriptor, before = _open_source(source)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ExportError(f"Cannot read source artifact {source.relative}: {exc}") from exc
    if _stat_identity(before) != _stat_identity(after) or len(payload) != source.size:
        raise ExportError(f"Source artifact changed while being read: {source.relative}")
    return payload


def _read_source_text(source: SourceFile, *, max_bytes: int | None = None) -> str:
    try:
        return _read_source_bytes(source, max_bytes=max_bytes).decode("utf-8")
    except UnicodeError as exc:
        raise ExportError(f"Source artifact is not valid UTF-8: {source.relative}") from exc


def _sha256_source(source: SourceFile) -> str:
    descriptor, before = _open_source(source)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ExportError(f"Cannot hash source artifact {source.relative}: {exc}") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise ExportError(f"Source artifact changed while being hashed: {source.relative}")
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def public_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path, *, max_bytes: int = MAX_METADATA_BYTES) -> Any:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise ExportError(f"Required JSON metadata is too large: {path.name} ({size} bytes)")
        text = path.read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ExportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExportError(f"Cannot read strict JSON from {path}: {exc}") from exc


def read_json_source(source: SourceFile, *, max_bytes: int = MAX_METADATA_BYTES) -> Any:
    try:
        return json.loads(
            _read_source_text(source, max_bytes=max_bytes),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ExportError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExportError(
            f"Cannot read strict JSON from source artifact {source.relative}: {exc}"
        ) from exc


def validate_relative_path(value: str, label: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ExportError(f"Unsafe {label}: {value!r}")
    if any(ord(character) < 32 for character in value):
        raise ExportError(f"Unsafe {label}: control character present")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ExportError(f"Unsafe {label}: {value!r}")
    normalized = candidate.as_posix()
    if normalized != value:
        raise ExportError(f"Non-canonical {label}: {value!r}")
    return normalized


def _scan_directory(root: Path) -> dict[str, SourceFile]:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ExportError(f"Cannot inspect run directory {root}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ExportError(f"Run path must be a real directory, not a link or special file: {root}")

    files: dict[str, SourceFile] = {}

    def visit(directory: Path, prefix: PurePosixPath | None = None) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ExportError(f"Cannot enumerate run directory {directory}: {exc}") from exc
        for entry in entries:
            name = entry.name
            validate_relative_path(name, "file name")
            relative_path = PurePosixPath(name) if prefix is None else prefix / name
            relative = validate_relative_path(relative_path.as_posix(), "run-relative path")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ExportError(f"Cannot inspect {relative}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ExportError(f"Run contains a symlink, which public export rejects: {relative}")
            if stat.S_ISDIR(info.st_mode):
                visit(Path(entry.path), relative_path)
            elif stat.S_ISREG(info.st_mode):
                files[relative] = SourceFile(
                    relative,
                    Path(entry.path),
                    info.st_size,
                    info.st_dev,
                    info.st_ino,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            else:
                raise ExportError(f"Run contains a special file, which public export rejects: {relative}")

    visit(root)
    return files


def _artifact_inventory_paths(files: Mapping[str, SourceFile]) -> set[str]:
    return {
        relative
        for relative in files
        if relative not in {"manifest.json", "state.json"}
        and not relative.startswith("sync/receipts/")
    }


def _acquire_source_lock(run_dir: Path | str) -> Any:
    root = Path(run_dir).expanduser().resolve(strict=False)
    lock_path = root / ".stage.lock"
    if not lock_path.exists():
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
        stream = os.fdopen(descriptor, "rb")
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        return stream
    except BlockingIOError as exc:
        try:
            stream.close()
        except UnboundLocalError:
            os.close(descriptor)
        raise ExportError("Source run is still locked by an active stage") from exc
    except OSError as exc:
        try:
            os.close(descriptor)
        except UnboundLocalError:
            pass
        raise ExportError(f"Cannot safely lock source run: {exc}") from exc


def verify_source_run(run_dir: Path | str) -> VerifiedRun:
    root = Path(run_dir).expanduser()
    if root.is_symlink():
        raise ExportError(f"Run path must not be a symlink: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ExportError(f"Run directory does not exist: {root}") from exc
    if not _SAFE_RUN_ID.fullmatch(root.name):
        raise ExportError(f"Unsafe run directory name: {root.name!r}")

    files = _scan_directory(root)
    for required in ("manifest.json", "plan.json", "state.json"):
        if required not in files:
            raise ExportError(f"Run is missing required metadata: {required}")

    manifest_value = read_json_source(files["manifest.json"])
    plan_value = read_json_source(files["plan.json"])
    state_value = read_json_source(files["state.json"])
    if not isinstance(manifest_value, dict) or manifest_value.get("schema") != "energy-repro/manifest/v1":
        raise ExportError("Source manifest has an unsupported or missing schema")
    if not isinstance(plan_value, dict) or plan_value.get("schema") != "energy-repro/plan/v1":
        raise ExportError("Source plan has an unsupported or missing schema")
    if not isinstance(state_value, dict) or state_value.get("schema") != "energy-repro/run-state/v1":
        raise ExportError("Source state has an unsupported or missing schema")
    if manifest_value.get("run_id") != root.name or plan_value.get("run_id") != root.name:
        raise ExportError("Run ID is not bound to the source directory name")
    if manifest_value.get("plan") != plan_value:
        raise ExportError("manifest.json and plan.json disagree")

    source_state = manifest_value.get("state")
    if source_state not in TERMINAL_STATES:
        raise ExportError(
            f"Only terminal runs may be exported; observed state {source_state!r}"
        )
    if state_value.get("state") != source_state:
        raise ExportError("manifest.json and state.json disagree about run state")
    if state_value.get("exit_code") != manifest_value.get("exit_code"):
        raise ExportError("manifest.json and state.json disagree about exit code")

    plan_payload = dict(plan_value)
    claimed_plan_hash = plan_payload.pop("plan_sha256", None)
    plan_payload.pop("run_id", None)
    if not isinstance(claimed_plan_hash, str) or not _HEX_SHA256.fullmatch(claimed_plan_hash):
        raise ExportError("Source plan is missing a valid plan_sha256")
    if sha256_bytes(canonical_bytes(plan_payload)) != claimed_plan_hash:
        raise ExportError("Source plan_sha256 does not match canonical plan content")

    artifacts = manifest_value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExportError("Source manifest artifacts must be a list")
    expected_paths: set[str] = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or item.get("type") != "file":
            raise ExportError(f"Source manifest artifact {index} is not a regular file")
        relative = validate_relative_path(item.get("path"), f"artifact path at index {index}")
        if relative in expected_paths:
            raise ExportError(f"Duplicate source artifact path: {relative}")
        expected_paths.add(relative)
        source = files.get(relative)
        if source is None:
            raise ExportError(f"Source artifact is missing: {relative}")
        if isinstance(item.get("bytes"), bool) or not isinstance(item.get("bytes"), int):
            raise ExportError(f"Source artifact has an invalid byte count: {relative}")
        if item["bytes"] != source.size:
            raise ExportError(f"Source artifact size mismatch: {relative}")
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not _HEX_SHA256.fullmatch(expected_hash):
            raise ExportError(f"Source artifact is missing a valid SHA-256: {relative}")
        if _sha256_source(source) != expected_hash:
            raise ExportError(f"Source artifact SHA-256 mismatch: {relative}")

    actual_paths = _artifact_inventory_paths(files)
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        unrecorded = sorted(actual_paths - expected_paths)
        detail = []
        if missing:
            detail.append(f"missing={missing[:10]!r}")
        if unrecorded:
            detail.append(f"unrecorded={unrecorded[:10]!r}")
        raise ExportError("Source manifest inventory is not exact: " + ", ".join(detail))

    return VerifiedRun(
        root=root,
        files=files,
        manifest=manifest_value,
        plan=plan_value,
        state=state_value,
        manifest_sha256=_sha256_source(files["manifest.json"]),
    )


def _is_absolute_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("~/")
        or value == "~"
        or bool(re.match(r"(?i)^(?:[A-Z]:\\|\\\\)", value))
        or value.startswith("file://")
    )


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_sensitive_key(value: str) -> bool:
    normalized = _normalized_key(value)
    if _SENSITIVE_KEY.fullmatch(normalized):
        return True
    tokens = [item for item in normalized.split("_") if item]
    if any(
        item
        in {
            "password",
            "passwd",
            "pwd",
            "secret",
            "credential",
            "credentials",
            "cookie",
            "authorization",
            "auth",
            "token",
        }
        for item in tokens
    ):
        return True
    pairs = set(zip(tokens, tokens[1:]))
    return bool(
        pairs
        & {
            ("api", "key"),
            ("access", "key"),
            ("private", "key"),
            ("client", "secret"),
            ("auth", "header"),
            ("auth", "token"),
        }
    )


def _is_identity_key(value: str) -> bool:
    normalized = _normalized_key(value)
    if _IDENTITY_KEY.fullmatch(normalized):
        return True
    return bool(
        re.search(
            r"""(?ix)
            (?:
              ^|_
            )
            (?:
              host|hostname|host_name|host_id|fqdn|node|nodename|node_name|node_id|
              worker|worker_name|worker_id|server|server_name|server_id|
              instance|instance_name|instance_id|
              machine_name|machine_id|master_addr|master_address|
              ip|ip_address|ipv4|ipv6|mac|mac_address|serial|serial_number
            )
            $
            """,
            normalized,
        )
    )


def sanitize_text(value: str) -> str:
    value = _PRIVATE_KEY.sub(REDACTED, value)
    value = _URL_CREDENTIAL.sub(r"\1<redacted>@", value)
    value = _BEARER.sub("Bearer <redacted>", value)
    value = _KNOWN_TOKEN.sub(REDACTED, value)
    value = _GPU_UUID.sub(REDACTED, value)
    value = _GENERIC_UUID.sub(REDACTED, value)
    value = _SECRET_LINE.sub(lambda match: f"{match.group('prefix')}{REDACTED}", value)
    value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", value)
    value = _IDENTITY_LINE.sub(lambda match: f"{match.group('prefix')}{REDACTED}", value)
    value = _HOST_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", value)
    value = _SHELL_PROMPT.sub(REDACTED, value)
    value = _MAC_ADDRESS.sub(REDACTED, value)
    value = _IPV4.sub(REDACTED, value)
    value = _IPV6.sub(REDACTED, value)
    value = _FILE_URI.sub(REDACTED_PATH, value)
    value = _WINDOWS_PATH.sub(REDACTED_PATH, value)
    value = _HOME_PATH.sub(REDACTED_PATH, value)
    value = _UNIX_PATH.sub(REDACTED_PATH, value)
    return value


def _summarize_sequence(value: Sequence[Any]) -> dict[str, Any]:
    numeric = [
        float(item)
        for item in value
        if isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
    ]
    summary: dict[str, Any] = {"count": len(value)}
    if numeric and len(numeric) == len(value):
        summary.update(
            {
                "minimum": min(numeric),
                "maximum": max(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        )
    summary["source_values_sha256"] = sha256_bytes(canonical_bytes(sanitize_data(list(value))))
    return summary


def sanitize_data(value: Any, *, key: str | None = None, compact: bool = False) -> Any:
    if key is not None and (_is_sensitive_key(key) or _is_identity_key(key)):
        return REDACTED
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for item_key in sorted(value, key=lambda candidate: str(candidate)):
            text_key = str(item_key)
            public_key = sanitize_text(text_key)
            if public_key != text_key:
                public_key = f"<redacted-key:{sha256_bytes(text_key.encode('utf-8'))[:16]}>"
            if public_key in result:
                raise ExportError("Sanitized metadata contains colliding public keys")
            result[public_key] = sanitize_data(
                value[item_key],
                key=text_key,
                compact=compact,
            )
        return result
    if isinstance(value, (list, tuple)):
        if compact and (len(value) > 24 or (key or "").lower() in _LARGE_SEQUENCE_KEYS):
            return _summarize_sequence(value)
        return [sanitize_data(item, compact=compact) for item in value]
    if isinstance(value, str):
        if _is_absolute_path(value):
            return REDACTED_PATH
        return sanitize_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "<non-finite>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return sanitize_text(str(value))


def _public_oci_reference(value: str) -> str:
    digest = re.search(r"(?i)(?:@)?sha256:([0-9a-f]{64})", value)
    if digest:
        return f"<redacted-oci-reference>@sha256:{digest.group(1).lower()}"
    return "<redacted-oci-reference>"


def _known_image_replacements(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    candidates: list[Any] = [plan.get("image")]
    verification = manifest.get("image_verification")
    if isinstance(verification, Mapping):
        candidates.append(verification.get("ref"))
        repo_digests = verification.get("repo_digests")
        if isinstance(repo_digests, list):
            candidates.extend(repo_digests)
    return {
        value: _public_oci_reference(value)
        for value in candidates
        if isinstance(value, str) and value
    }


def _replace_known_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _replace_known_strings(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_known_strings(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_replace_known_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for source in sorted(replacements, key=len, reverse=True):
            value = value.replace(source, replacements[source])
    return value


def _public_plan_and_manifest(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replacements = _known_image_replacements(plan, manifest)
    source_plan = _replace_known_strings(dict(plan), replacements)
    source_manifest = dict(manifest)
    source_manifest.pop("artifacts", None)
    failure = source_manifest.get("failure")
    if isinstance(failure, Mapping):
        public_failure = dict(failure)
        if "detail" in public_failure:
            public_failure["detail"] = "<redacted-failure-detail>"
        source_manifest["failure"] = public_failure
    source_manifest = _replace_known_strings(source_manifest, replacements)
    return (
        sanitize_data(source_plan, compact=True),
        sanitize_data(source_manifest, compact=True),
    )


def _banned_reason(relative: str, size: int, max_file_bytes: int) -> str | None:
    if relative.startswith("sync/receipts/"):
        return "unmanifested-sync-receipt"
    if sanitize_text(relative) != relative:
        return "sensitive-file-name"
    path = PurePosixPath(relative)
    lowered = [part.lower() for part in path.parts]
    for part in lowered:
        normalized = part.replace("-", "_")
        if (
            normalized == "wandb"
            or normalized in {"assets", "asset", "datasets", "dataset", "raw", "raw_data"}
            or normalized in {"final_model", "final_models"}
            or normalized == "checkpoints"
            or normalized.startswith("checkpoint_")
        ):
            return "prohibited-content"
    if path.suffix.lower() in _BANNED_EXTENSIONS:
        return "model-or-binary-artifact"
    if size > max_file_bytes:
        return "over-size-cap"
    return None


def _write_file(root: Path, relative: str, payload: bytes) -> None:
    validate_relative_path(relative, "export-relative path")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.chmod(0o644)


def _compact_json_artifact(source: SourceFile) -> Any:
    return sanitize_data(read_json_source(source), compact=True)


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    materialized = [value for value in values if math.isfinite(value)]
    if not materialized:
        return {"count": 0}
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
        "mean": sum(materialized) / len(materialized),
    }


def _is_public_numeric_key(value: str) -> bool:
    return bool(
        value
        and len(value) <= 128
        and sanitize_text(value) == value
        and not _is_sensitive_key(value)
        and not _is_identity_key(value)
        and _PUBLIC_NUMERIC_KEY.search(value)
    )


def _public_numeric_label(path: Sequence[str]) -> str | None:
    selected: list[str] = []
    for segment in path:
        for token in _normalized_key(segment).split("_"):
            if token in _PUBLIC_LABEL_TOKENS:
                selected.append(token)
    if not selected:
        return None
    label = "_".join(selected)
    return label[:128]


def _collect_public_numeric(
    value: Any,
    path: tuple[str, ...],
    output: dict[str, list[float]],
) -> None:
    if isinstance(value, Mapping):
        for raw_key in sorted(value, key=lambda candidate: str(candidate)):
            key = str(raw_key)
            if (
                len(path) >= 12
                or len(key) > 128
                or sanitize_text(key) != key
                or _is_sensitive_key(key)
                or _is_identity_key(key)
            ):
                continue
            _collect_public_numeric(value[raw_key], (*path, key), output)
        return
    if isinstance(value, (list, tuple)):
        if path and _is_public_numeric_key(path[-1]):
            label = _public_numeric_label(path)
            numeric = [
                float(item)
                for item in value
                if isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            ]
            if label is not None and len(numeric) == len(value):
                output[label].extend(numeric)
                return
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                _collect_public_numeric(item, path, output)
        return
    if (
        path
        and _is_public_numeric_key(path[-1])
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        label = _public_numeric_label(path)
        if label is not None:
            output[label].append(float(value))


def _numeric_only_json_summary(value: Any) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    _collect_public_numeric(value, (), numeric)
    return {
        "content_policy": "numeric allowlist only; strings and arbitrary payloads excluded",
        "numeric_fields": {
            key: _numeric_summary(values) for key, values in sorted(numeric.items())
        },
    }


def _summarize_json(source: SourceFile) -> dict[str, Any]:
    return _numeric_only_json_summary(read_json_source(source))


def _summarize_jsonl(source: SourceFile) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    record_count = 0
    text = _read_source_text(source)
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ExportError(
                f"Invalid JSONL at {source.relative}:{line_number}: {exc}"
            ) from exc
        record_count += 1
        _collect_public_numeric(value, (), numeric)
    return {
        "record_count": record_count,
        "content_policy": "numeric allowlist only; strings and arbitrary payloads excluded",
        "numeric_fields": {
            key: _numeric_summary(values) for key, values in sorted(numeric.items())
        },
    }


def _summarize_csv(source: SourceFile) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    row_count = 0
    try:
        with io.StringIO(_read_source_text(source), newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            for row in reader:
                row_count += 1
                for field in fields:
                    if not _is_public_numeric_key(field):
                        continue
                    raw = row.get(field)
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        numeric[field].append(value)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExportError(f"Cannot summarize CSV {source.relative}: {exc}") from exc
    return {
        "row_count": row_count,
        "column_count": len(fields),
        "content_policy": "numeric allowlist only; strings and arbitrary payloads excluded",
        "numeric_fields": {
            key: _numeric_summary(values)
            for key, values in sorted(numeric.items())
        },
    }


def _summarize_structured(source: SourceFile) -> Any:
    suffix = source.path.suffix.lower()
    if suffix == ".json":
        return _summarize_json(source)
    if suffix == ".jsonl":
        return _summarize_jsonl(source)
    if suffix == ".csv":
        return _summarize_csv(source)
    raise ExportError(f"Unsupported structured summary format: {source.relative}")


def _summarize_logs(
    sources: Sequence[SourceFile],
    excluded: list[tuple[str, str]],
) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    for index, source in enumerate(
        sorted(sources, key=lambda item: item.relative),
        1,
    ):
        try:
            text = _read_source_text(source)
        except ExportError:
            excluded.append((source.relative, "non-text-log"))
            continue
        lines = text.splitlines()
        metrics: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            for name, pattern in _LOG_METRIC_PATTERNS:
                for match in pattern.finditer(line):
                    raw = match.group(1).replace(",", "")
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        metrics.append(
                            {
                                "line_number": line_number,
                                "name": name,
                                "value": value,
                            }
                        )
                    if len(metrics) >= 200:
                        break
                if len(metrics) >= 200:
                    break
            if len(metrics) >= 200:
                break
        logs[f"log-{index:03d}"] = {
            "bytes": source.size,
            "line_count": len(lines),
            "source_sha256": _sha256_source(source),
            "numeric_metrics": metrics,
            "metrics_truncated": len(metrics) >= 200,
            "content_policy": "raw log text excluded",
        }
    return {"schema": f"{EXPORT_SCHEMA}/logs", "logs": logs}


def _tree_inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    files = _scan_directory(root)
    return [
        {
            "path": relative,
            "bytes": source.size,
            "sha256": _sha256_source(source),
        }
        for relative, source in sorted(files.items())
        if relative not in excluded
    ]


def _tree_hash(inventory: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes(list(inventory)))


def _safe_output_paths(
    source_root: Path,
    output_dir: Path | str,
    archive_path: Path | str | None,
) -> tuple[Path, Path | None]:
    output = Path(output_dir).expanduser().resolve(strict=False)
    archive = Path(archive_path).expanduser().resolve(strict=False) if archive_path else None
    if output == Path(output.anchor):
        raise ExportError("Export directory must not be a filesystem root")
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ExportError("Export directory must not be inside the source run")
    if output.exists() or output.is_symlink():
        raise ExportError(f"Export destination already exists: {output}")
    if archive is not None:
        if archive == Path(archive.anchor):
            raise ExportError("Archive path must not be a filesystem root")
        try:
            archive.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ExportError("Archive path must not be inside the source run")
        if archive.exists() or archive.is_symlink():
            raise ExportError(f"Archive destination already exists: {archive}")
        if archive == output:
            raise ExportError("Archive and export directory must be different paths")
        for child, parent in ((archive, output), (output, archive)):
            try:
                child.relative_to(parent)
            except ValueError:
                continue
            raise ExportError(
                "Archive and export paths must not contain one another"
            )
    return output, archive


def _path_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


def _reserve_directory(path: Path) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ExportError(f"Export destination already exists: {path}") from exc
    return _path_identity(path)


def _reserve_file(path: Path) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ExportError(f"Archive destination already exists: {path}") from exc
    os.close(descriptor)
    return _path_identity(path)


def _require_reservation(
    path: Path,
    identity: tuple[int, int, int],
    *,
    empty_directory: bool = False,
) -> None:
    try:
        current = _path_identity(path)
    except OSError as exc:
        raise ExportError(f"Reserved destination disappeared: {path}") from exc
    if current != identity:
        raise ExportError(f"Reserved destination was replaced concurrently: {path}")
    if empty_directory:
        try:
            if any(path.iterdir()):
                raise ExportError(
                    f"Reserved export destination was modified concurrently: {path}"
                )
        except OSError as exc:
            raise ExportError(f"Cannot inspect reserved destination {path}: {exc}") from exc


def _remove_owned_path(
    path: Path,
    identities: set[tuple[int, int, int]],
    *,
    directory: bool,
) -> None:
    try:
        if _path_identity(path) not in identities:
            return
        if directory:
            path.rmdir()
        else:
            path.unlink()
    except (FileNotFoundError, OSError):
        return


def _add_tar_entry(archive: tarfile.TarFile, name: str, payload: bytes | None) -> None:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if payload is None:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
        archive.addfile(info)
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def create_deterministic_archive(
    export_root: Path,
    destination: Path,
    *,
    reservation: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    files = _scan_directory(export_root)
    directories: set[str] = {"result"}
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(f"result/{parent.as_posix()}")
            parent = parent.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for directory in sorted(directories):
                        _add_tar_entry(archive, directory, None)
                    for relative, source in sorted(files.items()):
                        _add_tar_entry(
                            archive,
                            f"result/{relative}",
                            _read_source_bytes(source),
                        )
        os.chmod(temporary, 0o644)
        if reservation is not None:
            _require_reservation(destination, reservation)
        os.replace(temporary, destination)
        return _path_identity(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_public_text(relative: str, payload: str) -> None:
    forbidden = [
        ("private key", _PRIVATE_KEY),
        ("bearer credential", _BEARER),
        ("known token", _KNOWN_TOKEN),
        ("GPU UUID", _GPU_UUID),
        ("UUID", _GENERIC_UUID),
        ("URL credential", _URL_CREDENTIAL),
        ("shell hostname", _SHELL_PROMPT),
        ("IPv4 address", _IPV4),
        ("IPv6 address", _IPV6),
        ("MAC address", _MAC_ADDRESS),
        ("file URI", _FILE_URI),
        ("Windows absolute path", _WINDOWS_PATH),
        ("home path", _HOME_PATH),
        ("Unix absolute path", _UNIX_PATH),
    ]
    for label, pattern in forbidden:
        if pattern.search(payload):
            raise ExportError(f"Sanitization audit found {label} in {relative}")
    for label, pattern in (
        ("secret line", _SECRET_LINE),
        ("secret assignment", _SECRET_ASSIGNMENT),
        ("identity line", _IDENTITY_LINE),
        ("host assignment", _HOST_ASSIGNMENT),
    ):
        for match in pattern.finditer(payload):
            if REDACTED not in match.group(0):
                raise ExportError(f"Sanitization audit found {label} in {relative}")


def verify_export(export_dir: Path | str) -> dict[str, Any]:
    root = Path(export_dir).expanduser()
    if root.is_symlink():
        raise ExportError(f"Export path must not be a symlink: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ExportError(f"Export directory does not exist: {root}") from exc
    files = _scan_directory(root)
    for required in ("export-receipt.json", "SHA256SUMS"):
        if required not in files:
            raise ExportError(f"Export is missing {required}")

    try:
        checksum_text = _read_source_text(files["SHA256SUMS"])
    except ExportError as exc:
        raise ExportError(f"Cannot read SHA256SUMS: {exc}") from exc
    claims: dict[str, str] = {}
    for line_number, line in enumerate(checksum_text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ExportError(f"Malformed SHA256SUMS line {line_number}")
        relative = validate_relative_path(match.group(2), "checksum path")
        if relative == "SHA256SUMS" or relative in claims:
            raise ExportError(f"Invalid or duplicate SHA256SUMS entry: {relative}")
        claims[relative] = match.group(1)
    actual_paths = set(files) - {"SHA256SUMS"}
    if set(claims) != actual_paths:
        raise ExportError("SHA256SUMS does not exactly cover the public export")
    for relative, expected in sorted(claims.items()):
        if _sha256_source(files[relative]) != expected:
            raise ExportError(f"Public export SHA-256 mismatch: {relative}")

    receipt = read_json_source(files["export-receipt.json"])
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ExportError("Export receipt has an unsupported or missing schema")
    source_hash = receipt.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or not _HEX_SHA256.fullmatch(source_hash):
        raise ExportError("Export receipt has an invalid source manifest hash")
    expected_payload = _tree_inventory(
        root, exclude={"SHA256SUMS", "export-receipt.json"}
    )
    if receipt.get("payload_files") != expected_payload:
        raise ExportError("Export receipt payload inventory does not match the directory")
    if receipt.get("payload_tree_sha256") != _tree_hash(expected_payload):
        raise ExportError("Export receipt payload tree hash is invalid")

    for relative, source in sorted(files.items()):
        try:
            text = _read_source_text(source)
        except ExportError as exc:
            raise ExportError(f"Public export contains a non-text file: {relative}") from exc
        _verify_public_text(relative, text)
    return {
        "status": "PASS",
        "schema": EXPORT_SCHEMA,
        "run_id": receipt.get("run_id"),
        "source_manifest_sha256": source_hash,
        "file_count": len(files),
    }


def export_results(
    run_dir: Path | str,
    output_dir: Path | str,
    *,
    archive_path: Path | str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes < 1:
        raise ExportError("max_file_bytes must be a positive integer")
    source_lock = _acquire_source_lock(run_dir)
    try:
        verified = verify_source_run(run_dir)
        output, archive = _safe_output_paths(verified.root, output_dir, archive_path)
    except Exception:
        if source_lock is not None:
            source_lock.close()
        raise
    output_reservation: tuple[int, int, int] | None = None
    archive_identities: set[tuple[int, int, int]] = set()
    temporary: Path | None = None
    output_published = False
    excluded: list[tuple[str, str]] = []
    try:
        output_reservation = _reserve_directory(output)
        if archive is not None:
            archive_identities.add(_reserve_file(archive))
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
        )
        reason_by_path = {
            relative: _banned_reason(relative, source.size, max_file_bytes)
            for relative, source in verified.files.items()
        }
        public_plan, public_manifest = _public_plan_and_manifest(
            verified.plan,
            verified.manifest,
        )
        # Source-relative names can themselves contain host or dataset identity.
        # Public summaries carry source content hashes, so the raw artifact list
        # is intentionally omitted rather than trying to guess which names are safe.
        public_manifest["artifacts"] = []
        public_manifest["public_export"] = {
            "source_artifact_count": len(verified.manifest.get("artifacts", [])),
            "published_artifact_metadata_count": 0,
            "source_manifest_sha256": verified.manifest_sha256,
        }
        _write_file(temporary, "plan.json", public_json_bytes(public_plan))
        _write_file(temporary, "state.json", public_json_bytes(sanitize_data(verified.state, compact=True)))
        _write_file(temporary, "manifest.json", public_json_bytes(public_manifest))

        resolved_extensions = {".json", ".yaml", ".yml", ".toml", ".txt"}
        for relative, source in sorted(verified.files.items()):
            if not relative.startswith("resolved/"):
                continue
            reason = reason_by_path[relative]
            if reason is not None:
                excluded.append((relative, reason))
                continue
            if not _SAFE_RESOLVED_NAME.fullmatch(PurePosixPath(relative).name):
                excluded.append((relative, "unsafe-resolved-name"))
                continue
            if source.path.suffix.lower() not in resolved_extensions:
                excluded.append((relative, "unsupported-resolved-format"))
                continue
            if source.path.suffix.lower() == ".json":
                payload = public_json_bytes(_compact_json_artifact(source))
            else:
                try:
                    payload = sanitize_text(_read_source_text(source)).encode("utf-8")
                except ExportError:
                    excluded.append((relative, "non-text-resolved-config"))
                    continue
            _write_file(temporary, relative, payload)

        energy_sources: dict[str, Any] = {}
        energy_candidates = [
            (relative, source)
            for relative, source in sorted(verified.files.items())
            if relative.startswith("energy/")
        ]
        for index, (relative, source) in enumerate(energy_candidates, 1):
            reason = reason_by_path[relative]
            if reason is not None:
                excluded.append((relative, reason))
                continue
            if source.path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                excluded.append((relative, "unsupported-energy-format"))
                continue
            energy_sources[f"energy-{index:03d}"] = {
                "bytes": source.size,
                "source_sha256": _sha256_source(source),
                "summary": _summarize_structured(source),
            }
        _write_file(
            temporary,
            "energy-summary.json",
            public_json_bytes(
                {"schema": f"{EXPORT_SCHEMA}/energy", "sources": energy_sources}
            ),
        )

        metric_sources: dict[str, Any] = {}
        metric_candidates = [
            (relative, source)
            for relative, source in sorted(verified.files.items())
            if not relative.startswith(("resolved/", "energy/", "logs/", "wandb/"))
            and relative not in {"manifest.json", "plan.json", "state.json"}
            and _METRIC_NAME.search(source.path.name)
        ]
        for index, (relative, source) in enumerate(metric_candidates, 1):
            reason = reason_by_path[relative]
            if reason is not None:
                excluded.append((relative, reason))
                continue
            if source.path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                excluded.append((relative, "unsupported-metric-format"))
                continue
            metric_sources[f"metric-{index:03d}"] = {
                "bytes": source.size,
                "source_sha256": _sha256_source(source),
                "summary": _summarize_structured(source),
            }
        _write_file(
            temporary,
            "metrics-summary.json",
            public_json_bytes(
                {"schema": f"{EXPORT_SCHEMA}/metrics", "sources": metric_sources}
            ),
        )

        log_sources: list[SourceFile] = []
        for relative, source in sorted(verified.files.items()):
            if not relative.startswith("logs/"):
                continue
            reason = reason_by_path[relative]
            if reason is not None:
                excluded.append((relative, reason))
                continue
            log_sources.append(source)
        _write_file(
            temporary,
            "logs-summary.json",
            public_json_bytes(_summarize_logs(log_sources, excluded)),
        )

        explicitly_handled = {
            "manifest.json",
            "plan.json",
            "state.json",
        }
        for relative, source in sorted(verified.files.items()):
            if relative in explicitly_handled or relative.startswith(
                ("resolved/", "energy/", "logs/")
            ):
                continue
            reason = reason_by_path[relative]
            if reason is not None and (relative, reason) not in excluded:
                excluded.append((relative, reason))

        counts = Counter(reason for _, reason in excluded)
        excluded_examples = [
            {
                "path_sha256": sha256_bytes(relative.encode("utf-8")),
                "reason": reason,
            }
            for relative, reason in sorted(set(excluded))[:200]
        ]
        payload_inventory = _tree_inventory(temporary)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "run_id": verified.manifest["run_id"],
            "source_state": verified.manifest["state"],
            "source_exit_code": verified.manifest.get("exit_code"),
            "source_manifest_sha256": verified.manifest_sha256,
            "source_manifest_bytes": verified.files["manifest.json"].size,
            "source_verification": "full-artifact-sha256-and-exact-inventory",
            "policy": {
                "max_source_file_bytes": max_file_bytes,
                "terminal_states": sorted(TERMINAL_STATES),
                "redaction": "keys, credentials, host identity, GPU UUIDs, and absolute paths",
            },
            "excluded": {
                "count": len(set(excluded)),
                "counts_by_reason": dict(sorted(counts.items())),
                "examples": excluded_examples,
                "examples_truncated": len(set(excluded)) > len(excluded_examples),
            },
            "payload_files": payload_inventory,
            "payload_tree_sha256": _tree_hash(payload_inventory),
        }
        _write_file(temporary, "export-receipt.json", public_json_bytes(receipt))

        checksummed = _tree_inventory(temporary)
        checksum_payload = "".join(
            f"{item['sha256']}  {item['path']}\n" for item in checksummed
        ).encode("utf-8")
        _write_file(temporary, "SHA256SUMS", checksum_payload)
        verify_export(temporary)

        if archive is not None:
            archive_reservation = next(iter(archive_identities))
            archive_identities.add(
                create_deterministic_archive(
                    temporary,
                    archive,
                    reservation=archive_reservation,
                )
            )
        if output_reservation is None:
            raise ExportError("Export destination reservation is missing")
        _require_reservation(output, output_reservation, empty_directory=True)
        os.replace(temporary, output)
        output_published = True
    except Exception:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if archive is not None:
            _remove_owned_path(archive, archive_identities, directory=False)
        if output_reservation is not None and not output_published:
            _remove_owned_path(
                output,
                {output_reservation},
                directory=True,
            )
        if source_lock is not None:
            source_lock.close()
        raise

    if source_lock is not None:
        source_lock.close()
    result = verify_export(output)
    result.update(
        {
            "output": str(output),
            "archive": str(archive) if archive is not None else None,
            "archive_sha256": sha256_file(archive) if archive is not None else None,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Maximum source file size eligible for public content (default: 1 MiB)",
    )
    parser.add_argument(
        "--verify-export",
        type=Path,
        help="Verify an existing public result directory instead of exporting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_export is not None:
            if args.run_dir is not None or args.output is not None or args.archive is not None:
                parser.error("--verify-export cannot be combined with export arguments")
            result = verify_export(args.verify_export)
        else:
            if args.run_dir is None or args.output is None:
                parser.error("run_dir and --output are required for export")
            result = export_results(
                args.run_dir,
                args.output,
                archive_path=args.archive,
                max_file_bytes=args.max_file_bytes,
            )
    except ExportError as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
