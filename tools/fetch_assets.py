#!/usr/bin/env python3
"""Fetch immutable Hugging Face assets into an atomic content-addressed cache."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LOCK_SCHEMA = "energy-repro/assets-lock/v1"
RECEIPT_SCHEMA = "energy-repro/assets-receipt/v1"
OBJECT_SCHEMA = "energy-repro/asset-object/v1"
MARKER_NAME = ".energy-asset.json"
TREE_HASH_ALGORITHM = "sha256-jsonl-posix-path-size-content-sha256-v1"
HASH_CHUNK_BYTES = 1024 * 1024
FULL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_LIKE_RE = re.compile(
    r"(?i)(?:bearer\s+)?hf_[a-z0-9]{8,}|(?:token|authorization)=\S+"
)


class AssetLockError(ValueError):
    """Raised when the asset lock is malformed or mutable."""


class AssetVerificationError(RuntimeError):
    """Raised when a published cache object is incomplete or mismatched."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = TOKEN_LIKE_RE.sub("<redacted>", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text[:800]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_string_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AssetLockError(f"{field} must be a list of non-empty strings")


def load_asset_lock(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise AssetLockError("asset lock root must be a JSON object")
    if data.get("schema") != LOCK_SCHEMA:
        raise AssetLockError(f"asset lock schema must be {LOCK_SCHEMA}")

    assets = data.get("assets")
    sets = data.get("sets")
    if not isinstance(assets, dict):
        raise AssetLockError("assets must be a JSON object")
    if not isinstance(sets, dict):
        raise AssetLockError("sets must be a JSON object")

    for name, asset in assets.items():
        if not isinstance(name, str) or not name:
            raise AssetLockError("asset names must be non-empty strings")
        if not isinstance(asset, dict):
            raise AssetLockError(f"asset {name!r} must be a JSON object")
        if asset.get("kind") not in {"model", "dataset"}:
            raise AssetLockError(f"asset {name!r} kind must be model or dataset")
        repository = asset.get("repository")
        if not isinstance(repository, str) or not repository:
            raise AssetLockError(f"asset {name!r} repository must be non-empty")
        revision = asset.get("revision")
        if not isinstance(revision, str) or not FULL_REVISION_RE.fullmatch(revision):
            raise AssetLockError(
                f"asset {name!r} revision must be a full lowercase 40-character SHA"
            )
        for field in ("allow_patterns", "ignore_patterns"):
            if field in asset:
                _validate_string_list(asset[field], f"asset {name!r} {field}")
        expected_bytes = asset.get("expected_bytes")
        if expected_bytes is not None and (
            not isinstance(expected_bytes, int) or expected_bytes < 0
        ):
            raise AssetLockError(
                f"asset {name!r} expected_bytes must be null or a non-negative integer"
            )

    for set_name, members in sets.items():
        if not isinstance(set_name, str) or not set_name:
            raise AssetLockError("set names must be non-empty strings")
        _validate_string_list(members, f"set {set_name!r}")
        unknown = [member for member in members if member not in assets]
        if unknown:
            raise AssetLockError(
                f"set {set_name!r} references unknown assets: {', '.join(unknown)}"
            )
    return data


def resolve_sets(lock: Mapping[str, Any], set_names: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    sets = lock["sets"]
    for set_name in set_names:
        if set_name not in sets:
            raise AssetLockError(f"unknown asset set: {set_name}")
        for asset_name in sets[set_name]:
            if asset_name not in seen:
                resolved.append(asset_name)
                seen.add(asset_name)
    return resolved


def _descriptor(asset: Mapping[str, Any]) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "kind": asset["kind"],
        "repository": asset["repository"],
        "revision": asset["revision"],
    }
    for field in ("allow_patterns", "ignore_patterns"):
        if field in asset:
            descriptor[field] = sorted(asset[field])
    return descriptor


def asset_cache_key(asset: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _descriptor(asset),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total_bytes += len(chunk)
    return digest.hexdigest(), total_bytes


def _scan_payload(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise AssetVerificationError(f"cache object is not a directory: {root}")

    payload_files: list[tuple[str, Path]] = []
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in dir_names:
            path = current_path / directory
            if path.is_symlink():
                raise AssetVerificationError(f"symlinked directory is not allowed: {path}")
        for filename in file_names:
            path = current_path / filename
            if path.name == MARKER_NAME and path.parent == root:
                continue
            if path.is_symlink():
                raise AssetVerificationError(f"symlinked file is not allowed: {path}")
            if not path.is_file():
                raise AssetVerificationError(f"special file is not allowed: {path}")
            payload_files.append((path.relative_to(root).as_posix(), path))

    if not payload_files:
        raise AssetVerificationError(f"cache object contains no payload files: {root}")

    tree_digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, path in sorted(payload_files, key=lambda item: item[0]):
        content_sha256, size = _sha256_file(path)
        total_bytes += size
        record = {
            "path": relative_path,
            "sha256": content_sha256,
            "size": size,
        }
        tree_digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        tree_digest.update(b"\n")

    return {
        "file_count": len(payload_files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
    }


def _write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def verify_object(
    object_path: Path,
    asset: Mapping[str, Any],
    cache_key: str,
) -> dict[str, Any]:
    marker_path = object_path / MARKER_NAME
    if not marker_path.is_file():
        raise AssetVerificationError(f"cache marker is missing: {marker_path}")
    try:
        marker = _read_json(marker_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetVerificationError(f"cache marker is unreadable: {exc}") from exc

    if not isinstance(marker, dict) or marker.get("schema") != OBJECT_SCHEMA:
        raise AssetVerificationError("cache marker has an unsupported schema")
    if marker.get("cache_key") != cache_key:
        raise AssetVerificationError("cache marker key does not match its directory")
    if marker.get("descriptor") != _descriptor(asset):
        raise AssetVerificationError("cache marker does not match the locked asset")
    if marker.get("tree_hash_algorithm") != TREE_HASH_ALGORITHM:
        raise AssetVerificationError(
            "cache marker has an unsupported tree hash algorithm"
        )

    stats = _scan_payload(object_path)
    if marker.get("file_count") != stats["file_count"]:
        raise AssetVerificationError("cached file count differs from its receipt")
    if marker.get("total_bytes") != stats["total_bytes"]:
        raise AssetVerificationError("cached byte count differs from its receipt")
    if marker.get("tree_sha256") != stats["tree_sha256"]:
        raise AssetVerificationError(
            "cached payload tree SHA-256 differs from its receipt"
        )

    expected_bytes = asset.get("expected_bytes")
    if expected_bytes is not None and stats["total_bytes"] != expected_bytes:
        raise AssetVerificationError(
            f"cached byte count {stats['total_bytes']} does not equal "
            f"expected_bytes {expected_bytes}"
        )
    return stats


def _remove_hf_local_metadata(path: Path) -> None:
    metadata = path / ".cache" / "huggingface"
    if metadata.exists():
        shutil.rmtree(metadata)
    cache_dir = path / ".cache"
    try:
        cache_dir.rmdir()
    except OSError:
        pass


def _allocated_bytes(root: Path) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*dir_names, *file_names]:
            path = current_path / name
            if path.is_symlink():
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            total += int(getattr(info, "st_blocks", 0)) * 512 or int(info.st_size)
    return total


def _snapshot_kwargs(
    snapshot_download_fn: Callable[..., Any],
    *,
    asset: Mapping[str, Any],
    local_dir: Path,
    hf_cache_dir: Path,
    offline: bool,
    token: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "repo_id": asset["repository"],
        "repo_type": asset["kind"],
        "revision": asset["revision"],
        "local_dir": str(local_dir),
        "cache_dir": str(hf_cache_dir),
        "local_files_only": offline,
        "token": token,
    }
    for field in ("allow_patterns", "ignore_patterns"):
        if field in asset:
            kwargs[field] = list(asset[field])

    try:
        parameters = inspect.signature(snapshot_download_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "local_dir_use_symlinks" in parameters:
        kwargs["local_dir_use_symlinks"] = False
    return kwargs


def _publish_asset(
    *,
    asset: Mapping[str, Any],
    object_path: Path,
    objects_dir: Path,
    partials_dir: Path,
    hf_cache_dir: Path,
    cache_key: str,
    offline: bool,
    token: str | None,
    snapshot_download_fn: Callable[..., Any],
) -> dict[str, Any]:
    temp_path = partials_dir / cache_key
    temp_path.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        kwargs = _snapshot_kwargs(
            snapshot_download_fn,
            asset=asset,
            local_dir=temp_path,
            hf_cache_dir=hf_cache_dir,
            offline=offline,
            token=token,
        )
        snapshot_download_fn(**kwargs)
        _remove_hf_local_metadata(temp_path)
        stats = _scan_payload(temp_path)

        expected_bytes = asset.get("expected_bytes")
        if expected_bytes is not None and stats["total_bytes"] != expected_bytes:
            raise AssetVerificationError(
                f"downloaded byte count {stats['total_bytes']} does not equal "
                f"expected_bytes {expected_bytes}"
            )

        marker = {
            "schema": OBJECT_SCHEMA,
            "cache_key": cache_key,
            "descriptor": _descriptor(asset),
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            **stats,
        }
        _write_json_fsync(temp_path / MARKER_NAME, marker)
        _fsync_directory(temp_path)

        try:
            os.rename(temp_path, object_path)
            published = True
            _fsync_directory(objects_dir)
        except OSError:
            if not object_path.exists():
                raise
            # A concurrent process won publication. Verify its result before
            # discarding this complete temporary copy.
            result = verify_object(object_path, asset, cache_key)
            shutil.rmtree(temp_path, ignore_errors=True)
            return result

        return stats
    finally:
        # Keep an incomplete local_dir and its Hugging Face download metadata.
        # snapshot_download can resume the same immutable revision on retry.
        if published:
            _fsync_directory(partials_dir)


def fetch_assets(
    *,
    lock_path: Path,
    cache_dir: Path,
    set_names: Sequence[str],
    verify_only: bool = False,
    offline: bool = False,
    snapshot_download_fn: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    lock = load_asset_lock(lock_path)
    asset_names = resolve_sets(lock, set_names)
    environment = os.environ if environ is None else environ
    token = environment.get("HF_TOKEN") or environment.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    secrets = [token] if token else []

    objects_dir = cache_dir / "objects"
    hf_cache_dir = cache_dir / ".hf"
    partials_dir = cache_dir / "partials"
    locks_dir = cache_dir / "locks"

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "timestamp_utc": _utc_now(),
        "status": "pass",
        "mode": {
            "verify_only": verify_only,
            "offline": offline,
        },
        "sets": list(set_names),
        "assets": [],
    }
    if not verify_only:
        missing_expected = 0
        reusable_partial_bytes = 0
        unknown_size: list[str] = []
        for asset_name in asset_names:
            asset = lock["assets"][asset_name]
            cache_key = asset_cache_key(asset)
            if (objects_dir / cache_key).exists():
                continue
            partial_bytes = _allocated_bytes(partials_dir / cache_key)
            reusable_partial_bytes += partial_bytes
            expected = asset.get("expected_bytes")
            if expected is None:
                unknown_size.append(asset_name)
            else:
                missing_expected += max(0, int(expected) - partial_bytes)
        probe = cache_dir
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free_bytes = shutil.disk_usage(probe).free
        safety_bytes = max(1024**3, (missing_expected + 19) // 20)
        required_bytes = missing_expected + safety_bytes if missing_expected else 0
        receipt["disk_preflight"] = {
            "probe_path": str(probe),
            "free_bytes": free_bytes,
            "missing_expected_bytes": missing_expected,
            "reusable_partial_bytes": reusable_partial_bytes,
            "safety_bytes": safety_bytes if missing_expected else 0,
            "required_bytes": required_bytes,
            "unknown_size_assets": unknown_size,
        }
        if free_bytes < required_bytes:
            receipt["status"] = "fail"
            receipt["error"] = (
                f"insufficient free disk: need at least {required_bytes} bytes "
                f"for locked missing assets, found {free_bytes}"
            )
            return receipt
        objects_dir.mkdir(parents=True, exist_ok=True)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        partials_dir.mkdir(parents=True, exist_ok=True)
        locks_dir.mkdir(parents=True, exist_ok=True)

    downloader = snapshot_download_fn
    for asset_name in asset_names:
        asset = lock["assets"][asset_name]
        cache_key = asset_cache_key(asset)
        object_path = objects_dir / cache_key
        item: dict[str, Any] = {
            "name": asset_name,
            "kind": asset["kind"],
            "repository": asset["repository"],
            "revision": asset["revision"],
            "cache_key": cache_key,
            "path": str(object_path),
        }
        try:
            if verify_only:
                if not object_path.exists():
                    raise AssetVerificationError(
                        f"content-addressed object is missing: {object_path}"
                    )
                stats = verify_object(object_path, asset, cache_key)
                item["action"] = "verified"
            else:
                lock_path = locks_dir / f"{cache_key}.lock"
                with lock_path.open("w") as lock_file:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise AssetVerificationError(
                            f"another process is fetching asset {asset_name}"
                        ) from exc
                    if object_path.exists():
                        stats = verify_object(object_path, asset, cache_key)
                        item["action"] = "verified"
                    else:
                        if downloader is None:
                            hub = importlib.import_module("huggingface_hub")
                            downloader = hub.snapshot_download
                        stats = _publish_asset(
                            asset=asset,
                            object_path=object_path,
                            objects_dir=objects_dir,
                            partials_dir=partials_dir,
                            hf_cache_dir=hf_cache_dir,
                            cache_key=cache_key,
                            offline=offline,
                            token=token,
                            snapshot_download_fn=downloader,
                        )
                        item["action"] = "published"
            item.update(stats)
            item["status"] = "pass"
        except Exception as exc:
            item["status"] = "fail"
            item["error"] = _redact(exc, secrets)
            receipt["status"] = "fail"
        receipt["assets"].append(item)
    return receipt


def _atomic_write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                receipt,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _normalize_sets(values: Sequence[str] | None) -> list[str]:
    if not values:
        return ["smoke"]
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    capsule_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=capsule_root / "assets.lock.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("ENERGY_ASSET_CACHE", "/cache/assets")),
    )
    parser.add_argument(
        "--set",
        dest="set_names",
        action="append",
        help="Named asset set; repeat the option or use comma-separated names",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify already-published objects without importing Hugging Face",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access through snapshot_download",
    )
    parser.add_argument(
        "--receipt-file",
        type=Path,
        help="Also atomically write the JSON receipt to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Compatibility flag; output is always one JSON document",
    )
    return parser


def _failure_receipt(
    *,
    set_names: Sequence[str],
    verify_only: bool,
    offline: bool,
    error: Any,
    secrets: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "timestamp_utc": _utc_now(),
        "status": "fail",
        "mode": {
            "verify_only": verify_only,
            "offline": offline,
        },
        "sets": list(set_names),
        "assets": [],
        "error": _redact(error, secrets),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_names = _normalize_sets(args.set_names)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    secrets = [token] if token else []

    try:
        receipt = fetch_assets(
            lock_path=args.lock_file,
            cache_dir=args.cache_dir,
            set_names=set_names,
            verify_only=args.verify_only,
            offline=args.offline,
        )
    except Exception as exc:
        receipt = _failure_receipt(
            set_names=set_names,
            verify_only=args.verify_only,
            offline=args.offline,
            error=exc,
            secrets=secrets,
        )

    if args.receipt_file is not None:
        try:
            _atomic_write_receipt(args.receipt_file, receipt)
        except Exception as exc:
            receipt["status"] = "fail"
            receipt["receipt_file_error"] = _redact(exc, secrets)

    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 1 if receipt["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
