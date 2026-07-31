#!/usr/bin/env python3
"""Resolve an upstream Energy experiment into a deterministic run config.

The resolver is intentionally side-effect free until the final write step.  In
particular, ``--check-only`` never creates the run directory or cache paths.
It also never reads credentials from the environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised by the test skip
    yaml = None


SCHEMA = "energy-repro/resolved-config/v1"
ASSET_OBJECT_SCHEMA = "energy-repro/asset-object/v1"
ASSET_MARKER_NAME = ".energy-asset.json"
TREE_HASH_ALGORITHM = "sha256-jsonl-posix-path-size-content-sha256-v1"
DERIVED_FINGERPRINT_SCHEMA = "energy-repro/derived-fingerprint/v1"
DERIVED_RECEIPT_SCHEMA = "energy-repro/derived-receipt/v1"
EXECUTION_PROVENANCE_SCHEMA = "energy-repro/execution-provenance/v1"
PILOT_CLASSIFICATION = "workflow-pilot-not-paper-reproduction"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
PORTABLE_PROFILE_RE = re.compile(r"(?:^|[-_])portable(?:$|[-_])", re.IGNORECASE)
SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "hf_token",
    "huggingface_token",
    "password",
    "passwd",
    "secret",
    "wandb_api_key",
}
STAGE_RAW_ROLES = {
    "smoke": frozenset(),
    "preprocess": frozenset({"dataset", "tokenizer"}),
    "teacher": frozenset({"teacher", "tokenizer"}),
    "student": frozenset({"student", "tokenizer"}),
    "eval": frozenset({"tokenizer"}),
    "all": frozenset({"teacher", "student", "tokenizer", "dataset"}),
}


class ConfigResolutionError(ValueError):
    """Raised when a config cannot be resolved safely."""


def _require_yaml() -> None:
    if yaml is None:
        raise ConfigResolutionError(
            "PyYAML is required to resolve configs. Install the capsule dependencies first."
        )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; non-mapping values and lists are replaced."""

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    _require_yaml()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigResolutionError(f"cannot read YAML file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigResolutionError(f"invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError(f"expected a mapping at the root of {path}")
    return copy.deepcopy(dict(payload))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ConfigResolutionError(f"cannot read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigResolutionError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError(f"expected a mapping at the root of {path}")
    return copy.deepcopy(dict(payload))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigResolutionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _asset_descriptor(asset: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact immutable descriptor used by fetch_assets.py."""

    descriptor: dict[str, Any] = {
        "kind": asset["kind"],
        "repository": asset["repository"],
        "revision": asset["revision"],
    }
    for field in ("allow_patterns", "ignore_patterns"):
        if field in asset:
            descriptor[field] = sorted(asset[field])
    return descriptor


def _asset_cache_key(asset: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _asset_descriptor(asset),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_asset_payload(root: Path) -> tuple[int, int, str]:
    payload_files: list[tuple[str, Path]] = []
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in dir_names:
            path = current_path / directory
            if path.is_symlink():
                raise ConfigResolutionError(
                    f"asset object contains a symlinked directory: {path}"
                )
        for filename in file_names:
            path = current_path / filename
            if path.name == ASSET_MARKER_NAME and path.parent == root:
                continue
            if path.is_symlink():
                raise ConfigResolutionError(
                    f"asset object contains a symlinked file: {path}"
                )
            if not path.is_file():
                raise ConfigResolutionError(
                    f"asset object contains a special file: {path}"
                )
            payload_files.append((path.relative_to(root).as_posix(), path))
    if not payload_files:
        raise ConfigResolutionError(f"asset object contains no payload files: {root}")

    tree_digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, path in sorted(payload_files, key=lambda item: item[0]):
        content_digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content_digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise ConfigResolutionError(
                f"cannot hash asset payload file {path}: {exc}"
            ) from exc
        record = {
            "path": relative_path,
            "sha256": content_digest.hexdigest(),
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
        total_bytes += size
    return len(payload_files), total_bytes, tree_digest.hexdigest()


def _verify_asset_object(
    object_path: Path,
    asset: Mapping[str, Any],
    cache_key: str,
) -> tuple[bool, str | None, dict[str, int | str] | None]:
    """Verify a content-addressed object and its fetch receipt."""

    if not object_path.is_dir():
        return False, f"asset object directory is missing: {object_path}", None
    marker_path = object_path / ASSET_MARKER_NAME
    if not marker_path.is_file():
        return False, f"asset object marker is missing: {marker_path}", None
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"asset object marker is unreadable at {marker_path}: {exc}", None
    if not isinstance(marker, Mapping) or marker.get("schema") != ASSET_OBJECT_SCHEMA:
        return False, f"asset object marker has an unsupported schema: {marker_path}", None
    if marker.get("cache_key") != cache_key:
        return False, f"asset object marker key does not match {cache_key}: {marker_path}", None
    if marker.get("descriptor") != _asset_descriptor(asset):
        return False, f"asset object marker does not match its locked descriptor: {marker_path}", None
    try:
        file_count, total_bytes, tree_sha256 = _scan_asset_payload(object_path)
    except ConfigResolutionError as exc:
        return False, str(exc), None
    if marker.get("file_count") != file_count:
        return False, f"asset object file count differs from marker: {marker_path}", None
    if marker.get("total_bytes") != total_bytes:
        return False, f"asset object byte count differs from marker: {marker_path}", None
    marker_algorithm = marker.get("tree_hash_algorithm")
    if marker_algorithm != TREE_HASH_ALGORITHM:
        return False, f"asset object marker uses an unsupported tree hash: {marker_path}", None
    if marker.get("tree_sha256") != tree_sha256:
        return False, f"asset object tree hash differs from marker: {marker_path}", None
    expected_bytes = asset.get("expected_bytes")
    if expected_bytes is not None and expected_bytes != total_bytes:
        return (
            False,
            f"asset object byte count {total_bytes} differs from expected_bytes "
            f"{expected_bytes}: {object_path}",
            None,
        )
    return (
        True,
        None,
        {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "tree_sha256": tree_sha256,
        },
    )


def _verify_derived_receipt(
    receipt_path: Path,
    artifact_path: Path,
    artifact: str,
    fingerprint: str,
    expected_provenance: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Verify that an artifact was produced for this exact immutable input set."""

    if receipt_path.is_symlink():
        return False, f"derived receipt must not be a symlink: {receipt_path}"
    if not receipt_path.is_file():
        return False, f"derived receipt is missing: {receipt_path}"
    try:
        with receipt_path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"derived receipt is unreadable at {receipt_path}: {exc}"
    if not isinstance(receipt, Mapping):
        return False, f"derived receipt must contain a JSON object: {receipt_path}"
    if receipt.get("schema") != DERIVED_RECEIPT_SCHEMA:
        return False, f"derived receipt has an unsupported schema: {receipt_path}"
    if receipt.get("artifact") != artifact:
        return False, f"derived receipt names a different artifact: {receipt_path}"
    if receipt.get("fingerprint") != fingerprint:
        return False, f"derived receipt fingerprint does not match: {receipt_path}"
    if receipt.get("status") != "succeeded":
        return False, f"derived receipt status is not 'succeeded': {receipt_path}"
    if receipt.get("tree_hash_algorithm") != TREE_HASH_ALGORITHM:
        return False, f"derived receipt tree hash algorithm is unsupported: {receipt_path}"
    if (
        expected_provenance is not None
        and receipt.get("producer", {}).get("execution_provenance")
        != dict(expected_provenance)
    ):
        return False, f"derived receipt producer provenance does not match: {receipt_path}"
    if not artifact_path.is_dir():
        return False, f"derived artifact directory is missing: {artifact_path}"
    try:
        file_count, total_bytes, tree_sha256 = _scan_asset_payload(artifact_path)
    except ConfigResolutionError as exc:
        return False, str(exc)
    payload = receipt.get("payload")
    expected = {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree_sha256,
    }
    if payload != expected:
        return False, f"derived artifact content tree differs from receipt: {artifact_path}"
    return True, None


def deterministic_yaml(config: Mapping[str, Any]) -> str:
    """Return stable, sorted YAML without aliases."""

    _require_yaml()

    class NoAliasSafeDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:
            return True

    return yaml.dump(
        dict(config),
        Dumper=NoAliasSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        explicit_start=False,
        sort_keys=True,
        width=4096,
    )


def _as_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _ensure_mapping(parent: MutableMapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        value = {}
        parent[key] = value
    elif not isinstance(value, dict):
        value = dict(value)
        parent[key] = value
    return value


def _set_nested(config: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        current = _ensure_mapping(current, key)
    current[keys[-1]] = copy.deepcopy(value)


def _get_nested(config: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _scrub_secrets(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    """Remove known credential fields without recording their values."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        removed: list[str] = []
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.lower() in SECRET_KEYS:
                removed.append(child_path)
                continue
            child, child_removed = _scrub_secrets(raw_value, child_path)
            clean[key] = child
            removed.extend(child_removed)
        return clean, removed
    if isinstance(value, list):
        clean_list: list[Any] = []
        removed: list[str] = []
        for index, item in enumerate(value):
            child, child_removed = _scrub_secrets(item, f"{path}[{index}]")
            clean_list.append(child)
            removed.extend(child_removed)
        return clean_list, removed
    return copy.deepcopy(value), []


def _placeholder_locations(value: Any, path: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            locations.extend(_placeholder_locations(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_placeholder_locations(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.rstrip("/")
        if "/path/to/" in value or normalized == "/path/to":
            locations.append(path)
    return locations


def _profile_is_portable(profile: Mapping[str, Any]) -> bool:
    name = str(profile.get("name", ""))
    label = str(profile.get("compliance_label", ""))
    return bool(PORTABLE_PROFILE_RE.search(name) or "portable" in label.lower())


def _validate_profile(profile: Mapping[str, Any]) -> None:
    schema = profile.get("schema")
    if schema not in (None, "energy-repro/profile/v1"):
        raise ConfigResolutionError(f"unsupported profile schema: {schema!r}")
    if not profile.get("name"):
        raise ConfigResolutionError("profile is missing a non-empty 'name'")


def _validate_recipe(recipe: Mapping[str, Any]) -> None:
    if not recipe:
        return
    schema = recipe.get("schema")
    if schema not in (None, "energy-repro/recipe/v1"):
        raise ConfigResolutionError(f"unsupported recipe schema: {schema!r}")
    if not recipe.get("name"):
        raise ConfigResolutionError("recipe is missing a non-empty 'name'")
    for field in ("allowed_profiles", "allowed_stages"):
        values = recipe.get(field)
        if values is not None and (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ConfigResolutionError(
                f"recipe.{field} must be a non-empty list of names"
            )
    sft_data_mode = recipe.get("sft_data_mode")
    if sft_data_mode not in (None, "synthetic", "original"):
        raise ConfigResolutionError(
            "recipe.sft_data_mode must be 'synthetic' or 'original'"
        )
    if sft_data_mode == "original" and recipe.get("pipeline") != "sft":
        raise ConfigResolutionError(
            "recipe.sft_data_mode='original' requires pipeline='sft'"
        )
    pilot = recipe.get("pilot")
    if pilot is not None:
        if not isinstance(pilot, Mapping):
            raise ConfigResolutionError("recipe.pilot must be a mapping")
        if pilot.get("enabled") and (
            recipe.get("classification") != PILOT_CLASSIFICATION
            or pilot.get("classification") != PILOT_CLASSIFICATION
        ):
            raise ConfigResolutionError(
                "pilot recipes must use the workflow-pilot-not-paper-reproduction classification"
            )


def _validate_assets_lock(lock: Mapping[str, Any], formal: bool) -> None:
    schema = lock.get("schema")
    if schema not in (None, "energy-repro/assets-lock/v1"):
        raise ConfigResolutionError(f"unsupported assets lock schema: {schema!r}")
    assets = lock.get("assets", {})
    sets = lock.get("sets", {})
    if not isinstance(assets, Mapping) or not isinstance(sets, Mapping):
        raise ConfigResolutionError("assets lock must contain mapping-valued 'assets' and 'sets'")
    for asset_id, spec in assets.items():
        if not isinstance(spec, Mapping):
            raise ConfigResolutionError(f"asset {asset_id!r} must be a mapping")
        repository = spec.get("repository")
        revision = spec.get("revision")
        if not repository or not revision:
            raise ConfigResolutionError(
                f"asset {asset_id!r} must contain repository and revision"
            )
        if formal and not HEX40_RE.fullmatch(str(revision)):
            raise ConfigResolutionError(
                f"asset {asset_id!r} revision is not a lowercase 40-character commit: {revision!r}"
                )


def _validate_execution_provenance(
    provenance: Mapping[str, Any],
    formal: bool,
) -> None:
    if not provenance:
        if formal:
            raise ConfigResolutionError(
                "formal config resolution requires execution provenance"
            )
        return
    if provenance.get("schema") != EXECUTION_PROVENANCE_SCHEMA:
        raise ConfigResolutionError("unsupported execution provenance schema")
    required_sha256 = (
        "upstream_archive_sha256",
        "dependency_lock_sha256",
        "asset_lock_sha256",
        "build_context_sha256",
        "resolver_tool_sha256",
    )
    if not HEX40_RE.fullmatch(str(provenance.get("upstream_commit") or "")):
        raise ConfigResolutionError("execution provenance has no full upstream commit")
    for key in required_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key) or "")):
            raise ConfigResolutionError(f"execution provenance has no valid {key}")
    capsule_tools = provenance.get("capsule_tools")
    if capsule_tools is not None:
        if not isinstance(capsule_tools, Mapping) or not capsule_tools:
            raise ConfigResolutionError(
                "execution provenance capsule_tools must be a non-empty mapping"
            )
        for name, digest in capsule_tools.items():
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9._-]+\.py", name)
                or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
            ):
                raise ConfigResolutionError(
                    "execution provenance contains an invalid capsule tool digest"
                )


def _asset_ids_for_recipe(
    recipe: Mapping[str, Any],
    assets_lock: Mapping[str, Any],
) -> list[str]:
    asset_set = recipe.get("asset_set") if recipe else None
    if not asset_set:
        return []
    sets = assets_lock.get("sets", {})
    if asset_set not in sets:
        raise ConfigResolutionError(f"asset set {asset_set!r} is not present in assets lock")
    asset_ids = sets[asset_set]
    if not isinstance(asset_ids, list) or not all(isinstance(item, str) for item in asset_ids):
        raise ConfigResolutionError(f"asset set {asset_set!r} must be a list of asset ids")
    missing = sorted(set(asset_ids) - set(assets_lock.get("assets", {})))
    if missing:
        raise ConfigResolutionError(
            f"asset set {asset_set!r} references unknown assets: {', '.join(missing)}"
        )
    return sorted(set(asset_ids))


def _find_asset_by_repository(
    repository: str | None,
    assets: Mapping[str, Any],
    preferred_ids: Iterable[str],
) -> tuple[str, Mapping[str, Any]] | None:
    if not repository:
        return None
    preferred = set(preferred_ids)
    matches = [
        (str(asset_id), spec)
        for asset_id, spec in assets.items()
        if isinstance(spec, Mapping) and spec.get("repository") == repository
    ]
    if not matches:
        return None
    preferred_matches = [item for item in matches if item[0] in preferred]
    candidates = preferred_matches or matches
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1:
        raise ConfigResolutionError(
            f"repository {repository!r} maps to multiple assets: "
            + ", ".join(asset_id for asset_id, _ in candidates)
        )
    return candidates[0]


def _role_repositories(
    config: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, str | None]:
    return {
        "teacher": str(
            recipe.get("teacher")
            or _get_nested(config, ("model", "teacher"), "")
        )
        or None,
        "student": str(
            recipe.get("student")
            or _get_nested(config, ("model", "student"), "")
        )
        or None,
        "tokenizer": str(_get_nested(config, ("data", "tokenizer_name"), "")) or None,
        "dataset": str(
            recipe.get("dataset")
            or _get_nested(config, ("data", "dataset_name"), "")
        )
        or None,
    }


def _resolve_assets(
    config: MutableMapping[str, Any],
    recipe: Mapping[str, Any],
    assets_lock: Mapping[str, Any],
    asset_cache: Path,
    stage: str,
    formal: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    assets = assets_lock.get("assets", {})
    preferred_ids = _asset_ids_for_recipe(recipe, assets_lock)
    role_repositories = _role_repositories(config, recipe)
    required_roles = STAGE_RAW_ROLES.get(stage, frozenset())
    role_assets: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    verification_cache: dict[
        str, tuple[bool, str | None, dict[str, Any] | None]
    ] = {}

    def verify_once(
        snapshot_path: Path,
        spec: Mapping[str, Any],
        cache_key: str,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        if cache_key not in verification_cache:
            verification_cache[cache_key] = _verify_asset_object(
                snapshot_path,
                spec,
                cache_key,
            )
        return verification_cache[cache_key]

    for role in ("teacher", "student", "tokenizer", "dataset"):
        repository = role_repositories[role]
        match = _find_asset_by_repository(repository, assets, preferred_ids)
        if match is None:
            if role in required_roles:
                message = (
                    f"required {role} repository {repository!r} has no entry in assets lock"
                )
                if formal:
                    raise ConfigResolutionError(message)
                warnings.append(message)
            continue
        asset_id, spec = match
        revision = str(spec["revision"])
        cache_key = _asset_cache_key(spec)
        snapshot_path = asset_cache / "assets" / "objects" / cache_key
        if role in required_roles:
            verified, verification_error, payload = verify_once(
                snapshot_path, spec, cache_key
            )
        else:
            verified, verification_error, payload = (
                False,
                "not content-verified because this role is not required for the selected stage",
                None,
            )
        record = {
            "asset_id": asset_id,
            "kind": spec.get("kind"),
            "repository": str(spec["repository"]),
            "revision": revision,
            "cache_key": cache_key,
            "path": str(snapshot_path),
            "exists": snapshot_path.is_dir(),
            "verified": verified,
            "marker_path": str(snapshot_path / ASSET_MARKER_NAME),
            "verification_error": verification_error,
            "payload": payload,
            "required_for_stage": role in required_roles,
            "roles": [role],
        }
        role_assets[role] = record

    # Record the complete recipe set as well, coalescing roles that share an asset.
    records_by_id: dict[str, dict[str, Any]] = {}
    for asset_id in preferred_ids:
        spec = assets[asset_id]
        revision = str(spec["revision"])
        cache_key = _asset_cache_key(spec)
        snapshot_path = asset_cache / "assets" / "objects" / cache_key
        if cache_key in verification_cache:
            verified, verification_error, payload = verification_cache[cache_key]
        else:
            verified, verification_error, payload = (
                False,
                "not content-verified because this asset is not required for the selected stage",
                None,
            )
        records_by_id[asset_id] = {
            "asset_id": asset_id,
            "kind": spec.get("kind"),
            "repository": str(spec["repository"]),
            "revision": revision,
            "cache_key": cache_key,
            "path": str(snapshot_path),
            "exists": snapshot_path.is_dir(),
            "verified": verified,
            "marker_path": str(snapshot_path / ASSET_MARKER_NAME),
            "verification_error": verification_error,
            "payload": payload,
            "required_for_stage": False,
            "roles": [],
        }
    for role, role_record in role_assets.items():
        record = records_by_id.setdefault(role_record["asset_id"], copy.deepcopy(role_record))
        record["required_for_stage"] = bool(
            record["required_for_stage"] or role_record["required_for_stage"]
        )
        if role not in record["roles"]:
            record["roles"].append(role)
            record["roles"].sort()

    for record in records_by_id.values():
        if record["required_for_stage"] and not record["verified"]:
            message = (
                f"required asset snapshot is missing or invalid for stage {stage!r}: "
                f"{record['asset_id']} at {record['path']}; "
                f"{record['verification_error']}"
            )
            if formal:
                raise ConfigResolutionError(message)
            warnings.append(message)
        elif not record["verified"]:
            warnings.append(
                f"non-required asset snapshot is not cached or invalid: "
                f"{record['asset_id']} at {record['path']}; "
                f"{record['verification_error']}"
            )

    return (
        [records_by_id[key] for key in sorted(records_by_id)],
        role_assets,
        sorted(set(warnings)),
    )


def _apply_recipe(config: dict[str, Any], recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not recipe:
        return config
    overlay: dict[str, Any] = {}
    if recipe.get("pipeline"):
        overlay["pipeline"] = recipe["pipeline"]
    model: dict[str, Any] = {}
    if recipe.get("teacher"):
        model["teacher"] = recipe["teacher"]
    if recipe.get("student"):
        model["student"] = recipe["student"]
    if model:
        overlay["model"] = model
    if recipe.get("dataset"):
        overlay["data"] = {"dataset_name": recipe["dataset"]}
    config = deep_merge(config, overlay)
    for key in ("config_overrides", "overrides"):
        value = recipe.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ConfigResolutionError(f"recipe.{key} must be a mapping")
            config = deep_merge(config, value)
    return config


def _apply_profile(config: dict[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("config_overrides", "overrides"):
        value = profile.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ConfigResolutionError(f"profile.{key} must be a mapping")
            config = deep_merge(config, value)

    energy_overrides = profile.get("energy_overrides", {})
    if not isinstance(energy_overrides, Mapping):
        raise ConfigResolutionError("profile.energy_overrides must be a mapping")
    config = deep_merge(config, {"energy": dict(energy_overrides)})

    if _profile_is_portable(profile):
        config = deep_merge(
            config,
            {
                "energy": {
                    "track_cpu": False,
                    "total_energy_policy": "gpu_only",
                }
            },
        )
    return config


def _apply_runtime_paths(
    config: MutableMapping[str, Any],
    recipe: Mapping[str, Any],
    role_assets: Mapping[str, Mapping[str, Any]],
    run_root: Path,
    asset_cache: Path,
    derived_fingerprint: str,
    stage: str,
    execution_provenance: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    output = _ensure_mapping(config, "output")
    output["run_dir"] = str(run_root / "energy" / stage)
    output["output_dir"] = str(run_root / "outputs" / stage)
    if output.get("checkpoint_dir") == "None":
        output["checkpoint_dir"] = None

    benchmark = _ensure_mapping(config, "benchmark")
    benchmark["output_dir"] = str(run_root / "outputs" / "eval")
    benchmark["model"] = str(run_root / "outputs" / "student" / "final_model")

    recipe_name = str(recipe.get("name") or "experiment")
    derived_root = (
        asset_cache / "derived" / recipe_name / derived_fingerprint[:16]
    )
    receipt_root = derived_root / ".receipts"
    artifact_paths = {
        "preprocessed": derived_root / "preprocessed",
        "logprob_cache": derived_root / "logprob_cache",
        "teacher_logprobs": derived_root / "logprob_cache" / "teacher_logprobs",
        "synthetic_dataset": derived_root / "synthetic_dataset",
    }
    original_sft = (
        str(config.get("pipeline", "")).lower() == "sft"
        and recipe.get("sft_data_mode") == "original"
    )
    derived: dict[str, dict[str, Any]] = {
        "preprocessed": {
            "required_for_stage": stage == "teacher"
            or (stage == "student" and original_sft),
            "target_for_stage": stage == "preprocess",
        },
        "logprob_cache": {
            "required_for_stage": stage == "student"
            and str(config.get("pipeline", "")).lower() == "kd",
            # This is only a parent/chunk directory.  The final DatasetDict
            # below is the reusable scientific artifact and gets the receipt.
            "target_for_stage": False,
        },
        "teacher_logprobs": {
            "required_for_stage": stage == "student"
            and str(config.get("pipeline", "")).lower() == "kd",
            "target_for_stage": stage == "teacher"
            and str(config.get("pipeline", "")).lower() == "kd",
        },
        "synthetic_dataset": {
            "required_for_stage": stage == "student"
            and str(config.get("pipeline", "")).lower() == "sft"
            and not original_sft,
            "target_for_stage": stage == "teacher"
            and str(config.get("pipeline", "")).lower() == "sft"
            and not original_sft,
        },
    }
    for artifact, spec in derived.items():
        artifact_path = artifact_paths[artifact]
        receipt_path = receipt_root / f"{artifact}.json"
        receipt_valid, receipt_error = _verify_derived_receipt(
            receipt_path=receipt_path,
            artifact_path=artifact_path,
            artifact=artifact,
            fingerprint=derived_fingerprint,
            expected_provenance=execution_provenance,
        )
        spec.update(
            {
                "path": str(artifact_path),
                "exists": artifact_path.is_dir(),
                "receipt_path": str(receipt_path),
                "receipt_valid": receipt_valid,
                "receipt_error": receipt_error,
            }
        )

    data = _ensure_mapping(config, "data")
    data["dataset_path"] = derived["preprocessed"]["path"]
    data["dataset_teacher_logprobs"] = derived["teacher_logprobs"]["path"]

    dataset_choice = str(data.get("dataset_choice") or "tulu")
    selected_dataset = {}
    existing_datasets = data.get("datasets")
    if isinstance(existing_datasets, Mapping):
        candidate = existing_datasets.get(dataset_choice)
        if isinstance(candidate, Mapping):
            selected_dataset = copy.deepcopy(dict(candidate))
    selected_dataset["dataset_path"] = derived["preprocessed"]["path"]
    data["datasets"] = {dataset_choice: selected_dataset}

    distillation = _ensure_mapping(config, "distillation")
    distillation["logprob_cache_path"] = derived["logprob_cache"]["path"]
    synthetic = _ensure_mapping(config, "synthetic_data")
    synthetic["synthetic_dataset_path"] = (
        derived["preprocessed"]["path"]
        if original_sft
        else derived["synthetic_dataset"]["path"]
    )

    field_map = {
        "teacher": (("model", "teacher"), ("model", "teacher_revision")),
        "student": (("model", "student"), ("model", "student_revision")),
        "tokenizer": (("data", "tokenizer_name"), ("data", "tokenizer_revision")),
        "dataset": (("data", "dataset_name"), ("data", "dataset_revision")),
    }
    for role, (path_keys, revision_keys) in field_map.items():
        record = role_assets.get(role)
        if not record:
            continue
        _set_nested(config, path_keys, record["path"])
        _set_nested(config, revision_keys, record["revision"])
        if role == "dataset":
            selected_dataset["dataset_name"] = record["path"]
            selected_dataset["dataset_revision"] = record["revision"]
        elif role == "student":
            benchmark["model_type"] = record["path"]

    return derived


def _validate_stage_inputs(
    config: Mapping[str, Any],
    derived: Mapping[str, Mapping[str, Any]],
    stage: str,
    formal: bool,
) -> list[str]:
    warnings: list[str] = []
    required_derived = [
        (name, spec)
        for name, spec in derived.items()
        if spec.get("required_for_stage")
    ]
    # KD's usable student input is the final teacher_logprobs dataset, not merely
    # its parent cache directory.
    if stage == "student" and str(config.get("pipeline", "")).lower() == "kd":
        required_derived = [
            item for item in required_derived if item[0] != "logprob_cache"
        ]
    for name, spec in required_derived:
        if spec.get("exists") and spec.get("receipt_valid"):
            continue
        details: list[str] = []
        if not spec.get("exists"):
            details.append("artifact directory is missing")
        if not spec.get("receipt_valid"):
            details.append(str(spec.get("receipt_error") or "receipt is invalid"))
        message = (
            f"required derived artifact is missing for stage {stage!r}: "
            f"{name} at {spec['path']}; "
            + "; ".join(details)
        )
        if formal:
            raise ConfigResolutionError(message)
        warnings.append(message)
    return warnings


def resolve_config(
    *,
    base_path: str | os.PathLike[str],
    experiment_path: str | os.PathLike[str],
    profile_path: str | os.PathLike[str],
    stage: str,
    run_root: str | os.PathLike[str],
    assets_lock_path: str | os.PathLike[str],
    asset_cache: str | os.PathLike[str],
    recipe_path: str | os.PathLike[str] | None = None,
    provenance_path: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    formal: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve config and return ``(config, metadata, deterministic_yaml)``."""

    if stage not in STAGE_RAW_ROLES:
        raise ConfigResolutionError(
            f"unsupported stage {stage!r}; expected one of: "
            + ", ".join(sorted(STAGE_RAW_ROLES))
        )

    base_file = _as_absolute(base_path)
    experiment_file = _as_absolute(experiment_path)
    profile_file = _as_absolute(profile_path)
    assets_file = _as_absolute(assets_lock_path)
    recipe_file = _as_absolute(recipe_path) if recipe_path is not None else None
    provenance_file = (
        _as_absolute(provenance_path) if provenance_path is not None else None
    )
    resolved_run_root = _as_absolute(run_root)
    resolved_asset_cache = _as_absolute(asset_cache)

    base = _load_yaml(base_file)
    experiment = _load_yaml(experiment_file)
    profile = _load_json(profile_file)
    recipe = _load_json(recipe_file) if recipe_file is not None else {}
    execution_provenance = (
        _load_json(provenance_file) if provenance_file is not None else {}
    )
    assets_lock = _load_json(assets_file)

    _validate_profile(profile)
    _validate_recipe(recipe)
    _validate_assets_lock(assets_lock, formal=formal)
    _validate_execution_provenance(execution_provenance, formal=formal)

    config = deep_merge(base, experiment)
    config = _apply_recipe(config, recipe)
    config = _apply_profile(config, profile)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigResolutionError("seed must be a non-negative integer")
        _set_nested(config, ("experiment", "seed"), seed)
    resolved_seed = _get_nested(config, ("experiment", "seed"), 42)

    asset_records, role_assets, asset_warnings = _resolve_assets(
        config=config,
        recipe=recipe,
        assets_lock=assets_lock,
        asset_cache=resolved_asset_cache,
        stage=stage,
        formal=formal,
    )
    source_files = {
        "base": base_file,
        "experiment": experiment_file,
        "profile": profile_file,
        "assets_lock": assets_file,
    }
    if recipe_file is not None:
        source_files["recipe"] = recipe_file
    if provenance_file is not None:
        source_files["execution_provenance"] = provenance_file
    sources = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in sorted(source_files.items())
    }
    fingerprint_source_hashes = {
        "base": sources["base"]["sha256"],
        "experiment": sources["experiment"]["sha256"],
        "profile": sources["profile"]["sha256"],
        "recipe": (
            sources["recipe"]["sha256"]
            if "recipe" in sources
            else _sha256_json({})
        ),
        "assets_lock": sources["assets_lock"]["sha256"],
    }
    selected_asset_descriptors = [
        {
            "asset_id": record["asset_id"],
            "descriptor": _asset_descriptor(
                assets_lock["assets"][record["asset_id"]]
            ),
        }
        for record in asset_records
    ]
    derived_fingerprint_inputs = {
        "schema": DERIVED_FINGERPRINT_SCHEMA,
        "sources": fingerprint_source_hashes,
        "selected_assets": selected_asset_descriptors,
        "execution_provenance": execution_provenance,
        "parameters": {
            "seed": resolved_seed,
        },
    }
    derived_fingerprint = _sha256_json(derived_fingerprint_inputs)

    derived = _apply_runtime_paths(
        config=config,
        recipe=recipe,
        role_assets=role_assets,
        run_root=resolved_run_root,
        asset_cache=resolved_asset_cache,
        derived_fingerprint=derived_fingerprint,
        stage=stage,
        execution_provenance=execution_provenance,
    )
    derived_root = (
        resolved_asset_cache
        / "derived"
        / str(recipe.get("name") or "experiment")
        / derived_fingerprint[:16]
    )

    wandb = _ensure_mapping(config, "wandb")
    wandb["mode"] = "offline"
    wandb["offline"] = True

    config, secrets_removed = _scrub_secrets(config)
    assert isinstance(config, dict)

    warnings = list(asset_warnings)
    warnings.extend(
        _validate_stage_inputs(
            config=config,
            derived=derived,
            stage=stage,
            formal=formal,
        )
    )

    placeholder_locations = _placeholder_locations(config)
    if formal and placeholder_locations:
        raise ConfigResolutionError(
            "formal config contains unresolved '/path/to/' placeholders at: "
            + ", ".join(placeholder_locations)
        )
    if placeholder_locations:
        warnings.append(
            "unresolved '/path/to/' placeholders remain at: "
            + ", ".join(placeholder_locations)
        )

    repro = {
        "schema": SCHEMA,
        "profile": str(profile["name"]),
        "recipe": str(recipe.get("name") or experiment_file.stem),
        "stage": stage,
        "seed": resolved_seed,
        "formal": bool(formal),
        "asset_cache": str(resolved_asset_cache),
        "derived_fingerprint": derived_fingerprint,
        "derived_root": str(derived_root),
        "assets": {
            role: {
                "asset_id": record["asset_id"],
                "repository": record["repository"],
                "revision": record["revision"],
                "cache_key": record["cache_key"],
                "path": record["path"],
            }
            for role, record in sorted(role_assets.items())
        },
    }
    config["energy_repro"] = repro

    yaml_text = deterministic_yaml(config)
    config_hash = _sha256_json(config)
    yaml_hash = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()

    pilot_enabled = bool(
        isinstance(recipe.get("pilot"), Mapping)
        and recipe["pilot"].get("enabled")
    )
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": stage,
        "seed": resolved_seed,
        "formal": bool(formal),
        "run_root": str(resolved_run_root),
        "asset_cache": str(resolved_asset_cache),
        "derived_fingerprint": derived_fingerprint,
        "derived_fingerprint_inputs": derived_fingerprint_inputs,
        "execution_provenance": execution_provenance,
        "derived_root": str(derived_root),
        "profile": {
            "name": str(profile["name"]),
            "source_mode": profile.get("source_mode"),
            "compliance_label": profile.get("compliance_label"),
            "portable": _profile_is_portable(profile),
        },
        "recipe": {
            "name": str(recipe.get("name") or experiment_file.stem),
            "pipeline": config.get("pipeline"),
            "asset_set": recipe.get("asset_set"),
            "sft_data_mode": recipe.get("sft_data_mode"),
            "classification": recipe.get("classification"),
            "pilot": pilot_enabled,
        },
        "runtime_environment": {
            "WANDB_MODE": "disabled" if pilot_enabled else "offline",
        },
        "assets": asset_records,
        "derived_artifacts": {
            key: derived[key] for key in sorted(derived)
        },
        "sources": sources,
        "secrets_removed": sorted(secrets_removed),
        "warnings": sorted(set(warnings)),
        "hashes": {
            "canonical_config_sha256": config_hash,
            "resolved_yaml_sha256": yaml_hash,
            "asset_selection_sha256": _sha256_json(asset_records),
            "derived_fingerprint_sha256": derived_fingerprint,
            "source_bundle_sha256": _sha256_json(
                {name: value["sha256"] for name, value in sources.items()}
            ),
        },
    }
    return config, metadata, yaml_text


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a deterministic Energy experiment configuration."
    )
    parser.add_argument(
        "--base",
        "--base-config",
        dest="base_path",
        required=True,
        help="Upstream base YAML.",
    )
    parser.add_argument(
        "--experiment",
        "--experiment-config",
        dest="experiment_path",
        required=True,
        help="Upstream experiment YAML.",
    )
    parser.add_argument("--profile", dest="profile_path", required=True, help="Profile JSON.")
    parser.add_argument("--recipe", dest="recipe_path", default=None, help="Recipe JSON.")
    parser.add_argument(
        "--provenance",
        dest="provenance_path",
        default=None,
        help="Execution provenance JSON binding source, dependencies, tools, and image context.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGE_RAW_ROLES),
        help="Stage whose inputs are being resolved.",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=None,
        help="Override the experiment seed with a non-negative integer.",
    )
    parser.add_argument("--run-root", required=True, help="Canonical run root.")
    parser.add_argument(
        "--assets-lock",
        "--asset-lock",
        dest="assets_lock_path",
        required=True,
        help="Immutable assets lock JSON.",
    )
    parser.add_argument(
        "--asset-cache",
        required=True,
        help="Asset cache root; verified raw snapshots live below assets/objects/<content-hash>.",
    )
    parser.add_argument(
        "--output",
        "--resolved-config",
        dest="output_path",
        default=None,
        help="Resolved YAML output (default: <run-root>/configs/<stage>.yaml).",
    )
    parser.add_argument(
        "--metadata-output",
        "--metadata",
        dest="metadata_path",
        default=None,
        help="Metadata JSON output (default: <run-root>/configs/<stage>.metadata.json).",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Fail on placeholders and missing stage inputs.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print metadata without creating files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _, metadata, yaml_text = resolve_config(
            base_path=args.base_path,
            experiment_path=args.experiment_path,
            profile_path=args.profile_path,
            recipe_path=args.recipe_path,
            provenance_path=args.provenance_path,
            stage=args.stage,
            run_root=args.run_root,
            assets_lock_path=args.assets_lock_path,
            asset_cache=args.asset_cache,
            seed=args.seed,
            formal=args.formal,
        )

        if args.check_only:
            sys.stdout.write(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            return 0

        run_root = _as_absolute(args.run_root)
        output_path = (
            _as_absolute(args.output_path)
            if args.output_path
            else run_root / "configs" / f"{args.stage}.yaml"
        )
        metadata_path = (
            _as_absolute(args.metadata_path)
            if args.metadata_path
            else run_root / "configs" / f"{args.stage}.metadata.json"
        )
        if output_path == metadata_path:
            raise ConfigResolutionError("config and metadata output paths must differ")

        _atomic_write(output_path, yaml_text.encode("utf-8"))
        _atomic_write(metadata_path, _canonical_json_bytes(metadata))
        sys.stdout.write(
            json.dumps(
                {
                    "config": str(output_path),
                    "metadata": str(metadata_path),
                    "canonical_config_sha256": metadata["hashes"][
                        "canonical_config_sha256"
                    ],
                    "resolved_yaml_sha256": metadata["hashes"][
                        "resolved_yaml_sha256"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    except ConfigResolutionError as exc:
        print(f"resolve_config: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
