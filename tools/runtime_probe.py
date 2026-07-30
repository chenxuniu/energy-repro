#!/usr/bin/env python3
"""Fail-closed runtime probe for the Energy reproduction environment.

The module deliberately imports torch, transformers, and pynvml only while a
probe is running.  This keeps profile validation and unit tests independent of
the heavyweight runtime.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA = "energy-repro/runtime-probe/v1"
EXPECTED_SOFTWARE = {
    "python": "3.10.13",
    "pytorch": "2.6.0+cu124",
    "cuda_userspace": "12.4",
    "transformers": "4.51.3",
}
EXPECTED_GPU_COUNT = 1
EXPECTED_POWER_LIMIT_WATTS = 700.0
POWER_TOLERANCE_WATTS = 1.0


def _clean_text(value: Any, limit: int = 400) -> str:
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text[:limit]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cpu_brand() -> str:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def _physical_memory_gib() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * page_count) / float(1024**3)
    except (AttributeError, OSError, ValueError):
        return 0.0


def default_profile() -> dict[str, Any]:
    return {
        "schema": "energy-repro/profile/v1",
        "name": "h100-portable",
        "hardware": {
            "gpu_count": 1,
            "gpu_name_contains": "H100",
            "min_vram_mib": 79000,
            "power_limit_watts": None,
            "cpu_name_contains": None,
            "min_ram_gib": None,
            "require_rapl": False,
        },
    }


def load_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_profile()

    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError("profile root must be a JSON object")
    if profile.get("schema") != "energy-repro/profile/v1":
        raise ValueError("unsupported profile schema")
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise ValueError("profile name must be a non-empty string")
    if not isinstance(profile.get("hardware", {}), dict):
        raise ValueError("profile hardware must be a JSON object")
    if "software" in profile and not isinstance(profile["software"], dict):
        raise ValueError("profile software must be a JSON object")
    return profile


def _is_strict(profile: Mapping[str, Any]) -> bool:
    name = str(profile.get("name", ""))
    label = str(profile.get("compliance_label", ""))
    return name == "paper-strict" or label.startswith("paper-strict")


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        status: str,
        *,
        expected: Any = None,
        actual: Any = None,
        message: str | None = None,
    ) -> None:
        item: dict[str, Any] = {"name": name, "status": status}
        if expected is not None:
            item["expected"] = expected
        if actual is not None:
            item["actual"] = actual
        if message:
            item["message"] = _clean_text(message)
        self.items.append(item)

    def require(
        self,
        condition: bool,
        name: str,
        *,
        expected: Any = None,
        actual: Any = None,
        message: str | None = None,
        false_status: str = "fail",
    ) -> None:
        self.add(
            name,
            "pass" if condition else false_status,
            expected=expected,
            actual=actual,
            message=None if condition else message,
        )

    @property
    def status(self) -> str:
        statuses = {item["status"] for item in self.items}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"


def _software_expectations(profile: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(EXPECTED_SOFTWARE)
    configured = profile.get("software", {})
    if isinstance(configured, Mapping):
        for key in expected:
            value = configured.get(key)
            if value is not None:
                expected[key] = str(value)
    return expected


def _load_module(
    name: str,
    loader: Callable[[str], Any],
    checks: Checks,
) -> Any | None:
    try:
        module = loader(name)
    except Exception as exc:  # imports can fail for several platform reasons
        checks.add(
            f"module.{name}",
            "fail",
            expected="importable",
            actual="unavailable",
            message=_clean_text(exc),
        )
        return None
    checks.add(f"module.{name}", "pass", expected="importable", actual="importable")
    return module


def _query_ecc(pynvml: Any, handle: Any, checks: Checks, gpu: dict[str, Any]) -> None:
    try:
        current, pending = pynvml.nvmlDeviceGetEccMode(handle)
        enabled = getattr(pynvml, "NVML_FEATURE_ENABLED", 1)
        gpu["ecc_mode"] = {
            "current": "enabled" if current == enabled else "disabled",
            "pending": "enabled" if pending == enabled else "disabled",
        }
        checks.require(
            current == enabled and pending == enabled,
            "gpu.ecc.mode",
            expected={"current": "enabled", "pending": "enabled"},
            actual=gpu["ecc_mode"],
            message="H100 ECC must be enabled now and after the next reboot",
        )
    except Exception as exc:
        checks.add(
            "gpu.ecc.mode",
            "fail",
            expected="queryable and enabled",
            actual="unavailable",
            message=_clean_text(exc),
        )
        return

    error_types = {
        "corrected": getattr(pynvml, "NVML_MEMORY_ERROR_TYPE_CORRECTED", 0),
        "uncorrected": getattr(pynvml, "NVML_MEMORY_ERROR_TYPE_UNCORRECTED", 1),
    }
    counter_types = {
        "volatile": getattr(pynvml, "NVML_VOLATILE_ECC", 0),
        "aggregate": getattr(pynvml, "NVML_AGGREGATE_ECC", 1),
    }
    counters: dict[str, dict[str, int]] = {}
    try:
        for counter_name, counter_type in counter_types.items():
            counters[counter_name] = {}
            for error_name, error_type in error_types.items():
                value = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    error_type,
                    counter_type,
                )
                counters[counter_name][error_name] = int(value)
        gpu["ecc_errors"] = counters
    except Exception as exc:
        checks.add(
            "gpu.ecc.counters",
            "fail",
            expected="queryable",
            actual="unavailable",
            message=_clean_text(exc),
        )
        return

    uncorrected = sum(values["uncorrected"] for values in counters.values())
    corrected = sum(values["corrected"] for values in counters.values())
    checks.require(
        uncorrected == 0,
        "gpu.ecc.uncorrected",
        expected=0,
        actual=uncorrected,
        message="uncorrected ECC errors make this GPU unsuitable for a run",
    )
    checks.require(
        corrected == 0,
        "gpu.ecc.corrected",
        expected=0,
        actual=corrected,
        false_status="warn",
        message="corrected ECC errors were observed; retain this in run provenance",
    )


def _query_row_remapper(
    pynvml: Any,
    handle: Any,
    checks: Checks,
    gpu: dict[str, Any],
    strict: bool,
) -> None:
    getter = getattr(pynvml, "nvmlDeviceGetRemappedRows", None)
    if getter is None:
        return
    try:
        corrected, uncorrected, pending, failed = getter(handle)
    except Exception:
        # ECC mode and counters are the required cross-version checks.  Row
        # remapper support varies across NVML releases.
        return

    gpu["row_remapper"] = {
        "corrected_rows": int(corrected),
        "uncorrected_rows": int(uncorrected),
        "pending": bool(pending),
        "failure": bool(failed),
    }
    checks.require(
        not bool(failed),
        "gpu.row_remapper.failure",
        expected=False,
        actual=bool(failed),
        message="NVML reports a row-remapping failure",
    )
    checks.require(
        not bool(pending),
        "gpu.row_remapper.pending",
        expected=False,
        actual=bool(pending),
        false_status="fail" if strict else "warn",
        message="a row remap is pending; reboot or replace the GPU before a strict run",
    )


def probe_runtime(
    profile: Mapping[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    checks = Checks()
    strict = _is_strict(profile)
    hardware = profile.get("hardware", {})
    if not isinstance(hardware, Mapping):
        hardware = {}
    expected_software = _software_expectations(profile)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp_utc": _utc_now(),
        "profile": {
            "name": str(profile.get("name", "h100-portable")),
            "strict": strict,
        },
        "software": {},
        "host": {},
        "gpus": [],
    }

    python_version = platform.python_version()
    report["software"]["python"] = python_version
    checks.require(
        python_version == expected_software["python"],
        "software.python",
        expected=expected_software["python"],
        actual=python_version,
    )

    torch = _load_module("torch", module_loader, checks)
    transformers = _load_module("transformers", module_loader, checks)
    pynvml = _load_module("pynvml", module_loader, checks)

    if transformers is not None:
        version = str(getattr(transformers, "__version__", "unknown"))
        report["software"]["transformers"] = version
        checks.require(
            version == expected_software["transformers"],
            "software.transformers",
            expected=expected_software["transformers"],
            actual=version,
        )

    if torch is not None:
        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda = str(getattr(getattr(torch, "version", None), "cuda", None))
        report["software"]["pytorch"] = torch_version
        report["software"]["cuda_userspace"] = torch_cuda
        checks.require(
            torch_version == expected_software["pytorch"],
            "software.pytorch",
            expected=expected_software["pytorch"],
            actual=torch_version,
        )
        checks.require(
            torch_cuda == expected_software["cuda_userspace"],
            "software.cuda_userspace",
            expected=expected_software["cuda_userspace"],
            actual=torch_cuda,
        )

        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            checks.add(
                "torch.cuda.available",
                "fail",
                expected=True,
                actual=False,
                message=_clean_text(exc),
            )
        else:
            checks.require(
                cuda_available,
                "torch.cuda.available",
                expected=True,
                actual=cuda_available,
            )

        if cuda_available:
            try:
                torch_count = int(torch.cuda.device_count())
            except Exception as exc:
                torch_count = -1
                checks.add(
                    "torch.cuda.device_count",
                    "fail",
                    expected=EXPECTED_GPU_COUNT,
                    actual="unavailable",
                    message=_clean_text(exc),
                )
            else:
                checks.require(
                    torch_count == EXPECTED_GPU_COUNT,
                    "torch.cuda.device_count",
                    expected=EXPECTED_GPU_COUNT,
                    actual=torch_count,
                )

            if torch_count == EXPECTED_GPU_COUNT:
                try:
                    name = str(torch.cuda.get_device_name(0))
                    capability = tuple(torch.cuda.get_device_capability(0))
                    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
                    report["gpus"].append(
                        {
                            "index": 0,
                            "torch_name": name,
                            "compute_capability": list(capability),
                            "torch_vram_mib": round(total_memory / float(1024**2), 2),
                        }
                    )
                    expected_name = str(hardware.get("gpu_name_contains") or "H100")
                    checks.require(
                        "H100" in name and expected_name in name,
                        "torch.cuda.gpu_name",
                        expected=f"contains H100 and {expected_name}",
                        actual=name,
                    )
                    checks.require(
                        capability == (9, 0),
                        "torch.cuda.compute_capability",
                        expected=[9, 0],
                        actual=list(capability),
                    )
                    min_vram = float(hardware.get("min_vram_mib") or 79000)
                    checks.require(
                        total_memory / float(1024**2) >= min_vram,
                        "torch.cuda.vram_mib",
                        expected=f">={min_vram}",
                        actual=round(total_memory / float(1024**2), 2),
                    )
                except Exception as exc:
                    checks.add(
                        "torch.cuda.device_details",
                        "fail",
                        expected="queryable H100",
                        actual="unavailable",
                        message=_clean_text(exc),
                    )

    if pynvml is not None:
        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            checks.add("nvml.init", "pass", expected="available", actual="available")

            driver = _decode(pynvml.nvmlSystemGetDriverVersion())
            report["software"]["nvidia_driver"] = driver
            count = int(pynvml.nvmlDeviceGetCount())
            checks.require(
                count == EXPECTED_GPU_COUNT,
                "nvml.device_count",
                expected=EXPECTED_GPU_COUNT,
                actual=count,
            )

            if count == EXPECTED_GPU_COUNT:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                nvml_name = _decode(pynvml.nvmlDeviceGetName(handle))
                gpu = report["gpus"][0] if report["gpus"] else {"index": 0}
                gpu["nvml_name"] = nvml_name
                try:
                    gpu["uuid"] = _decode(pynvml.nvmlDeviceGetUUID(handle))
                except Exception:
                    pass
                if not report["gpus"]:
                    report["gpus"].append(gpu)

                checks.require(
                    "H100" in nvml_name,
                    "nvml.gpu_name",
                    expected="contains H100",
                    actual=nvml_name,
                )

                try:
                    power_limit = (
                        float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0
                    )
                    power_usage = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    gpu["power_limit_watts"] = power_limit
                    gpu["power_usage_watts"] = power_usage
                    power_ok = (
                        abs(power_limit - EXPECTED_POWER_LIMIT_WATTS)
                        <= POWER_TOLERANCE_WATTS
                    )
                    checks.require(
                        power_ok,
                        "gpu.power_limit_watts",
                        expected=EXPECTED_POWER_LIMIT_WATTS,
                        actual=power_limit,
                        false_status="fail" if strict else "warn",
                        message=(
                            "paper-strict requires a 700 W H100 power limit"
                            if strict
                            else "portable run is allowed, but energy is not 700 W comparable"
                        ),
                    )
                    checks.require(
                        0.0 < power_usage <= power_limit + POWER_TOLERANCE_WATTS,
                        "gpu.power_usage_watts",
                        expected=f">0 and <={power_limit}",
                        actual=power_usage,
                        message="NVML returned an invalid instantaneous power reading",
                    )
                except Exception as exc:
                    checks.add(
                        "gpu.power",
                        "fail",
                        expected="queryable",
                        actual="unavailable",
                        message=_clean_text(exc),
                    )

                _query_ecc(pynvml, handle, checks, gpu)
                _query_row_remapper(pynvml, handle, checks, gpu, strict)
        except Exception as exc:
            checks.add(
                "nvml.runtime",
                "fail",
                expected="queryable",
                actual="unavailable",
                message=_clean_text(exc),
            )
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    configured_count = hardware.get("gpu_count")
    if configured_count is not None:
        checks.require(
            int(configured_count) == EXPECTED_GPU_COUNT,
            "profile.gpu_count",
            expected=EXPECTED_GPU_COUNT,
            actual=int(configured_count),
            message="this tool supports the paper's single-GPU contract only",
        )

    cpu_brand = _cpu_brand()
    memory_gib = _physical_memory_gib()
    report["host"] = {
        "cpu_brand": cpu_brand,
        "memory_gib": round(memory_gib, 2),
    }
    expected_cpu = hardware.get("cpu_name_contains")
    if expected_cpu:
        checks.require(
            str(expected_cpu) in cpu_brand,
            "host.cpu",
            expected=f"contains {expected_cpu}",
            actual=cpu_brand,
        )
    min_ram = hardware.get("min_ram_gib")
    if min_ram is not None:
        checks.require(
            memory_gib >= float(min_ram),
            "host.memory_gib",
            expected=f">={float(min_ram)}",
            actual=round(memory_gib, 2),
        )

    if bool(hardware.get("require_rapl")):
        rapl_root = str(
            profile.get("energy_overrides", {}).get(
                "rapl_root",
                "/sys/class/powercap",
            )
        )
        report["host"]["rapl_root"] = rapl_root
        checks.require(
            Path(rapl_root).exists()
            and any(Path(rapl_root).glob("intel-rapl:*")),
            "host.rapl",
            expected="readable Intel RAPL zones",
            actual=rapl_root,
        )

    report["checks"] = checks.items
    report["status"] = checks.status
    return report


def _error_report(profile_name: str, message: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "timestamp_utc": _utc_now(),
        "profile": {"name": profile_name, "strict": False},
        "status": "fail",
        "software": {},
        "host": {},
        "gpus": [],
        "checks": [
            {
                "name": "probe.configuration",
                "status": "fail",
                "expected": "valid configuration",
                "actual": "invalid",
                "message": _clean_text(message),
            }
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="Path to an energy-repro/profile/v1 JSON profile",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Compatibility flag; output is always one JSON document",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_profile(args.profile_file)
        report = probe_runtime(profile)
    except Exception as exc:
        profile_name = args.profile_file.name if args.profile_file else "h100-portable"
        report = _error_report(profile_name, _clean_text(exc))

    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
