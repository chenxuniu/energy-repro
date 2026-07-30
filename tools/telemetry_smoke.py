#!/usr/bin/env python3
"""Bounded H100 workload that proves CUDA and NVML energy telemetry work together."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "energy-repro/telemetry-smoke/v1"


def integrate_power(samples: list[dict[str, float]]) -> float:
    energy_joules = 0.0
    for previous, current in zip(samples, samples[1:]):
        elapsed = current["monotonic_seconds"] - previous["monotonic_seconds"]
        energy_joules += elapsed * (previous["power_watts"] + current["power_watts"]) / 2.0
    return energy_joules


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_smoke(duration_seconds: float, sample_interval: float, matrix_size: int) -> dict[str, Any]:
    import pynvml
    import torch

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    samples: list[dict[str, float]] = []
    sample_errors: list[str] = []
    stop = threading.Event()
    pynvml.nvmlInit()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("telemetry smoke requires exactly one visible CUDA GPU")
        name = str(torch.cuda.get_device_name(0))
        if "H100" not in name:
            raise RuntimeError(f"telemetry smoke requires an H100, observed {name!r}")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def sample() -> None:
            while not stop.is_set():
                try:
                    item = {
                        "monotonic_seconds": time.monotonic(),
                        "power_watts": float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                    }
                    utilization = getattr(pynvml, "nvmlDeviceGetUtilizationRates", None)
                    if utilization is not None:
                        item["gpu_utilization_percent"] = float(utilization(handle).gpu)
                    samples.append(item)
                except Exception as exc:
                    sample_errors.append(str(exc)[:400])
                    stop.set()
                    return
                stop.wait(sample_interval)

        torch.manual_seed(0)
        left = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            product = torch.mm(left, right)
        torch.cuda.synchronize()

        sampler = threading.Thread(target=sample, name="nvml-sampler", daemon=True)
        sampler.start()
        workload_started = time.monotonic()
        deadline = workload_started + duration_seconds
        iterations = 0
        checksum = 0.0
        while time.monotonic() < deadline and not sample_errors:
            product = torch.mm(left, right)
            torch.cuda.synchronize()
            checksum = float(product[0, 0].item())
            iterations += 1
        workload_finished = time.monotonic()
        stop.set()
        sampler.join(timeout=max(2.0, sample_interval * 4.0))
        if sampler.is_alive():
            raise RuntimeError("NVML sampler did not stop")
        if sample_errors:
            raise RuntimeError(f"NVML sampling failed: {sample_errors[0]}")
        if len(samples) < 2:
            raise RuntimeError(f"NVML returned only {len(samples)} power samples")

        energy_joules = integrate_power(samples)
        relative_samples = [
            {
                **{key: value for key, value in item.items() if key != "monotonic_seconds"},
                "elapsed_seconds": round(item["monotonic_seconds"] - workload_started, 6),
            }
            for item in samples
        ]
        powers = [item["power_watts"] for item in samples]
        utilizations = [
            item["gpu_utilization_percent"]
            for item in samples
            if "gpu_utilization_percent" in item
        ]
        errors: list[str] = []
        if iterations < 1:
            errors.append("CUDA workload completed no iterations")
        if energy_joules <= 0.0:
            errors.append("integrated GPU energy is not positive")
        if not all(0.0 < value <= float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0 + 1.0 for value in powers):
            errors.append("NVML power samples are outside the valid power-limit range")
        if utilizations and max(utilizations) <= 0.0:
            errors.append("NVML never observed non-zero GPU utilization")

        return {
            "schema": SCHEMA,
            "status": "pass" if not errors else "fail",
            "started_at": started_utc,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                "pytorch": str(torch.__version__),
                "cuda_userspace": str(torch.version.cuda),
                "nvidia_driver": str(pynvml.nvmlSystemGetDriverVersion()),
            },
            "gpu": {
                "name": name,
                "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                "power_limit_watts": float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0,
            },
            "workload": {
                "dtype": "bfloat16",
                "matrix_size": matrix_size,
                "iterations": iterations,
                "checksum": checksum,
                "duration_seconds": round(workload_finished - workload_started, 6),
            },
            "telemetry": {
                "sample_interval_seconds": sample_interval,
                "sample_count": len(samples),
                "gpu_energy_joules": round(energy_joules, 6),
                "average_power_watts": round(energy_joules / (samples[-1]["monotonic_seconds"] - samples[0]["monotonic_seconds"]), 6),
                "samples": relative_samples,
            },
            "errors": errors,
            "total_wall_seconds": round(time.monotonic() - started, 6),
        }
    finally:
        stop.set()
        pynvml.nvmlShutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.duration_seconds < 3.0 or not 0.1 <= args.sample_interval <= 2.0 or args.matrix_size < 1024:
        parser.error("duration must be >=3s, interval 0.1-2s, and matrix size >=1024")
    try:
        report = run_smoke(args.duration_seconds, args.sample_interval, args.matrix_size)
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "fail",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc)[:1000],
        }
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
