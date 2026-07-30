from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_probe = _load_tool("runtime_probe")
fetch_assets = _load_tool("fetch_assets")
telemetry_smoke = _load_tool("telemetry_smoke")


class FakeNvml:
    NVML_FEATURE_ENABLED = 1
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_VOLATILE_ECC = 0
    NVML_AGGREGATE_ECC = 1

    def __init__(self, power_limit_mw: int):
        self.power_limit_mw = power_limit_mw

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        return None

    def nvmlSystemGetDriverVersion(self):
        return b"580.173.02"

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA H100 80GB HBM3"

    def nvmlDeviceGetUUID(self, handle):
        return b"GPU-test"

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return self.power_limit_mw

    def nvmlDeviceGetPowerUsage(self, handle):
        return 100_000

    def nvmlDeviceGetEccMode(self, handle):
        return self.NVML_FEATURE_ENABLED, self.NVML_FEATURE_ENABLED

    def nvmlDeviceGetTotalEccErrors(self, handle, error_type, counter_type):
        return 0

    def nvmlDeviceGetRemappedRows(self, handle):
        return 0, 0, False, False


def _fake_modules(power_limit_mw: int):
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "NVIDIA H100 80GB HBM3",
        get_device_capability=lambda index: (9, 0),
        get_device_properties=lambda index: types.SimpleNamespace(
            total_memory=80 * 1024**3
        ),
    )
    return {
        "torch": types.SimpleNamespace(
            __version__="2.6.0+cu124",
            version=types.SimpleNamespace(cuda="12.4"),
            cuda=cuda,
        ),
        "transformers": types.SimpleNamespace(__version__="4.51.3"),
        "pynvml": FakeNvml(power_limit_mw),
    }


def _module_loader(modules):
    def load(name):
        return modules[name]

    return load


class RuntimeProbeTests(unittest.TestCase):
    def _run(self, profile, power_limit_mw):
        modules = _fake_modules(power_limit_mw)
        with (
            mock.patch.object(
                runtime_probe.platform,
                "python_version",
                return_value="3.10.13",
            ),
            mock.patch.object(
                runtime_probe,
                "_cpu_brand",
                return_value="Test CPU",
            ),
            mock.patch.object(
                runtime_probe,
                "_physical_memory_gib",
                return_value=256.0,
            ),
        ):
            return runtime_probe.probe_runtime(
                profile,
                module_loader=_module_loader(modules),
            )

    def test_portable_non_700w_is_warning(self):
        report = self._run(runtime_probe.default_profile(), 650_000)
        self.assertEqual(report["status"], "warn")
        power = next(
            item
            for item in report["checks"]
            if item["name"] == "gpu.power_limit_watts"
        )
        self.assertEqual(power["status"], "warn")
        self.assertFalse(
            any(item["status"] == "fail" for item in report["checks"])
        )

    def test_strict_non_700w_fails(self):
        profile = runtime_probe.default_profile()
        profile["name"] = "paper-strict"
        report = self._run(profile, 650_000)
        self.assertEqual(report["status"], "fail")
        power = next(
            item
            for item in report["checks"]
            if item["name"] == "gpu.power_limit_watts"
        )
        self.assertEqual(power["status"], "fail")


class TelemetrySmokeTests(unittest.TestCase):
    def test_trapezoidal_power_integration(self):
        samples = [
            {"monotonic_seconds": 0.0, "power_watts": 100.0},
            {"monotonic_seconds": 1.0, "power_watts": 200.0},
            {"monotonic_seconds": 3.0, "power_watts": 300.0},
        ]
        self.assertEqual(telemetry_smoke.integrate_power(samples), 650.0)


class FetchAssetsTests(unittest.TestCase):
    def _write_lock(self, root: Path, revision: str = "a" * 40) -> Path:
        path = root / "assets.lock.json"
        path.write_text(
            json.dumps(
                {
                    "schema": fetch_assets.LOCK_SCHEMA,
                    "assets": {
                        "tiny": {
                            "kind": "model",
                            "repository": "example/tiny",
                            "revision": revision,
                            "expected_bytes": 7,
                        }
                    },
                    "sets": {"smoke": ["tiny"]},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_fetch_publishes_then_verify_only_uses_no_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = self._write_lock(root)
            cache_dir = root / "cache"
            calls = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                Path(kwargs["local_dir"], "weights").write_bytes(b"payload")

            receipt = fetch_assets.fetch_assets(
                lock_path=lock_path,
                cache_dir=cache_dir,
                set_names=["smoke"],
                snapshot_download_fn=fake_snapshot_download,
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], "a" * 40)
            object_path = Path(receipt["assets"][0]["path"])
            marker_path = object_path / fetch_assets.MARKER_NAME
            self.assertTrue(marker_path.is_file())
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(
                marker["tree_hash_algorithm"],
                fetch_assets.TREE_HASH_ALGORITHM,
            )
            self.assertRegex(marker["tree_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(receipt["assets"][0]["tree_sha256"], marker["tree_sha256"])

            def must_not_download(**kwargs):
                raise AssertionError("verify-only attempted a download")

            verified = fetch_assets.fetch_assets(
                lock_path=lock_path,
                cache_dir=cache_dir,
                set_names=["smoke"],
                verify_only=True,
                offline=True,
                snapshot_download_fn=must_not_download,
            )
            self.assertEqual(verified["status"], "pass")
            self.assertEqual(verified["assets"][0]["action"], "verified")

            weights_path = object_path / "weights"
            original_size = weights_path.stat().st_size
            weights_path.write_bytes(b"PAYLOAD")
            self.assertEqual(weights_path.stat().st_size, original_size)
            self.assertEqual(weights_path.stat().st_size, marker["total_bytes"])

            corrupted = fetch_assets.fetch_assets(
                lock_path=lock_path,
                cache_dir=cache_dir,
                set_names=["smoke"],
                verify_only=True,
                offline=True,
                snapshot_download_fn=must_not_download,
            )
            self.assertEqual(corrupted["status"], "fail")
            self.assertEqual(corrupted["assets"][0]["status"], "fail")
            self.assertIn("tree SHA-256", corrupted["assets"][0]["error"])

    def test_interrupted_download_reuses_stable_partial_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = self._write_lock(root)
            cache_dir = root / "cache"
            local_dirs = []

            def interrupted(**kwargs):
                local_dir = Path(kwargs["local_dir"])
                local_dirs.append(local_dir)
                (local_dir / "weights").write_bytes(b"part")
                raise RuntimeError("connection interrupted")

            failed = fetch_assets.fetch_assets(
                lock_path=lock_path,
                cache_dir=cache_dir,
                set_names=["smoke"],
                snapshot_download_fn=interrupted,
            )
            self.assertEqual(failed["status"], "fail")
            self.assertTrue((local_dirs[0] / "weights").is_file())

            def resumed(**kwargs):
                local_dir = Path(kwargs["local_dir"])
                local_dirs.append(local_dir)
                self.assertEqual(local_dir, local_dirs[0])
                self.assertEqual((local_dir / "weights").read_bytes(), b"part")
                (local_dir / "weights").write_bytes(b"payload")

            passed = fetch_assets.fetch_assets(
                lock_path=lock_path,
                cache_dir=cache_dir,
                set_names=["smoke"],
                snapshot_download_fn=resumed,
            )
            self.assertEqual(passed["status"], "pass")
            self.assertEqual(local_dirs[0], local_dirs[1])

    def test_full_revision_required_and_token_is_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short_lock = self._write_lock(root, revision="a3b76e8")
            with self.assertRaises(fetch_assets.AssetLockError):
                fetch_assets.load_asset_lock(short_lock)

            full_lock = self._write_lock(root, revision="b" * 40)
            secret = "hf_" + "do_not_print_this_secret"

            def failing_download(**kwargs):
                raise RuntimeError(f"Authorization Bearer {secret}")

            receipt = fetch_assets.fetch_assets(
                lock_path=full_lock,
                cache_dir=root / "cache-full",
                set_names=["smoke"],
                snapshot_download_fn=failing_download,
                environ={"HF_TOKEN": secret},
            )
            serialized = json.dumps(receipt)
            self.assertEqual(receipt["status"], "fail")
            self.assertNotIn(secret, serialized)
            self.assertIn("<redacted>", serialized)

    def test_disk_preflight_fails_before_downloader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = self._write_lock(root)
            called = False

            def must_not_download(**kwargs):
                nonlocal called
                called = True

            with mock.patch.object(
                fetch_assets.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(free=1),
            ):
                receipt = fetch_assets.fetch_assets(
                    lock_path=lock_path,
                    cache_dir=root / "cache",
                    set_names=["smoke"],
                    snapshot_download_fn=must_not_download,
                )
            self.assertEqual(receipt["status"], "fail")
            self.assertIn("insufficient free disk", receipt["error"])
            self.assertFalse(called)

    def test_disk_preflight_credits_resumable_partial_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = self._write_lock(root)
            cache = root / "cache"
            partial = cache / "partials" / fetch_assets.asset_cache_key(
                json.loads(lock_path.read_text(encoding="utf-8"))["assets"]["tiny"]
            )
            partial.mkdir(parents=True)
            (partial / "weights").write_bytes(b"part")

            def resumed(**kwargs):
                Path(kwargs["local_dir"], "weights").write_bytes(b"payload")

            with (
                mock.patch.object(fetch_assets, "_allocated_bytes", return_value=4),
                mock.patch.object(
                    fetch_assets.shutil,
                    "disk_usage",
                    return_value=types.SimpleNamespace(free=1024**3 + 3),
                ),
            ):
                receipt = fetch_assets.fetch_assets(
                    lock_path=lock_path,
                    cache_dir=cache,
                    set_names=["smoke"],
                    snapshot_download_fn=resumed,
                )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["disk_preflight"]["missing_expected_bytes"], 3)


if __name__ == "__main__":
    unittest.main()
