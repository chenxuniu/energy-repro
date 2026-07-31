from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "repro"


def load_repro_module():
    loader = importlib.machinery.SourceFileLoader("energy_repro_cli", str(REPRO))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def invoke(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [str(REPRO), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged,
        check=False,
    )


class BundleTests(unittest.TestCase):
    def test_all_json_files_parse(self) -> None:
        for path in sorted(ROOT.rglob("*.json")):
            with self.subTest(path=path):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_all_revisions_are_immutable(self) -> None:
        lock = json.loads((ROOT / "assets.lock.json").read_text(encoding="utf-8"))
        for asset_id, asset in lock["assets"].items():
            with self.subTest(asset=asset_id):
                self.assertRegex(asset["revision"], r"^[0-9a-f]{40}$")
        upstream = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
        self.assertRegex(upstream["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(upstream["archive_sha256"], r"^[0-9a-f]{64}$")

    def test_base_image_is_digest_pinned(self) -> None:
        lock = json.loads((ROOT / "image" / "image.lock.json").read_text(encoding="utf-8"))
        self.assertRegex(lock["base_image"], r"@sha256:[0-9a-f]{64}$")
        dockerfile = (ROOT / "image" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(lock["base_image"], dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("--only-binary=:all:", dockerfile)
        self.assertIn("--no-cache-dir", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cu124", dockerfile)
        self.assertIn("io.energy-repro.build-context-sha256", dockerfile)
        self.assertIn("io.energy-repro.dependency-lock-sha256", dockerfile)
        self.assertNotIn("apt-get", dockerfile)

    def test_locked_torch_wheel_has_a_hash(self) -> None:
        requirements = (ROOT / "locks" / "core-py310-cu124.txt").read_text(encoding="utf-8")
        self.assertRegex(
            requirements,
            r"torch==2\.6\.0\+cu124[^\n]*\\\n\s+--hash=sha256:[0-9a-f]{64}",
        )

    def test_host_setup_is_guarded_and_does_not_install_a_driver(self) -> None:
        path = ROOT / "scripts" / "setup_docker_nvidia_ubuntu.sh"
        script = path.read_text(encoding="utf-8")
        self.assertTrue(path.stat().st_mode & 0o111)
        self.assertIn("nvidia-smi", script)
        self.assertIn("26.04", script)
        self.assertIn('NVIDIA_CONTAINER_TOOLKIT_VERSION="1.19.1-1"', script)
        self.assertIn("conflicting/provider-managed packages detected", script)
        self.assertIn("ENERGY_REPRO_ALLOW_PROVIDER_DOCKER_MUTATION", script)
        self.assertIn(
            "no host packages or configuration were changed",
            script,
        )
        self.assertNotIn("nvidia-driver-", script)
        self.assertNotIn("apt-get remove", script)

    def test_short_lease_docs_use_persistent_unique_attempt_and_python3(self) -> None:
        for name in ("PILOT.md", "QUICKSTART.md"):
            with self.subTest(document=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("/root/energy-repro.env", text)
                self.assertIn("ENERGY_REPRO_ATTEMPT", text)
                self.assertNotIn("--attempt 1", text)
                self.assertNotIn("python tools/export_results.py", text)
        results_text = (ROOT / "results" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("python tools/export_results.py", results_text)


class DryRunTests(unittest.TestCase):
    def test_dry_runs_are_hermetic_and_do_not_create_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cache = base / "cache"
            runs = base / "runs"
            state = base / "state"
            common = (
                "--cache-dir",
                str(cache),
                "--runs-dir",
                str(runs),
                "--state-dir",
                str(state),
                "--dry-run",
                "--json",
            )
            cases = [
                ("doctor", "--profile", "h100-portable", *common),
                ("bootstrap", "--profile", "h100-portable", "--build-image", *common),
                ("fetch-assets", "--profile", "h100-portable", "--set", "kd-1b", *common),
                ("smoke", "--profile", "h100-portable", *common),
                (
                    "run",
                    "--profile",
                    "h100-portable",
                    "--recipe",
                    "kd-1b",
                    "--stage",
                    "preprocess",
                    *common,
                ),
                (
                    "run",
                    "--profile",
                    "h100-portable",
                    "--recipe",
                    "kd-1b-pilot",
                    "--stage",
                    "preprocess",
                    *common,
                ),
                (
                    "export-results",
                    "fixture-run",
                    "--output",
                    str(base / "public-result"),
                    "--archive",
                    str(base / "public-result.tar.gz"),
                    *common,
                ),
            ]
            poisoned = base / "bin"
            poisoned.mkdir()
            for executable in ("docker", "podman", "git", "nvidia-smi", "curl", "ssh"):
                path = poisoned / executable
                path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
                path.chmod(0o755)
            env = {"PATH": f"{poisoned}{os.pathsep}{os.environ.get('PATH', '')}"}
            for case in cases:
                with self.subTest(command=case[0]):
                    result = invoke(*case, env=env)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIsInstance(json.loads(result.stdout), dict)
                    self.assertFalse(cache.exists())
                    self.assertFalse(runs.exists())
                    self.assertFalse(state.exists())

    def test_run_id_is_deterministic(self) -> None:
        args = (
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "kd-1b",
            "--stage",
            "teacher",
            "--seed",
            "42",
            "--attempt",
            "7",
            "--dry-run",
            "--json",
        )
        first = invoke(*args)
        second = invoke(*args)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(json.loads(first.stdout)["run_id"], json.loads(second.stdout)["run_id"])

    def test_planned_containers_use_host_identity_and_writable_runtime_cache(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "kd-1b",
            "--stage",
            "preprocess",
            "--ack-known-deviations",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        for argv in (payload["resolver_argv"], payload["execution_argv"]):
            self.assertIn("--user", argv)
            self.assertIn("HF_HOME=/cache/runtime/huggingface", argv)
            self.assertTrue(
                any("/cache/runtime" in item for item in argv if isinstance(item, str))
            )

    def test_telemetry_smoke_uses_capsule_owned_bounded_probe(self) -> None:
        result = invoke("smoke", "--profile", "h100-portable", "--dry-run", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(result.stdout)["argv"]
        self.assertIn("/opt/energy-repro/tools/telemetry_smoke.py", argv)
        self.assertNotIn("distill_bench.core.prerun", argv)
        self.assertIn("--duration-seconds", argv)

    def test_strict_training_is_blocked_even_in_dry_run(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "paper-strict",
            "--recipe",
            "kd-1b",
            "--stage",
            "student",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 4)
        payload = json.loads(result.stdout)
        self.assertIn("blocks stage", payload["error"])

    def test_pilot_recipe_uses_capsule_preprocessor_and_non_paper_label(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "kd-1b-pilot",
            "--stage",
            "preprocess",
            "--ack-known-deviations",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["pilot"])
        self.assertEqual(
            payload["result_classification"],
            "workflow-pilot-not-paper-reproduction",
        )
        self.assertEqual(
            payload["plan"]["result_classification"],
            "workflow-pilot-not-paper-reproduction",
        )
        self.assertIn(
            "/opt/energy-repro/tools/pilot_preprocess.py",
            payload["execution_argv"],
        )
        self.assertNotIn("tulu_preprocess_dataset", payload["execution_argv"])

    def test_pilot_recipe_rejects_nonportable_profile(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "upstream-exact",
            "--recipe",
            "kd-1b-pilot",
            "--stage",
            "preprocess",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 4)
        self.assertIn("allowed only with profiles", json.loads(result.stdout)["error"])

    def test_pilot_recipes_are_bounded_and_explicitly_classified(self) -> None:
        for recipe_name in (
            "kd-1b-pilot",
            "sft-1b-pilot",
            "sft-original-1b-pilot",
        ):
            with self.subTest(recipe=recipe_name):
                recipe = json.loads(
                    (ROOT / "recipes" / f"{recipe_name}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    recipe["classification"],
                    "workflow-pilot-not-paper-reproduction",
                )
                self.assertEqual(recipe["allowed_profiles"], ["h100-portable"])
                self.assertEqual(recipe["preprocess_tool"], "pilot_preprocess.py")
                if "teacher" in recipe.get(
                    "allowed_stages",
                    ["preprocess", "teacher", "student"],
                ):
                    self.assertEqual(recipe["teacher_tool"], "pilot_teacher.py")
                pilot = recipe["config_overrides"]["energy_repro_pilot"]
                self.assertGreater(pilot["train_examples"], 0)
                self.assertGreater(pilot["eval_examples"], 0)
                self.assertLessEqual(
                    pilot["train_examples"] + pilot["eval_examples"],
                    128,
                )
                self.assertFalse(
                    recipe["config_overrides"]["experiment"]["debug_mode"]
                )
                training = recipe["config_overrides"]["training"]
                self.assertEqual(training["num_epochs"], 1)
                self.assertEqual(training["gradient_accumulation_steps"], 1)
                self.assertGreaterEqual(
                    min(training["eval_steps"], training["save_steps"]),
                    1000,
                )

    def test_original_sft_pilot_rejects_teacher_stage(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "sft-original-1b-pilot",
            "--stage",
            "teacher",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 4)
        self.assertIn("allows only stages", json.loads(result.stdout)["error"])

    def test_pilot_teacher_stage_uses_seeded_capsule_wrapper(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "kd-1b-pilot",
            "--stage",
            "teacher",
            "--ack-known-deviations",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(result.stdout)["execution_argv"]
        self.assertIn("/opt/energy-repro/tools/pilot_teacher.py", argv)
        self.assertNotIn("distill_bench.data.logit_caching", argv)

    def test_pilot_stage_disables_wandb_at_the_process_boundary(self) -> None:
        for recipe_name, stage in (
            ("kd-1b-pilot", "student"),
            ("sft-1b-pilot", "teacher"),
            ("sft-original-1b-pilot", "student"),
        ):
            with self.subTest(recipe=recipe_name, stage=stage):
                result = invoke(
                    "run",
                    "--profile",
                    "h100-portable",
                    "--recipe",
                    recipe_name,
                    "--stage",
                    stage,
                    "--ack-known-deviations",
                    "--dry-run",
                    "--json",
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr,
                )
                argv = json.loads(result.stdout)["execution_argv"]
                self.assertIn("WANDB_MODE=disabled", argv)
                self.assertNotIn("WANDB_MODE=offline", argv)

    def test_nonpilot_stage_preserves_offline_wandb_mode(self) -> None:
        result = invoke(
            "run",
            "--profile",
            "h100-portable",
            "--recipe",
            "kd-1b",
            "--stage",
            "student",
            "--ack-known-deviations",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(result.stdout)["execution_argv"]
        self.assertIn("WANDB_MODE=offline", argv)
        self.assertNotIn("WANDB_MODE=disabled", argv)

    def test_manifest_environment_matches_planned_pilot_wandb_mode(self) -> None:
        module = load_repro_module()
        profile = json.loads(
            (ROOT / "profiles" / "h100-portable.json").read_text(
                encoding="utf-8"
            )
        )
        pilot_manifest = module.initial_manifest(
            {"run_id": "pilot", "pilot": True},
            profile,
            "planned",
        )
        regular_manifest = module.initial_manifest(
            {"run_id": "regular", "pilot": False},
            profile,
            "planned",
        )
        self.assertEqual(
            pilot_manifest["execution"]["environment"]["WANDB_MODE"],
            "disabled",
        )
        self.assertEqual(
            regular_manifest["execution"]["environment"]["WANDB_MODE"],
            "offline",
        )

    def test_secret_value_is_never_rendered(self) -> None:
        secret = "hf_" + "abcdefghijklmnopqrstuvwxyz" + "123456"
        result = invoke(
            "fetch-assets",
            "--set",
            "kd-1b",
            "--dry-run",
            "--json",
            env={"HF_TOKEN": secret},
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(secret, result.stdout)

    def test_export_results_dry_run_is_host_only_and_describes_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = invoke(
                "export-results",
                "fixture-run",
                "--runs-dir",
                str(root / "runs"),
                "--output",
                str(root / "public"),
                "--archive",
                str(root / "public.tar.gz"),
                "--dry-run",
                "--json",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "PLANNED")
            self.assertIn("no models", payload["public_policy"])
            self.assertFalse((root / "runs").exists())
            self.assertFalse((root / "public").exists())
            self.assertFalse((root / "public.tar.gz").exists())

    def test_export_results_rejects_nonpositive_file_cap(self) -> None:
        result = invoke(
            "export-results",
            "fixture-run",
            "--output",
            "/tmp/public-result",
            "--max-file-bytes",
            "0",
            "--dry-run",
            "--json",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be positive", json.loads(result.stdout)["error"])


class LocalArtifactTests(unittest.TestCase):
    def test_execution_provenance_binds_all_capsule_tools(self) -> None:
        module = load_repro_module()
        provenance = module.execution_provenance(
            {
                "labels": {
                    "io.energy-repro.build-context-sha256": "a" * 64,
                }
            }
        )
        expected = {
            path.name
            for path in (ROOT / "tools").glob("*.py")
        }
        self.assertEqual(set(provenance["capsule_tools"]), expected)
        for digest in provenance["capsule_tools"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_signal_exit_codes_are_recorded_as_interrupted(self) -> None:
        module = load_repro_module()
        self.assertEqual(module.execution_outcome(-2), ("interrupted", 130))
        self.assertEqual(module.execution_outcome(130), ("interrupted", 130))
        self.assertEqual(module.execution_outcome(-15), ("interrupted", 143))
        self.assertEqual(module.execution_outcome(143), ("interrupted", 143))
        self.assertEqual(module.execution_outcome(7), ("failed", module.EXIT_EXEC))

    def test_failed_preflight_manifest_remains_auditable(self) -> None:
        module = load_repro_module()
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "fixture-run"
            run_root.mkdir()
            (run_root / "plan.json").write_text("{}\n", encoding="utf-8")
            manifest = {
                "state": "planned",
                "timestamps": {"created_at": "fixture"},
                "artifacts": [],
                "exit_code": None,
            }
            module.finalize_run_manifest(
                run_root,
                manifest,
                "failed",
                module.EXIT_PREREQ,
                phase="runtime-probe",
            )
            saved_manifest = json.loads(
                (run_root / "manifest.json").read_text(encoding="utf-8")
            )
            saved_state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["state"], "failed")
            self.assertEqual(saved_manifest["exit_code"], module.EXIT_PREREQ)
            self.assertEqual(saved_manifest["artifacts"][0]["path"], "plan.json")
            self.assertEqual(saved_state["state"], "failed")
            self.assertEqual(saved_state["phase"], "runtime-probe")

    def test_image_provenance_labels_are_enforced(self) -> None:
        module = load_repro_module()
        upstream = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
        record = [
            {
                "Id": "sha256:" + "1" * 64,
                "RepoDigests": [],
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": upstream["commit"],
                        "io.energy-repro.source-sha256": upstream["archive_sha256"],
                        "io.energy-repro.build-context-sha256": "2" * 64,
                        "io.energy-repro.dependency-lock-sha256": hashlib.sha256(
                            (ROOT / "locks" / "core-py310-cu124.txt").read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ]
        completed = subprocess.CompletedProcess(
            ["docker", "image", "inspect"],
            0,
            stdout=json.dumps(record),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary:
            original = module.run_capture
            module.run_capture = lambda *args, **kwargs: completed
            try:
                result = module.verify_local_image(
                    types.SimpleNamespace(runtime="docker"),
                    Path(temporary),
                    "energy-repro:test",
                    check_recorded=False,
                    expected_build_context_sha256="2" * 64,
                )
                self.assertEqual(result["id"], "sha256:" + "1" * 64)
                record[0]["Config"]["Labels"]["org.opencontainers.image.revision"] = "0" * 40
                completed.stdout = json.dumps(record)
                with self.assertRaisesRegex(module.ReproError, "provenance"):
                    module.verify_local_image(
                        types.SimpleNamespace(runtime="docker"),
                        Path(temporary),
                        "energy-repro:test",
                        check_recorded=False,
                        expected_build_context_sha256="2" * 64,
                    )
            finally:
                module.run_capture = original

    def test_build_context_is_idempotent_and_detects_tampering(self) -> None:
        module = load_repro_module()
        source = ROOT.parent / "tmp" / "Energy-a3b76e8"
        if not source.is_dir():
            self.skipTest("pinned upstream audit checkout is not available")
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            first, first_manifest = module.prepare_build_context(cache, source)
            second, second_manifest = module.prepare_build_context(cache, source)
            self.assertEqual(first, second)
            self.assertEqual(first_manifest, second_manifest)
            copied_readme = first / "upstream" / "README.md"
            copied_readme.write_bytes(b"x" * copied_readme.stat().st_size)
            with self.assertRaisesRegex(module.ReproError, "modified"):
                module.prepare_build_context(cache, source)

    def test_derived_receipt_is_written_only_for_a_real_target(self) -> None:
        module = load_repro_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            artifact = cache / "derived" / "kd-1b" / ("a" * 16) / "preprocessed"
            artifact.mkdir(parents=True)
            (artifact / "dataset.arrow").write_bytes(b"payload")
            receipt = cache / "derived" / "kd-1b" / ("a" * 16) / ".receipts" / "preprocessed.json"
            metadata = {
                "derived_fingerprint": "a" * 64,
                "derived_artifacts": {
                    "preprocessed": {
                        "path": f"/cache/{artifact.relative_to(cache).as_posix()}",
                        "receipt_path": f"/cache/{receipt.relative_to(cache).as_posix()}",
                        "target_for_stage": True,
                    }
                },
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            plan = {
                "run_id": "fixture-run",
                "plan_sha256": "b" * 64,
                "stage": "preprocess",
                "upstream_commit": "c" * 40,
                "asset_lock_sha256": "d" * 64,
            }
            published = module.publish_derived_receipts(metadata_path, cache, plan)
            self.assertEqual(published[0]["artifact"], "preprocessed")
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], "energy-repro/derived-receipt/v1")
            self.assertEqual(saved["fingerprint"], "a" * 64)
            self.assertEqual(saved["status"], "succeeded")
            status = module.live_derived_status(
                "preprocessed",
                {
                    "path": f"/cache/{artifact.relative_to(cache).as_posix()}",
                    "receipt_path": f"/cache/{receipt.relative_to(cache).as_posix()}",
                },
                "a" * 64,
                cache,
            )
            self.assertEqual(status["status"], "valid")
            metadata["derived_artifacts"]["preprocessed"]["target_for_stage"] = True
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            reuse = module.inspect_derived_production(
                metadata_path,
                cache,
                {"pipeline": "kd"},
                "preprocess",
                "a" * 64,
            )
            self.assertEqual(reuse["action"], "reuse")
            (artifact / "dataset.arrow").write_bytes(b"PAYLOAD")
            self.assertEqual(
                module.live_derived_status(
                    "preprocessed",
                    {
                        "path": f"/cache/{artifact.relative_to(cache).as_posix()}",
                        "receipt_path": f"/cache/{receipt.relative_to(cache).as_posix()}",
                    },
                    "a" * 64,
                    cache,
                )["status"],
                "invalid",
            )

    def test_partial_kd_workspace_is_blocked_before_execution(self) -> None:
        module = load_repro_module()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            root = cache / "derived" / "kd-1b" / ("a" * 16)
            chunk = root / "logprob_cache" / "teacher_logprobs_train" / "chunk_0"
            chunk.mkdir(parents=True)
            (chunk / "data.arrow").write_bytes(b"partial")
            metadata = {
                "derived_fingerprint": "a" * 64,
                "derived_artifacts": {
                    "teacher_logprobs": {
                        "path": "/cache/derived/kd-1b/" + "a" * 16 + "/logprob_cache/teacher_logprobs",
                        "receipt_path": "/cache/derived/kd-1b/" + "a" * 16 + "/.receipts/teacher_logprobs.json",
                        "target_for_stage": True,
                    },
                    "logprob_cache": {
                        "path": "/cache/derived/kd-1b/" + "a" * 16 + "/logprob_cache",
                        "receipt_path": "/cache/derived/kd-1b/" + "a" * 16 + "/.receipts/logprob_cache.json",
                        "target_for_stage": False,
                    },
                },
            }
            metadata_path = Path(temporary) / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = module.inspect_derived_production(
                metadata_path,
                cache,
                {"pipeline": "kd"},
                "teacher",
                "a" * 64,
            )
            self.assertEqual(result["action"], "block")
            self.assertIn("partial KD", result["errors"][0]["error"])

    def test_sync_is_incremental_and_never_deletes_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runs = base / "runs"
            run_id = "fixture-run"
            source = runs / run_id
            source.mkdir(parents=True)
            payload = source / "result.txt"
            payload.write_text("result\n", encoding="utf-8")
            manifest = {
                "schema": "energy-repro/manifest/v1",
                "run_id": run_id,
                "state": "succeeded",
                "artifacts": [
                    {
                        "path": "result.txt",
                        "type": "file",
                        "bytes": payload.stat().st_size,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ],
            }
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            target_root = base / "persistent"
            first = invoke(
                "sync",
                run_id,
                "--runs-dir",
                str(runs),
                "--to",
                str(target_root),
                "--verify",
                "--json",
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            extra = target_root / run_id / "keep-me.txt"
            extra.write_text("do not delete\n", encoding="utf-8")
            second = invoke(
                "sync",
                run_id,
                "--runs-dir",
                str(runs),
                "--to",
                str(target_root),
                "--verify",
                "--json",
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertTrue(extra.is_file())
            self.assertEqual(json.loads(second.stdout)["copied_files"], 0)

    def test_running_sync_receipt_is_explicitly_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runs = base / "runs"
            run_id = "running-fixture"
            source = runs / run_id
            source.mkdir(parents=True)
            (source / "checkpoint.bin").write_bytes(b"stable checkpoint")
            (source / "state.json").write_text(
                json.dumps(
                    {
                        "schema": "energy-repro/run-state/v1",
                        "state": "running",
                    }
                ),
                encoding="utf-8",
            )
            result = invoke(
                "sync",
                run_id,
                "--runs-dir",
                str(runs),
                "--to",
                str(base / "persistent"),
                "--json",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "PARTIAL")
            self.assertEqual(payload["receipt"]["status"], "partial")
            self.assertTrue(payload["receipt"]["verified"])

    def test_manifest_detects_tampering(self) -> None:
        module = load_repro_module()
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            profile = json.loads(
                (ROOT / "profiles" / "h100-portable.json").read_text(encoding="utf-8")
            )
            recipe = json.loads(
                (ROOT / "recipes" / "kd-1b.json").read_text(encoding="utf-8")
            )
            plan, run_id = module.deterministic_plan(
                profile,
                recipe,
                "preprocess",
                42,
                1,
                "energy-repro:test",
            )
            root = runs / run_id
            root.mkdir(parents=True)
            payload = root / "artifact.bin"
            payload.write_bytes(b"original")
            module.atomic_json(root / "plan.json", plan)
            module.write_run_state(root, "succeeded", exit_code=0, finished_at="fixture")
            manifest = module.initial_manifest(plan, profile, "succeeded")
            manifest["exit_code"] = 0
            manifest["timestamps"]["finished_at"] = "fixture"
            manifest["artifacts"] = module.artifact_inventory(root)
            module.atomic_json(root / "manifest.json", manifest)
            good = invoke("manifest", run_id, "--runs-dir", str(runs), "--verify", "--json")
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            payload.write_bytes(b"tampered")
            bad = invoke("manifest", run_id, "--runs-dir", str(runs), "--verify", "--json")
            self.assertEqual(bad.returncode, 4)
            self.assertIn("mismatch", bad.stdout)


if __name__ == "__main__":
    unittest.main()
