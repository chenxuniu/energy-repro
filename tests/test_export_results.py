from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "export_results.py"
SPEC = importlib.util.spec_from_file_location("energy_export_results", TOOL)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_run(self, state: str = "succeeded") -> Path:
        self.fake_hf_token = "hf" + "_" + "abcdefghijklmnop"
        self.fake_github_token = "github_" + "pat_" + "this_should_never_escape"
        run_id = "kd-1b-pilot__h100-portable__seed-42__attempt-001__01234567"
        run = self.root / run_id
        (run / "resolved").mkdir(parents=True)
        (run / "logs").mkdir()
        (run / "energy" / "student").mkdir(parents=True)
        (run / "outputs" / "student" / "checkpoints" / "checkpoint-10").mkdir(
            parents=True
        )
        (run / "outputs" / "student" / "final_model").mkdir(parents=True)
        (run / "wandb").mkdir()
        (run / "assets").mkdir()

        plan_body = {
            "schema": "energy-repro/plan/v1",
            "profile": "h100-portable",
            "recipe": "kd-1b-pilot",
            "stage": "student",
            "seed": 42,
            "attempt": 1,
            "image": "energy-repro:h100-portable-a3b76e8",
        }
        plan_hash = hashlib.sha256(canonical(plan_body)).hexdigest()
        plan = {
            **plan_body,
            "plan_sha256": plan_hash,
            "run_id": run_id,
        }
        (run / "plan.json").write_bytes(canonical(plan))
        (run / "resolved" / "student.yaml").write_text(
            "output_dir: /mnt/energy-runs/secret\n"
            f"api_key: {self.fake_github_token}\n"
            "tokenizer: Qwen/Qwen2.5-1.5B\n",
            encoding="utf-8",
        )
        (run / "resolved" / "student.metadata.json").write_text(
            json.dumps(
                {
                    "run_root": "/mnt/energy-runs/run",
                    "hostname": "ipp2-0121",
                    "tokens_processed": 200,
                }
            ),
            encoding="utf-8",
        )
        (run / "logs" / "student.log").write_text(
            f"root@ipp2-0121: token={self.fake_hf_token}\n"
            "loading /mnt/energy-cache/model\n"
            "loss=1.25\n",
            encoding="utf-8",
        )
        (run / "energy" / "student" / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "total_gpu_energy_joules": 1234.5,
                    "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "stages": {
                        "student": {
                            "duration_seconds": 12.5,
                            "gpu_power_samples": [100.0 + index for index in range(40)],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "outputs" / "telemetry-smoke.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "gpu": {
                        "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                        "name": "NVIDIA H100 80GB HBM3",
                    },
                    "workload": {"iterations": 7},
                }
            ),
            encoding="utf-8",
        )
        (run / "outputs" / "student" / "checkpoints" / "checkpoint-10" / "model.pt").write_bytes(
            b"checkpoint-secret"
        )
        (run / "outputs" / "student" / "final_model" / "model.safetensors").write_bytes(
            b"final-model-secret"
        )
        (run / "wandb" / "wandb-history.jsonl").write_text(
            '{"password":"do-not-publish"}\n', encoding="utf-8"
        )
        (run / "assets" / "raw-dataset.jsonl").write_text(
            '{"prompt":"private training row"}\n', encoding="utf-8"
        )
        (run / "oversized-metrics.json").write_text(
            json.dumps({"blob": "x" * (module.DEFAULT_MAX_FILE_BYTES + 1)}),
            encoding="utf-8",
        )

        exit_code = 0 if state == "succeeded" else 5
        state_value = {
            "schema": "energy-repro/run-state/v1",
            "state": state,
            "exit_code": exit_code,
            "finished_at": "2026-07-30T12:00:00+00:00",
        }
        (run / "state.json").write_bytes(canonical(state_value))
        artifacts = []
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "state.json"}:
                artifacts.append(
                    {
                        "path": path.relative_to(run).as_posix(),
                        "type": "file",
                        "bytes": path.stat().st_size,
                        "sha256": sha(path),
                    }
                )
        manifest = {
            "schema": "energy-repro/manifest/v1",
            "run_id": run_id,
            "state": state,
            "exit_code": exit_code,
            "plan": plan,
            "hardware": {
                "hostname": "ipp2-0121",
                "gpus": [
                    {
                        "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "name": "NVIDIA H100 80GB HBM3",
                    }
                ],
            },
            "execution": {
                "argv": ["python", "/opt/energy/run_experiment.py"],
                "environment": {
                    "HF_TOKEN": self.fake_hf_token,
                    "WANDB_MODE": "offline",
                },
            },
            "artifacts": artifacts,
            "timestamps": {"finished_at": "2026-07-30T12:00:00+00:00"},
        }
        (run / "manifest.json").write_bytes(canonical(manifest))
        return run

    def test_public_export_redacts_and_excludes(self) -> None:
        run = self.make_run()
        output = self.root / "public-result"
        result = module.export_results(
            run, output, max_file_bytes=1024
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue((output / "SHA256SUMS").is_file())
        self.assertTrue((output / "resolved" / "student.yaml").is_file())
        self.assertFalse((output / "outputs").exists())
        self.assertFalse((output / "wandb").exists())
        self.assertFalse((output / "assets").exists())
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output.rglob("*")
            if path.is_file()
        )
        for forbidden in (
            "ipp2-0121",
            self.fake_hf_token,
            self.fake_github_token,
            "/mnt/energy",
            "/opt/energy",
            "GPU-aaaaaaaa",
            "checkpoint-secret",
            "final-model-secret",
            "private training row",
            "do-not-publish",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn("NVIDIA H100 80GB HBM3", combined)
        self.assertIn("total_gpu_energy_joules", combined)
        self.assertIn("tokens_processed", combined)
        self.assertNotIn("loading", combined)
        log_summary = json.loads(
            (output / "logs-summary.json").read_text(encoding="utf-8")
        )
        metrics = log_summary["logs"]["log-001"]["numeric_metrics"]
        self.assertTrue(
            any(
                metric["name"] == "loss" and metric["value"] == 1.25
                for metric in metrics
            )
        )
        receipt = json.loads((output / "export-receipt.json").read_text())
        self.assertEqual(receipt["source_manifest_sha256"], sha(run / "manifest.json"))
        self.assertGreaterEqual(
            receipt["excluded"]["counts_by_reason"]["prohibited-content"], 3
        )
        self.assertEqual(receipt["excluded"]["counts_by_reason"]["over-size-cap"], 1)

    def test_failed_terminal_run_is_exportable(self) -> None:
        run = self.make_run("failed")
        output = self.root / "failed-public"
        module.export_results(run, output)
        state = json.loads((output / "state.json").read_text())
        self.assertEqual(state["state"], "failed")

    def test_non_terminal_run_is_rejected(self) -> None:
        run = self.make_run()
        manifest_path = run / "manifest.json"
        state_path = run / "state.json"
        manifest = json.loads(manifest_path.read_text())
        state = json.loads(state_path.read_text())
        manifest["state"] = "running"
        manifest["exit_code"] = None
        state["state"] = "running"
        state["exit_code"] = None
        manifest_path.write_bytes(canonical(manifest))
        state_path.write_bytes(canonical(state))
        with self.assertRaisesRegex(module.ExportError, "Only terminal runs"):
            module.export_results(run, self.root / "out")

    def test_source_artifact_tampering_is_rejected(self) -> None:
        run = self.make_run()
        target = run / "logs" / "student.log"
        original = target.read_bytes()
        target.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(module.ExportError, "SHA-256 mismatch"):
            module.export_results(run, self.root / "out")

    def test_unrecorded_source_file_is_rejected(self) -> None:
        run = self.make_run()
        (run / "surprise.txt").write_text("unrecorded", encoding="utf-8")
        with self.assertRaisesRegex(module.ExportError, "inventory is not exact"):
            module.export_results(run, self.root / "out")

    def test_symlink_and_special_file_are_rejected(self) -> None:
        run = self.make_run()
        os.symlink(run / "plan.json", run / "linked-plan.json")
        with self.assertRaisesRegex(module.ExportError, "symlink"):
            module.export_results(run, self.root / "out")

        (run / "linked-plan.json").unlink()
        if hasattr(os, "mkfifo"):
            fifo = run / "pipe"
            os.mkfifo(fifo)
            try:
                with self.assertRaisesRegex(module.ExportError, "special file"):
                    module.export_results(run, self.root / "out")
            finally:
                fifo.unlink()

    def test_manifest_path_traversal_is_rejected(self) -> None:
        run = self.make_run()
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][0]["path"] = "../escape"
        manifest_path.write_bytes(canonical(manifest))
        with self.assertRaisesRegex(module.ExportError, "Unsafe artifact path"):
            module.export_results(run, self.root / "out")

    def test_directory_and_archive_are_deterministic(self) -> None:
        run = self.make_run()
        first = self.root / "first"
        second = self.root / "second"
        first_archive = self.root / "first.tar.gz"
        second_archive = self.root / "second.tar.gz"
        module.export_results(run, first, archive_path=first_archive)
        module.export_results(run, second, archive_path=second_archive)
        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)
        self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
        with tarfile.open(first_archive, "r:gz") as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.name == "result" or member.name.startswith("result/") for member in members))
            self.assertTrue(all(member.mtime == 0 for member in members))
            self.assertFalse(any("checkpoint" in member.name for member in members))

    def test_public_export_tampering_is_detected(self) -> None:
        run = self.make_run()
        output = self.root / "public"
        module.export_results(run, output)
        metrics = output / "metrics-summary.json"
        metrics.write_bytes(metrics.read_bytes() + b" ")
        with self.assertRaisesRegex(module.ExportError, "SHA-256 mismatch"):
            module.verify_export(output)

    def test_existing_destination_is_not_overwritten(self) -> None:
        run = self.make_run()
        output = self.root / "existing"
        output.mkdir()
        marker = output / "mine.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(module.ExportError, "already exists"):
            module.export_results(run, output)
        self.assertEqual(marker.read_text(), "keep")

    def add_manifest_artifact(self, run: Path, relative: str) -> None:
        path = run / relative
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"].append(
            {
                "path": relative,
                "type": "file",
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        )
        manifest["artifacts"].sort(key=lambda item: item["path"])
        manifest_path.write_bytes(canonical(manifest))

    def test_metric_json_never_copies_prompts_or_arbitrary_strings(self) -> None:
        run = self.make_run()
        marker = "FAKE_PRIVATE_PROMPT_DO_NOT_PUBLISH"
        path = run / "outputs" / "eval_results.json"
        path.write_text(
            json.dumps(
                {
                    "examples": [
                        {
                            "prompt": marker,
                            "response": "FAKE_PRIVATE_RESPONSE",
                            "eval_loss": 1.5,
                        }
                    ],
                    "customer_314159_loss": 2.5,
                }
            ),
            encoding="utf-8",
        )
        self.add_manifest_artifact(run, "outputs/eval_results.json")
        output = self.root / "public"
        module.export_results(run, output)
        summary_text = (output / "metrics-summary.json").read_text()
        self.assertNotIn(marker, summary_text)
        self.assertNotIn("FAKE_PRIVATE_RESPONSE", summary_text)
        self.assertNotIn("customer_314159", summary_text)
        summary = json.loads(summary_text)
        source = next(iter(summary["sources"].values()))
        self.assertIn("numeric_fields", source["summary"])
        self.assertTrue(
            any("loss" in name for name in source["summary"]["numeric_fields"])
        )

    def test_prefixed_secrets_and_network_identity_are_redacted(self) -> None:
        fake_secret = "FAKE_RANDOM_CREDENTIAL_123456"
        value = {
            "AWS_SECRET_ACCESS_KEY": fake_secret,
            "OPENAI_API_KEY": "FAKE_OPENAI_CREDENTIAL_456",
            "CLIENT_SECRET": "FAKE_CLIENT_CREDENTIAL_789",
            "remote_host": "ipp2-0121",
            "master_addr": "10.42.0.9",
            "mac_address": "aa:bb:cc:dd:ee:ff",
        }
        sanitized = module.sanitize_data(value)
        combined = json.dumps(sanitized)
        self.assertNotIn(fake_secret, combined)
        self.assertNotIn("ipp2-0121", combined)
        self.assertNotIn("10.42.0.9", combined)
        self.assertNotIn("aa:bb:cc:dd:ee:ff", combined)
        sanitized_text = module.sanitize_text(
            "Authorization: Basic RkFLRV9VU0VSOlBBU1M=\n"
            "auth_header: FAKE_CUSTOM_AUTH_123\n"
            "lease_node: ipp2-0121\n"
            "master_addr: 10.42.0.9\n"
            "mac_address: aa:bb:cc:dd:ee:ff\n"
        )
        for forbidden in (
            "RkFLRV9VU0VSOlBBU1M=",
            "FAKE_CUSTOM_AUTH_123",
            "ipp2-0121",
            "10.42.0.9",
            "aa:bb:cc:dd:ee:ff",
        ):
            self.assertNotIn(forbidden, sanitized_text)
        module._verify_public_text("sanitized.json", combined)

    def test_unmanifested_sync_receipt_cannot_influence_public_metrics(self) -> None:
        run = self.make_run()
        receipt = run / "sync" / "receipts" / "result.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps({"unmanifested_metric_loss": 987654321}),
            encoding="utf-8",
        )
        output = self.root / "public"
        module.export_results(run, output)
        summary = json.loads((output / "metrics-summary.json").read_text())
        combined = json.dumps(summary)
        self.assertNotIn("987654321", combined)
        export_receipt = json.loads((output / "export-receipt.json").read_text())
        self.assertEqual(
            export_receipt["excluded"]["counts_by_reason"][
                "unmanifested-sync-receipt"
            ],
            1,
        )

    def test_archive_must_not_be_nested_in_output(self) -> None:
        run = self.make_run()
        output = self.root / "public"
        with self.assertRaisesRegex(module.ExportError, "must not contain"):
            module.export_results(
                run,
                output,
                archive_path=output / "bundle.tar.gz",
            )
        self.assertFalse(output.exists())

    def test_source_replacement_after_scan_is_rejected(self) -> None:
        run = self.make_run()
        files = module._scan_directory(run)
        source = files["logs/student.log"]
        source.path.unlink()
        os.symlink(run / "plan.json", source.path)
        with self.assertRaisesRegex(module.ExportError, "safely open"):
            module._read_source_text(source)

    def test_private_image_reference_and_failure_detail_are_not_public(self) -> None:
        run = self.make_run()
        private_ref = (
            "registry.private.example/team/energy@sha256:" + ("a" * 64)
        )
        prompt_marker = "FAKE_PRIVATE_PROMPT_FROM_FAILURE"
        plan_path = run / "plan.json"
        manifest_path = run / "manifest.json"
        plan = json.loads(plan_path.read_text())
        plan["image"] = private_ref
        plan_payload = dict(plan)
        plan_payload.pop("plan_sha256")
        plan_payload.pop("run_id")
        plan["plan_sha256"] = hashlib.sha256(canonical(plan_payload)).hexdigest()
        plan_path.write_bytes(canonical(plan))

        manifest = json.loads(manifest_path.read_text())
        manifest["plan"] = plan
        manifest["execution"]["argv"].append(private_ref)
        manifest["image_verification"] = {
            "ref": private_ref,
            "repo_digests": [private_ref],
            "id": "sha256:" + ("b" * 64),
        }
        manifest["failure"] = {
            "phase": "execution",
            "subprocess_exit_code": 5,
            "detail": f"trainer rejected prompt: {prompt_marker}",
        }
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "plan.json":
                artifact["bytes"] = plan_path.stat().st_size
                artifact["sha256"] = sha(plan_path)
        manifest_path.write_bytes(canonical(manifest))

        output = self.root / "public"
        module.export_results(run, output)
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output.rglob("*")
            if path.is_file()
        )
        self.assertNotIn("registry.private.example", combined)
        self.assertNotIn(prompt_marker, combined)
        self.assertIn("<redacted-oci-reference>@sha256:" + ("a" * 64), combined)
        public_manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(
            public_manifest["failure"]["detail"],
            "<redacted-failure-detail>",
        )


if __name__ == "__main__":
    unittest.main()
