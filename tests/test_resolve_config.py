from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    yaml = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "resolve_config.py"


@unittest.skipIf(yaml is None, "PyYAML is not installed in the host environment")
class ResolveConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("resolve_config", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache = self.root / "cache"
        self.run_root = self.root / "run"

        self.repositories = {
            "teacher": "example/teacher",
            "student": "example/student",
            "tokenizer": "example/tokenizer",
            "dataset": "example/dataset",
        }
        self.revisions = {
            "teacher": "1" * 40,
            "student": "2" * 40,
            "tokenizer": "3" * 40,
            "dataset": "4" * 40,
        }
        self.asset_ids = {
            "teacher": "teacher-model",
            "student": "student-model",
            "tokenizer": "tokenizer",
            "dataset": "dataset",
        }

        self.base_path = self.root / "base.yaml"
        self.experiment_path = self.root / "experiment.yaml"
        self.profile_path = self.root / "profile.json"
        self.recipe_path = self.root / "recipe.json"
        self.assets_path = self.root / "assets.lock.json"
        self.provenance_path = self.root / "execution-provenance.json"
        self.provenance_path.write_text(
            json.dumps(
                {
                    "schema": self.module.EXECUTION_PROVENANCE_SCHEMA,
                    "upstream_commit": "a" * 40,
                    "upstream_archive_sha256": "b" * 64,
                    "dependency_lock_sha256": "c" * 64,
                    "asset_lock_sha256": "d" * 64,
                    "build_context_sha256": "e" * 64,
                    "resolver_tool_sha256": "f" * 64,
                }
            ),
            encoding="utf-8",
        )

        self.base_path.write_text(
            """
data:
  dataset_choice: tulu
  dataset_name: example/dataset
  dataset_path: /path/to/preprocessed
  tokenizer_name: example/tokenizer
  datasets:
    tulu:
      dataset_name: example/dataset
      dataset_path: /path/to/tulu
    math:
      dataset_name: example/unused
      dataset_path: /path/to/math
energy:
  track_cpu: true
  total_energy_policy: measured
experiment:
  seed: 42
model:
  teacher: example/teacher
output:
  output_dir: /path/to/output
  run_dir: ./logs
training:
  batch_size: 4
  nested:
    from_base: true
    replaced: base
wandb:
  enabled: true
  api_key: do-not-write-this
""".lstrip(),
            encoding="utf-8",
        )
        self.experiment_path.write_text(
            """
pipeline: kd
model:
  student: example/student
training:
  nested:
    from_experiment: true
    replaced: experiment
""".lstrip(),
            encoding="utf-8",
        )
        self.profile_path.write_text(
            json.dumps(
                {
                    "schema": "energy-repro/profile/v1",
                    "name": "h100-portable",
                    "source_mode": "upstream-exact",
                    "compliance_label": "portable",
                    "energy_overrides": {
                        "track_cpu": True,
                        "total_energy_policy": "measured",
                    },
                }
            ),
            encoding="utf-8",
        )
        self.recipe_path.write_text(
            json.dumps(
                {
                    "schema": "energy-repro/recipe/v1",
                    "name": "kd-test",
                    "pipeline": "kd",
                    "teacher": self.repositories["teacher"],
                    "student": self.repositories["student"],
                    "dataset": self.repositories["dataset"],
                    "asset_set": "kd-test",
                }
            ),
            encoding="utf-8",
        )
        assets = {}
        for role in ("teacher", "student", "tokenizer", "dataset"):
            asset_id = self.asset_ids[role]
            assets[asset_id] = {
                "kind": "dataset" if role == "dataset" else "model",
                "repository": self.repositories[role],
                "revision": self.revisions[role],
            }
        self.assets_path.write_text(
            json.dumps(
                {
                    "schema": "energy-repro/assets-lock/v1",
                    "assets": assets,
                    "sets": {"kd-test": list(assets)},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _snapshot_path(self, role: str) -> Path:
        asset = json.loads(self.assets_path.read_text(encoding="utf-8"))["assets"][
            self.asset_ids[role]
        ]
        return (
            self.cache
            / "assets"
            / "objects"
            / self.module._asset_cache_key(asset)
        )

    def _create_snapshot(self, role: str) -> Path:
        asset = json.loads(self.assets_path.read_text(encoding="utf-8"))["assets"][
            self.asset_ids[role]
        ]
        cache_key = self.module._asset_cache_key(asset)
        path = self._snapshot_path(role)
        path.mkdir(parents=True, exist_ok=True)
        payload = f"payload-for-{role}\n".encode("utf-8")
        (path / "payload.bin").write_bytes(payload)
        file_count, total_bytes, tree_sha256 = self.module._scan_asset_payload(path)
        marker = {
            "schema": self.module.ASSET_OBJECT_SCHEMA,
            "cache_key": cache_key,
            "descriptor": self.module._asset_descriptor(asset),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "tree_hash_algorithm": self.module.TREE_HASH_ALGORITHM,
            "tree_sha256": tree_sha256,
        }
        (path / self.module.ASSET_MARKER_NAME).write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path

    def _resolve(
        self,
        *,
        stage: str,
        formal: bool = True,
        seed: int | None = None,
    ):
        return self.module.resolve_config(
            base_path=self.base_path,
            experiment_path=self.experiment_path,
            profile_path=self.profile_path,
            recipe_path=self.recipe_path,
            provenance_path=self.provenance_path,
            stage=stage,
            run_root=self.run_root,
            assets_lock_path=self.assets_path,
            asset_cache=self.cache,
            seed=seed,
            formal=formal,
        )

    def _create_derived_receipt(self, artifact: str) -> Path:
        _, metadata, _ = self._resolve(stage="smoke", formal=False)
        artifact_metadata = metadata["derived_artifacts"][artifact]
        artifact_path = Path(artifact_metadata["path"])
        receipt_path = Path(artifact_metadata["receipt_path"])
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "payload.bin").write_bytes(artifact.encode("utf-8"))
        file_count, total_bytes, tree_sha256 = self.module._scan_asset_payload(
            artifact_path
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": self.module.DERIVED_RECEIPT_SCHEMA,
                    "artifact": artifact,
                    "fingerprint": metadata["derived_fingerprint"],
                    "status": "succeeded",
                    "tree_hash_algorithm": self.module.TREE_HASH_ALGORITHM,
                    "producer": {
                        "execution_provenance": metadata["execution_provenance"]
                    },
                    "payload": {
                        "file_count": file_count,
                        "total_bytes": total_bytes,
                        "tree_sha256": tree_sha256,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return artifact_path

    def test_recursive_merge_portable_paths_assets_and_hashes_are_deterministic(self) -> None:
        self._create_snapshot("dataset")
        self._create_snapshot("tokenizer")

        config_a, metadata_a, yaml_a = self._resolve(stage="preprocess")
        config_b, metadata_b, yaml_b = self._resolve(stage="preprocess")

        self.assertEqual(config_a["training"]["nested"]["from_base"], True)
        self.assertEqual(config_a["training"]["nested"]["from_experiment"], True)
        self.assertEqual(config_a["training"]["nested"]["replaced"], "experiment")
        self.assertEqual(
            config_a["output"]["run_dir"],
            str(self.run_root.resolve() / "energy" / "preprocess"),
        )
        self.assertEqual(
            config_a["output"]["output_dir"],
            str(self.run_root.resolve() / "outputs" / "preprocess"),
        )
        self.assertEqual(
            config_a["benchmark"]["output_dir"],
            str(self.run_root.resolve() / "outputs" / "eval"),
        )
        self.assertFalse(config_a["energy"]["track_cpu"])
        self.assertEqual(config_a["energy"]["total_energy_policy"], "gpu_only")
        self.assertEqual(config_a["wandb"]["mode"], "offline")
        self.assertTrue(config_a["wandb"]["offline"])
        self.assertNotIn("api_key", config_a["wandb"])
        self.assertNotIn("do-not-write-this", yaml_a)
        self.assertIn("$.wandb.api_key", metadata_a["secrets_removed"])

        self.assertEqual(
            config_a["data"]["dataset_name"],
            str(self._snapshot_path("dataset").resolve()),
        )
        self.assertEqual(
            config_a["data"]["tokenizer_name"],
            str(self._snapshot_path("tokenizer").resolve()),
        )
        self.assertEqual(
            config_a["model"]["teacher_revision"], self.revisions["teacher"]
        )
        self.assertEqual(
            config_a["model"]["student_revision"], self.revisions["student"]
        )
        self.assertEqual(
            config_a["data"]["dataset_revision"], self.revisions["dataset"]
        )
        self.assertEqual(
            config_a["data"]["tokenizer_revision"], self.revisions["tokenizer"]
        )
        self.assertEqual(list(config_a["data"]["datasets"]), ["tulu"])
        self.assertNotIn("/path/to/", yaml_a)

        fingerprint = metadata_a["derived_fingerprint"]
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        expected_derived_root = (
            self.cache.resolve() / "derived" / "kd-test" / fingerprint[:16]
        )
        self.assertEqual(metadata_a["derived_root"], str(expected_derived_root))
        self.assertEqual(
            config_a["data"]["dataset_path"],
            str(expected_derived_root / "preprocessed"),
        )
        for artifact, artifact_metadata in metadata_a["derived_artifacts"].items():
            self.assertEqual(
                artifact_metadata["receipt_path"],
                str(expected_derived_root / ".receipts" / f"{artifact}.json"),
            )
            self.assertFalse(artifact_metadata["receipt_valid"])
        _, smoke_metadata, _ = self._resolve(stage="smoke", formal=False)
        self.assertEqual(smoke_metadata["derived_fingerprint"], fingerprint)

        self.assertEqual(config_a, config_b)
        self.assertEqual(metadata_a, metadata_b)
        self.assertEqual(yaml_a, yaml_b)
        self.assertEqual(
            metadata_a["hashes"]["canonical_config_sha256"],
            metadata_b["hashes"]["canonical_config_sha256"],
        )
        self.assertEqual(
            metadata_a["hashes"]["resolved_yaml_sha256"],
            metadata_b["hashes"]["resolved_yaml_sha256"],
        )

    def test_pilot_recipe_policy_and_metadata_are_preserved(self) -> None:
        recipe = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        recipe.update(
            {
                "classification": "workflow-pilot-not-paper-reproduction",
                "allowed_profiles": ["h100-portable"],
                "pilot": {
                    "enabled": True,
                    "classification": "workflow-pilot-not-paper-reproduction",
                },
                "config_overrides": {
                    "energy_repro_pilot": {
                        "classification": "workflow-pilot-not-paper-reproduction",
                        "train_examples": 8,
                        "eval_examples": 2,
                        "candidate_multiplier": 2,
                        "workers": 1,
                    }
                },
            }
        )
        self.recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

        config, metadata, _ = self._resolve(stage="smoke", formal=False)

        self.assertEqual(
            config["energy_repro_pilot"]["classification"],
            "workflow-pilot-not-paper-reproduction",
        )
        self.assertTrue(metadata["recipe"]["pilot"])
        self.assertEqual(
            metadata["runtime_environment"]["WANDB_MODE"],
            "disabled",
        )
        self.assertEqual(
            metadata["recipe"]["classification"],
            "workflow-pilot-not-paper-reproduction",
        )

        recipe["pilot"]["classification"] = "paper-reproduction"
        self.recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError,
            "workflow-pilot-not-paper-reproduction",
        ):
            self._resolve(stage="smoke", formal=False)

    def test_formal_mode_rejects_unresolved_placeholder(self) -> None:
        with self.base_path.open("a", encoding="utf-8") as handle:
            handle.write("custom_unresolved: /path/to/remain\n")

        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "custom_unresolved"
        ):
            self._resolve(stage="smoke", formal=True)

        config, metadata, _ = self._resolve(stage="smoke", formal=False)
        self.assertEqual(config["custom_unresolved"], "/path/to/remain")
        self.assertTrue(
            any("custom_unresolved" in warning for warning in metadata["warnings"])
        )

    def test_formal_mode_rejects_same_size_asset_tampering(self) -> None:
        self._create_snapshot("dataset")
        tokenizer_path = self._create_snapshot("tokenizer")
        payload_path = tokenizer_path / "payload.bin"
        original = payload_path.read_bytes()
        payload_path.write_bytes(b"x" * len(original))

        with self.assertRaisesRegex(
            self.module.ConfigResolutionError,
            "tree hash differs",
        ):
            self._resolve(stage="preprocess")

    def test_formal_stage_dependencies_allow_targets_but_require_inputs(self) -> None:
        # Preprocess requires only raw dataset + tokenizer and permits its derived
        # output to be absent.
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "required asset snapshot"
        ):
            self._resolve(stage="preprocess")
        self._snapshot_path("dataset").mkdir(parents=True)
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "marker is missing"
        ):
            self._resolve(stage="preprocess")
        self._create_snapshot("dataset")
        self._create_snapshot("tokenizer")
        _, preprocess_metadata, _ = self._resolve(stage="preprocess")
        self.assertFalse(
            Path(preprocess_metadata["derived_artifacts"]["preprocessed"]["path"]).exists()
        )

        # Teacher additionally requires its raw model and the preprocessing input,
        # but permits the logprob target to be absent.
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "required asset snapshot"
        ):
            self._resolve(stage="teacher")
        self._create_snapshot("teacher")
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "preprocessed"
        ):
            self._resolve(stage="teacher")
        _, teacher_metadata, _ = self._resolve(stage="teacher", formal=False)
        Path(
            teacher_metadata["derived_artifacts"]["preprocessed"]["path"]
        ).mkdir(parents=True)
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "receipt is missing"
        ):
            self._resolve(stage="teacher")
        self._create_derived_receipt("preprocessed")
        _, teacher_metadata, _ = self._resolve(stage="teacher")
        preprocessed_payload = (
            Path(teacher_metadata["derived_artifacts"]["preprocessed"]["path"])
            / "payload.bin"
        )
        original_payload = preprocessed_payload.read_bytes()
        preprocessed_payload.write_bytes(b"x" * len(original_payload))
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "content tree differs"
        ):
            self._resolve(stage="teacher")
        self._create_derived_receipt("preprocessed")
        _, teacher_metadata, _ = self._resolve(stage="teacher")
        preprocessed_receipt = Path(
            teacher_metadata["derived_artifacts"]["preprocessed"]["receipt_path"]
        )
        stale_receipt = json.loads(preprocessed_receipt.read_text(encoding="utf-8"))
        stale_receipt["fingerprint"] = "0" * 64
        preprocessed_receipt.write_text(
            json.dumps(stale_receipt),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "fingerprint does not match"
        ):
            self._resolve(stage="teacher")
        self._create_derived_receipt("preprocessed")
        _, teacher_metadata, _ = self._resolve(stage="teacher")
        self.assertFalse(
            Path(
                teacher_metadata["derived_artifacts"]["teacher_logprobs"]["path"]
            ).exists()
        )

        # Student requires the student snapshot and the teacher's completed KD
        # dataset, not just the parent cache directory.
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "required asset snapshot"
        ):
            self._resolve(stage="student")
        self._create_snapshot("student")
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "teacher_logprobs"
        ):
            self._resolve(stage="student")
        _, student_metadata, _ = self._resolve(stage="student", formal=False)
        Path(
            student_metadata["derived_artifacts"]["teacher_logprobs"]["path"]
        ).mkdir(parents=True)
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError, "receipt is missing"
        ):
            self._resolve(stage="student")
        self._create_derived_receipt("teacher_logprobs")
        self._resolve(stage="student")

    def test_derived_fingerprint_changes_with_source_content(self) -> None:
        _, metadata_before, _ = self._resolve(stage="smoke", formal=False)
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profile["config_overrides"] = {"training": {"batch_size": 99}}
        self.profile_path.write_text(json.dumps(profile), encoding="utf-8")

        _, metadata_after, _ = self._resolve(stage="smoke", formal=False)

        self.assertNotEqual(
            metadata_before["derived_fingerprint"],
            metadata_after["derived_fingerprint"],
        )
        self.assertNotEqual(
            metadata_before["derived_root"],
            metadata_after["derived_root"],
        )

    def test_seed_override_is_hashed_and_rejects_negative_values(self) -> None:
        config_default, metadata_default, _ = self._resolve(
            stage="smoke",
            formal=False,
        )
        config_override, metadata_override, _ = self._resolve(
            stage="smoke",
            formal=False,
            seed=7,
        )

        self.assertEqual(config_default["experiment"]["seed"], 42)
        self.assertEqual(config_override["experiment"]["seed"], 7)
        self.assertEqual(metadata_override["seed"], 7)
        self.assertEqual(
            metadata_override["derived_fingerprint_inputs"]["parameters"]["seed"],
            7,
        )
        self.assertNotEqual(
            metadata_default["hashes"]["canonical_config_sha256"],
            metadata_override["hashes"]["canonical_config_sha256"],
        )
        self.assertNotEqual(
            metadata_default["derived_fingerprint"],
            metadata_override["derived_fingerprint"],
        )
        with self.assertRaisesRegex(
            self.module.ConfigResolutionError,
            "non-negative",
        ):
            self._resolve(stage="smoke", formal=False, seed=-1)

    def test_derived_fingerprint_binds_execution_provenance(self) -> None:
        _, before, _ = self._resolve(stage="smoke", formal=False)
        provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        provenance["build_context_sha256"] = "0" * 64
        self.provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        _, after, _ = self._resolve(stage="smoke", formal=False)
        self.assertNotEqual(
            before["derived_fingerprint"],
            after["derived_fingerprint"],
        )

    def test_check_only_writes_nothing(self) -> None:
        output = self.root / "should-not-exist.yaml"
        metadata_output = self.root / "should-not-exist.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = self.module.main(
                [
                    "--base",
                    str(self.base_path),
                    "--experiment",
                    str(self.experiment_path),
                    "--profile",
                    str(self.profile_path),
                    "--recipe",
                    str(self.recipe_path),
                    "--provenance",
                    str(self.provenance_path),
                    "--stage",
                    "smoke",
                    "--run-root",
                    str(self.run_root),
                    "--assets-lock",
                    str(self.assets_path),
                    "--asset-cache",
                    str(self.cache),
                    "--output",
                    str(output),
                    "--metadata-output",
                    str(metadata_output),
                    "--check-only",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertFalse(output.exists())
        self.assertFalse(metadata_output.exists())
        self.assertFalse(self.run_root.exists())
        check_metadata = json.loads(stdout.getvalue())
        self.assertEqual(check_metadata["stage"], "smoke")
        self.assertIn("canonical_config_sha256", check_metadata["hashes"])

    def test_cli_writes_deterministic_yaml_and_canonical_metadata(self) -> None:
        self._create_snapshot("dataset")
        self._create_snapshot("tokenizer")
        output = self.root / "resolved.yaml"
        metadata_output = self.root / "resolved.metadata.json"
        arguments = [
            "--base",
            str(self.base_path),
            "--experiment",
            str(self.experiment_path),
            "--profile",
            str(self.profile_path),
            "--recipe",
            str(self.recipe_path),
            "--provenance",
            str(self.provenance_path),
            "--stage",
            "preprocess",
            "--seed",
            "7",
            "--run-root",
            str(self.run_root),
            "--assets-lock",
            str(self.assets_path),
            "--asset-cache",
            str(self.cache),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata_output),
            "--formal",
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.module.main(arguments), 0)
        yaml_first = output.read_bytes()
        metadata_first = metadata_output.read_bytes()

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.module.main(arguments), 0)
        self.assertEqual(output.read_bytes(), yaml_first)
        self.assertEqual(metadata_output.read_bytes(), metadata_first)

        metadata = json.loads(metadata_first)
        resolved_config = yaml.safe_load(yaml_first)
        self.assertEqual(resolved_config["experiment"]["seed"], 7)
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(
            hashlib.sha256(yaml_first).hexdigest(),
            metadata["hashes"]["resolved_yaml_sha256"],
        )
        self.assertNotIn(b"do-not-write-this", yaml_first)
        self.assertNotIn(b"do-not-write-this", metadata_first)


@unittest.skipIf(yaml is None, "PyYAML is not installed in the host environment")
class ActualBundleConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("resolve_config_actual", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _provenance(self, root: Path) -> Path:
        path = root / "execution-provenance.json"
        path.write_text(
            json.dumps(
                {
                    "schema": self.module.EXECUTION_PROVENANCE_SCHEMA,
                    "upstream_commit": "a" * 40,
                    "upstream_archive_sha256": "b" * 64,
                    "dependency_lock_sha256": "c" * 64,
                    "asset_lock_sha256": "d" * 64,
                    "build_context_sha256": "e" * 64,
                    "resolver_tool_sha256": "f" * 64,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _publish_asset(
        self,
        cache: Path,
        lock: dict[str, Any],
        asset_id: str,
    ) -> None:
        asset = lock["assets"][asset_id]
        cache_key = self.module._asset_cache_key(asset)
        object_path = cache / "assets" / "objects" / cache_key
        object_path.mkdir(parents=True)
        (object_path / "payload.bin").write_bytes(asset_id.encode("utf-8"))
        file_count, total_bytes, tree_sha256 = self.module._scan_asset_payload(object_path)
        marker = {
            "schema": self.module.ASSET_OBJECT_SCHEMA,
            "cache_key": cache_key,
            "descriptor": self.module._asset_descriptor(asset),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "tree_hash_algorithm": self.module.TREE_HASH_ALGORITHM,
            "tree_sha256": tree_sha256,
        }
        (object_path / self.module.ASSET_MARKER_NAME).write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def _publish_derived(self, metadata: dict[str, Any], artifact: str) -> None:
        spec = metadata["derived_artifacts"][artifact]
        path = Path(spec["path"])
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(artifact.encode("utf-8"))
        file_count, total_bytes, tree_sha256 = self.module._scan_asset_payload(path)
        receipt = {
            "schema": self.module.DERIVED_RECEIPT_SCHEMA,
            "artifact": artifact,
            "fingerprint": metadata["derived_fingerprint"],
            "status": "succeeded",
            "tree_hash_algorithm": self.module.TREE_HASH_ALGORITHM,
            "producer": {
                "execution_provenance": metadata["execution_provenance"]
            },
            "payload": {
                "file_count": file_count,
                "total_bytes": total_bytes,
                "tree_sha256": tree_sha256,
            },
        }
        receipt_path = Path(spec["receipt_path"])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def test_actual_pilot_recipes_resolve_to_the_bounded_policy(self) -> None:
        bundle = MODULE_PATH.parents[1]
        upstream = bundle.parent / "tmp" / "Energy-a3b76e8"
        if not upstream.is_dir():
            self.skipTest("pinned upstream audit checkout is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            lock = json.loads(
                (bundle / "assets.lock.json").read_text(encoding="utf-8")
            )
            lock_path = root / "assets.lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            provenance = self._provenance(root)
            cases = {
                "kd-1b-pilot": "kd_32b_to_1b.yaml",
                "sft-1b-pilot": "sft_32b_to_1b.yaml",
                "sft-original-1b-pilot": "sft_32b_to_1b.yaml",
            }
            for recipe_name, experiment_name in cases.items():
                with self.subTest(recipe=recipe_name):
                    config, metadata, yaml_text = self.module.resolve_config(
                        base_path=upstream / "configs" / "base.yaml",
                        experiment_path=upstream
                        / "configs"
                        / "experiments"
                        / experiment_name,
                        profile_path=bundle
                        / "profiles"
                        / "h100-portable.json",
                        recipe_path=bundle
                        / "recipes"
                        / f"{recipe_name}.json",
                        provenance_path=provenance,
                        stage="smoke",
                        run_root=root / recipe_name,
                        assets_lock_path=lock_path,
                        asset_cache=cache,
                        seed=42,
                        formal=False,
                    )
                    pilot = config["energy_repro_pilot"]
                    self.assertEqual(
                        pilot["classification"],
                        "workflow-pilot-not-paper-reproduction",
                    )
                    self.assertEqual(
                        pilot["train_examples"] + pilot["eval_examples"],
                        128,
                    )
                    self.assertFalse(config["experiment"]["debug_mode"])
                    self.assertEqual(config["data"]["max_sequence_length"], 1024)
                    self.assertEqual(config["training"]["batch_size"], 1)
                    self.assertEqual(config["training"]["eval_batch_size"], 1)
                    self.assertEqual(
                        config["training"]["gradient_accumulation_steps"],
                        1,
                    )
                    self.assertEqual(config["training"]["num_epochs"], 1)
                    self.assertFalse(config["wandb"]["enabled"])
                    self.assertTrue(metadata["recipe"]["pilot"])
                    self.assertEqual(
                        metadata["runtime_environment"]["WANDB_MODE"],
                        "disabled",
                    )
                    self.assertNotIn("/path/to/", yaml_text)
                    if recipe_name == "sft-1b-pilot":
                        self.assertEqual(config["batch_size"], 1)
                        self.assertEqual(
                            config["synthetic_data"]["max_gen_examples"],
                            64,
                        )
                        self.assertEqual(
                            config["synthetic_data"]["generation"][
                                "decoding_strategy"
                            ],
                            "greedy",
                        )

    def test_actual_kd_1b_stage_configs_have_no_placeholders(self) -> None:
        bundle = MODULE_PATH.parents[1]
        upstream = bundle.parent / "tmp" / "Energy-a3b76e8"
        if not upstream.is_dir():
            self.skipTest("pinned upstream audit checkout is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            lock = json.loads((bundle / "assets.lock.json").read_text(encoding="utf-8"))
            for asset_id, spec in lock["assets"].items():
                spec["expected_bytes"] = len(asset_id.encode("utf-8"))
            test_lock_path = root / "assets.lock.json"
            test_lock_path.write_text(json.dumps(lock), encoding="utf-8")
            for asset_id in ("tokenizer-7b-sft", "tulu-3"):
                self._publish_asset(cache, lock, asset_id)

            kwargs = dict(
                base_path=upstream / "configs" / "base.yaml",
                experiment_path=upstream / "configs" / "experiments" / "kd_32b_to_1b.yaml",
                profile_path=bundle / "profiles" / "h100-portable.json",
                recipe_path=bundle / "recipes" / "kd-1b.json",
                provenance_path=self._provenance(root),
                run_root=root / "run",
                assets_lock_path=test_lock_path,
                asset_cache=cache,
                seed=99,
                formal=True,
            )
            config, metadata, yaml_text = self.module.resolve_config(
                stage="preprocess",
                **kwargs,
            )
            self.assertNotIn("/path/to/", yaml_text)
            self.assertEqual(config["experiment"]["seed"], 99)
            resolved_run = (root / "run").resolve()
            self.assertEqual(config["output"]["run_dir"], str(resolved_run / "energy" / "preprocess"))
            self.assertEqual(config["output"]["output_dir"], str(resolved_run / "outputs" / "preprocess"))
            self.assertEqual(metadata["seed"], 99)

            self._publish_derived(metadata, "preprocessed")
            self._publish_asset(cache, lock, "teacher-32b")
            teacher_config, teacher_metadata, teacher_yaml = self.module.resolve_config(
                stage="teacher",
                **kwargs,
            )
            self.assertNotIn("/path/to/", teacher_yaml)
            self.assertEqual(
                teacher_config["output"]["output_dir"],
                str(resolved_run / "outputs" / "teacher"),
            )

            self._publish_derived(teacher_metadata, "teacher_logprobs")
            self._publish_asset(cache, lock, "student-1b")
            student_config, _, student_yaml = self.module.resolve_config(
                stage="student",
                **kwargs,
            )
            self.assertNotIn("/path/to/", student_yaml)
            self.assertEqual(
                student_config["data"]["dataset_teacher_logprobs"],
                teacher_metadata["derived_artifacts"]["teacher_logprobs"]["path"],
            )
            self.assertEqual(
                student_config["output"]["output_dir"],
                str(resolved_run / "outputs" / "student"),
            )

    def test_actual_sft_1b_stage_configs_chain_synthetic_artifact(self) -> None:
        bundle = MODULE_PATH.parents[1]
        upstream = bundle.parent / "tmp" / "Energy-a3b76e8"
        if not upstream.is_dir():
            self.skipTest("pinned upstream audit checkout is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            lock = json.loads((bundle / "assets.lock.json").read_text(encoding="utf-8"))
            for asset_id, spec in lock["assets"].items():
                spec["expected_bytes"] = len(asset_id.encode("utf-8"))
            test_lock_path = root / "assets.lock.json"
            test_lock_path.write_text(json.dumps(lock), encoding="utf-8")
            for asset_id in ("tokenizer-7b-sft", "tulu-3"):
                self._publish_asset(cache, lock, asset_id)
            kwargs = dict(
                base_path=upstream / "configs" / "base.yaml",
                experiment_path=upstream / "configs" / "experiments" / "sft_32b_to_1b.yaml",
                profile_path=bundle / "profiles" / "h100-portable.json",
                recipe_path=bundle / "recipes" / "sft-1b.json",
                provenance_path=self._provenance(root),
                run_root=root / "run",
                assets_lock_path=test_lock_path,
                asset_cache=cache,
                seed=42,
                formal=True,
            )
            _, preprocess_metadata, _ = self.module.resolve_config(stage="preprocess", **kwargs)
            self._publish_derived(preprocess_metadata, "preprocessed")
            self._publish_asset(cache, lock, "teacher-32b")
            _, teacher_metadata, teacher_yaml = self.module.resolve_config(stage="teacher", **kwargs)
            self.assertNotIn("/path/to/", teacher_yaml)
            self._publish_derived(teacher_metadata, "synthetic_dataset")
            self._publish_asset(cache, lock, "student-1b")
            student_config, _, student_yaml = self.module.resolve_config(stage="student", **kwargs)
            self.assertNotIn("/path/to/", student_yaml)
            self.assertEqual(
                student_config["synthetic_data"]["synthetic_dataset_path"],
                teacher_metadata["derived_artifacts"]["synthetic_dataset"]["path"],
            )

    def test_actual_original_sft_pilot_uses_preprocessed_artifact_directly(self) -> None:
        bundle = MODULE_PATH.parents[1]
        upstream = bundle.parent / "tmp" / "Energy-a3b76e8"
        if not upstream.is_dir():
            self.skipTest("pinned upstream audit checkout is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            lock = json.loads(
                (bundle / "assets.lock.json").read_text(encoding="utf-8")
            )
            for asset_id, spec in lock["assets"].items():
                spec["expected_bytes"] = len(asset_id.encode("utf-8"))
            test_lock_path = root / "assets.lock.json"
            test_lock_path.write_text(json.dumps(lock), encoding="utf-8")
            for asset_id in ("tokenizer-7b-sft", "tulu-3"):
                self._publish_asset(cache, lock, asset_id)
            kwargs = dict(
                base_path=upstream / "configs" / "base.yaml",
                experiment_path=upstream
                / "configs"
                / "experiments"
                / "sft_32b_to_1b.yaml",
                profile_path=bundle / "profiles" / "h100-portable.json",
                recipe_path=bundle
                / "recipes"
                / "sft-original-1b-pilot.json",
                provenance_path=self._provenance(root),
                run_root=root / "run",
                assets_lock_path=test_lock_path,
                asset_cache=cache,
                seed=42,
                formal=True,
            )
            _, preprocess_metadata, _ = self.module.resolve_config(
                stage="preprocess",
                **kwargs,
            )
            self._publish_derived(preprocess_metadata, "preprocessed")
            self._publish_asset(cache, lock, "student-1b")
            student_config, student_metadata, student_yaml = (
                self.module.resolve_config(stage="student", **kwargs)
            )
            preprocessed_path = student_metadata["derived_artifacts"][
                "preprocessed"
            ]["path"]
            self.assertNotIn("/path/to/", student_yaml)
            self.assertEqual(
                student_config["synthetic_data"]["synthetic_dataset_path"],
                preprocessed_path,
            )
            self.assertTrue(
                student_metadata["derived_artifacts"]["preprocessed"][
                    "required_for_stage"
                ]
            )
            self.assertFalse(
                student_metadata["derived_artifacts"]["synthetic_dataset"][
                    "required_for_stage"
                ]
            )


if __name__ == "__main__":
    unittest.main()
