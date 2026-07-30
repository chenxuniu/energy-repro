# Energy Repro Capsule

This directory is a companion reproducibility layer for
[StellarLuminosity/Energy](https://github.com/StellarLuminosity/Energy).
It pins the upstream source, runtime environment, model and dataset revisions,
hardware checks, run directories, and result manifests so that the same
experiment can be started consistently on different temporary GPU machines.

Pinned inputs:

- Upstream source:
  `a3b76e861943bffa7be64aceb1d65993c9142249`
- Paper:
  [arXiv:2605.13981v1](https://arxiv.org/abs/2605.13981)
- Python: 3.10.13
- PyTorch: 2.6.0+cu124
- CUDA userspace: 12.4
- Transformers: 4.51.3

> **Current scope**
>
> This toolset currently provides trustworthy environment preparation, asset
> prefetching, an H100 telemetry smoke test, explicit
> preprocessing/teacher/student stage execution, manifests, and synchronization.
> It cannot yet claim a complete reproduction of the paper's 3×3 main
> experiments or its five evaluations. `paper-strict` therefore fails closed
> instead of presenting an approximate pipeline that merely appears runnable as
> a paper reproduction.

If you only want to get started on a new machine, begin with
[QUICKSTART.md](QUICKSTART.md). This document describes the design boundaries
and complete operating procedure.

Clone the immutable capsule release rather than a moving branch:

```bash
git clone \
  --branch v0.1.0-energy-a3b76e8 \
  --depth 1 \
  https://github.com/chenxuniu/energy-repro.git
cd energy-repro
```

## Which profile should you choose?

The three profiles represent three different comparability commitments, not
just three sets of performance parameters.

| Profile | Purpose | Stages currently available | How results should be described |
|---|---|---|---|
| `upstream-exact` | Run the pinned commit unchanged, with no scientific source patches | `smoke`, `preprocess`, `teacher`, `student` | Code-faithful, but not validated at paper level |
| `paper-strict` | Strictly require the hardware, software, and experimental semantics reported in the paper | Currently `smoke` only | Training and evaluation fail closed |
| `h100-portable` | Reproduce the workflow and GPU energy measurement across single-H100 hosts | `smoke`, `preprocess`, `teacher`, `student` | GPU-comparable; CPU and total energy are not strictly comparable |

### `upstream-exact`

This profile uses the pinned upstream commit without modifying its scientific
source. The companion layer only applies necessary host bindings, such as
pointing assets and outputs to pinned local directories, disabling network
access during formal runs, and recording the actual environment.

It preserves upstream behavior, including known upstream issues. For example,
microbatch 4 × gradient accumulation 16 in the released configuration actually
produces effective batch 64, while the paper reports effective batch 4.
Upstream checkpoint resume and early stopping also cannot be shown to match the
paper's description. This profile is therefore "code-faithful," not
"paper-strict."

### `paper-strict`

The hardware contract reported in the paper's appendix is:

- 1× NVIDIA H100 SXM 80GB HBM3
- 700W power limit
- Intel Xeon Gold 6442Y, with 48 physical cores and 16 used by the experiment
- Approximately 2TB RAM
- CPU energy readable through RAPL
- Python 3.10.13, PyTorch 2.6.0+cu124, CUDA 12.4, and Transformers 4.51.3
- NVML sampling every 500ms

Hardware is not the only current blocker. The following work listed in
`patches/paper-intent/series.json` has not yet completed audit:

- Deterministic preprocessing
- Synthetic-generation batching and token budget
- Unified optimizer-step semantics
- The early-stopping and best-checkpoint behavior described in the paper
- Exact checkpoint resume
- Complete five-task evaluation at pinned revisions, including the MT-Bench
  judge protocol
- The baseline-SFT branch in the paper's 3×3 grid

The paper and the released configuration also conflict on effective batch size.
Until these issues are resolved, `paper-strict` allows only smoke testing.
`preprocess`, `teacher`, `student`, `eval`, and `all` do not pass the policy
gate.

### `h100-portable`

This is the profile to use for a single-H100 machine that does not exactly match
the paper's host. For example, a Colossus host with 1× H100 80GB, an AMD CPU,
and approximately 256GB RAM does not match the paper's Xeon + approximately 2TB
RAM host, nor does it provide the Intel RAPL conditions required by the paper.

`h100-portable` requires:

- Exactly one visible H100
- At least 79,000 MiB VRAM
- At least 240 GiB host memory
- Record the actual power limit without modifying it automatically
- Exclude CPU energy:
  `energy.track_cpu=false`
- Use this total-energy policy:
  `gpu_only`

This profile therefore supports comparison of GPU energy, power, and throughput
across similar H100 systems. Its CPU energy or GPU+CPU total energy must not be
presented as a strict paper reproduction. Formal runs require explicit
acceptance of these known deviations.

## First startup on a new machine

Use this fixed sequence:

```text
doctor --dry-run
  → bootstrap
  → doctor
  → smoke
  → fetch-assets
  → run
  → manifest
  → sync
```

Place the cache and run directories on a mounted volume that survives the end
of the lease, not on the temporary system disk:

```bash
export ENERGY_REPRO_PROFILE=h100-portable
export ENERGY_REPRO_CACHE=/mnt/energy-cache
export ENERGY_REPRO_RUNS=/mnt/energy-runs
export ENERGY_REPRO_STATE=/mnt/energy-state
```

The launcher explicitly supports **Linux x86_64 + Docker Engine + NVIDIA
Container Toolkit**, with host Python 3.10 or newer. Docker must be usable by
the caller without `sudo`, and the configured cache, run, and state directories
must be writable. It does not claim general Podman compatibility. Containers
run under the caller's UID/GID, and the HF/datasets/torch runtime caches are
written to `${ENERGY_REPRO_CACHE}/runtime` so bind-mounted artifacts do not
become root-owned.

If the machine has no persistent volume, you may leave `ENERGY_REPRO_RUNS` on
the local disk, but you must run `sync` before the lease expires.

### 1. Generate the check plan first

```bash
./repro doctor --profile h100-portable --dry-run
```

`--dry-run` requires no Docker, GPU, or network and creates no files. It only
resolves the locks and profile and lists the checks that a real execution would
perform. You can run it first on a laptop or on a new node that has not yet been
fully configured.

### 2. Prepare the source and image

```bash
./repro bootstrap --profile h100-portable --build-image
./repro doctor --profile h100-portable
```

`bootstrap` prepares a detached source tree at the pinned commit, the pinned
dependency environment, and local state records. It does not automatically
install NVIDIA drivers or Docker, and it does not modify the GPU power limit.

No prebuilt image digest has yet been published in `image/image.lock.json`, so
the first use requires `--build-image`. The resulting image build is included
in run provenance. Until a registry digest is recorded in the image lock, the
local build must not be described as a published bit-for-bit image. Image labels
bind both the complete build context and dependency-lock SHA. Every doctor/run
operation inspects the image and compares it with the current bundle and
bootstrap state before allowing the container to start.

For near-instant reuse on later machines, publish the image once: push the local
tag to an OCI registry you are authorized to use, obtain
`registry/name@sha256:…`, and write that digest into the corresponding profile
in `image/image.lock.json`. On later machines, `bootstrap` without
`--build-image` pulls and verifies the pinned digest. Do not pin only a mutable
tag.

If the pinned source cache, bootstrap state, and verified image have all been
restored on the new machine, you can bootstrap offline:

```bash
./repro bootstrap --profile h100-portable --offline
```

If the required source is absent from the cache or the recorded image is absent
from the local Docker content store, offline mode fails explicitly and does not
silently access the network. Offline mode does not attempt to rebuild the image.

### 3. Prefetch assets before a formal run

For KD 32B→1B:

> On a newly leased GPU whose health has not yet been verified, perform the
> telemetry smoke test in Step 4 before downloading large assets. On a trusted
> node, or when warming a persistent cache on a non-billed preparation node, you
> may prefetch them here.

```bash
./repro fetch-assets --set kd-1b
./repro fetch-assets --set kd-1b --verify-only
```

Other available sets are:

```text
kd-7b   kd-13b
sft-1b  sft-7b  sft-13b
all-core
```

Prefer downloading only the set required by the current recipe; `all-core` is
large. The lock records the exact selected-file byte count for each pinned
revision as of 2026-07-30. The downloader performs a capacity check first, but a
large persistent volume is still recommended.

Models and datasets are pinned to commit revisions in `assets.lock.json`. Note
that neither the paper nor the upstream commit discloses the Hugging Face
revisions used by the authors. The current revisions are immutable snapshots
resolved from each repository's `main` branch on 2026-07-30. They can be
downloaded reproducibly, but they must not be claimed as the authors' original
snapshots.

If the complete assets have already been copied to the new machine:

```bash
./repro fetch-assets --set kd-1b --offline --verify-only
```

A formal `run` never downloads assets on the user's behalf. If assets are
missing or fail verification, it stops and instructs you to run `fetch-assets`
first.

Large-model downloads use a stable partial directory and mutex for each content
key. Hugging Face resume metadata is preserved after a network interruption.
Running the same command again resumes from that partial directory instead of
deleting tens of gigabytes of downloaded content. The final snapshot is
published atomically only after the complete tree hash and exact byte count pass
verification.

### 4. Run the telemetry smoke test first

```bash
./repro smoke --profile h100-portable --kind telemetry
```

The telemetry smoke test downloads no Hugging Face assets. It first performs a
strict runtime probe, then runs the capsule's own 10-second BF16 H100 matrix
load. It samples NVML every 500ms, validates CUDA, positive integrated GPU
energy, nonzero utilization, and ECC/row-remap status, and writes JSON output.
It does not invoke the interface-incompatible upstream `prerun.py` from the
pinned commit.

`--kind pipeline` reserves an interface for a future tiny-step smoke test, but
no audited bounded configuration exists yet. It can currently be used only to
inspect a dry-run plan; real execution fails closed:

```bash
./repro smoke \
  --profile h100-portable \
  --kind pipeline \
  --recipe kd-1b \
  --dry-run
```

Until the semantics of the tiny-step configuration and teacher-derived input
have been tested, do not treat this as a runnable pipeline smoke test, and never
treat it as paper data.

### 5. Run one stage explicitly

Inspect the complete plan before executing:

```bash
./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage preprocess \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations \
  --dry-run
```

After confirming the plan, run it:

```bash
./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage preprocess \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations
```

Then run teacher artifact generation and student training separately:

```bash
./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage teacher \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations

./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage student \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations
```

Record the run ID printed by every `./repro run`. The run ID, plan hash, recipe,
seed, and attempt jointly prevent existing results from being silently
overwritten. After a failure or interruption, do not reuse the same directory
and present it as a fresh complete run. Use a new attempt or follow the recovery
rules below.

Do not currently use `--stage eval` or `--stage all`:

- The required `olmes` package, MT-Bench-101 revision, and judge protocol have
  not all been pinned for evaluation.
- The released commit lacks a runnable baseline-SFT branch for the paper's 3×3
  grid.
- There is no trustworthy one-command end-to-end/resume orchestration spanning
  teacher artifacts, student checkpoints, and short leases.

### 6. Verify the manifest and synchronize

```bash
./repro manifest <RUN_ID> --verify

./repro sync <RUN_ID> \
  --to file:///mnt/persistent/energy-runs \
  --verify
```

`sync` is incremental and repeatable. It deletes neither source nor destination
files and performs SHA-256 verification whether or not `--verify` is specified
explicitly. A terminal run that is no longer being written produces a
`complete` receipt. A run that is active or whose status is unknown can produce
only a `partial` receipt and is never presented as a complete experiment. The
current implementation accepts only local paths or `file:///…`. For object
storage or a remote host, mount the destination as a filesystem first.

## Why formal runs disable network access

Mixing network downloads, experimental computation, and logging introduces
unauditable variation and additional energy consumption. This bundle separates
them:

1. `bootstrap` prepares the pinned source and image.
2. `fetch-assets` is the only model and dataset download phase.
3. `run` verifies local snapshots, then operates without network access.

Formal runs set:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WANDB_MODE=offline
```

The upstream KD pipeline calls `wandb.init()` even when W&B is disabled in its
configuration, so the wrapper enforces offline mode. Local W&B files remain in
the run directory. Any later upload must be performed separately after the
experiment and is not part of the measured stage.

At runtime, only one selected GPU is exposed to the container. Pinned source and
raw assets are mounted read-only. The current run directory and
content-addressed derived cache are writable. Missing assets, source drift,
placeholder paths, or a profile policy failure terminate the run before
training begins.

## Output directory

Each run uses a separate directory:

```text
${ENERGY_REPRO_RUNS}/<run-id>/
├── plan.json
├── state.json
├── manifest.json
├── resolved/
│   ├── execution-provenance.json
│   ├── <stage>.yaml
│   └── <stage>.metadata.json
├── logs/
│   └── <stage>.log
├── energy/
│   └── <stage>/
├── outputs/
│   └── <stage>/
│       ├── checkpoints/
│       └── final_model/
├── wandb/
└── sync/
    └── receipts/
```

The companion layer also corrects two distinct upstream output concepts:

```yaml
output:
  run_dir: <run-root>/energy/<stage>
  output_dir: <run-root>/outputs/<stage>
```

This is necessary because upstream `--run-dir` redirects only EnergyTracker; it
does not automatically redirect checkpoints and the final model. Do not bypass
`./repro` and pass only `--run-dir`, or the energy logs and model outputs may be
written to different locations.

`manifest.json` records the source SHA; actual image reference/ID, including a
registry digest when applicable; dependency lock; resolved configuration; asset
revisions; hardware; power limit; allowlisted environment variables; artifact
hashes; and execution state. Each sync receipt is stored separately under
`sync/receipts/` and in the destination directory. Secrets such as tokens,
passwords, and API keys are not included in these records.
`manifest --verify` also validates the schema, run ID, canonical plan hash,
current lock/profile/recipe bindings, `state.json`, and the equality
"complete set of actual files = manifest inventory" instead of merely spot
checking listed files.

## Cache and persistence layout

Keep these directory classes separate:

```text
/mnt/energy-cache/     Pinned source, HF snapshots, resumable downloads, runtime cache, reusable derived artifacts
/mnt/energy-runs/      Per-run configs, logs, energy data, checkpoints, models, and manifests
/mnt/energy-state/     Bootstrap records, mutable local image ID, and concurrency locks
```

The most valuable items to preserve across machines are:

- Teacher/student weights and tokenizers at pinned revisions
- Raw datasets at pinned revisions
- Preprocessing outputs
- KD teacher logits
- Synthetic-SFT teacher-generation outputs
- Student checkpoints
- Every run manifest and sync receipt

Teacher logits and synthetic datasets are expensive derived artifacts, not raw
assets downloaded by `fetch-assets`. They live in fingerprinted derived-cache
paths and are marked by successful receipts. Their configurations and input
assets are matched before reuse; a matching filename alone is not sufficient
evidence that an artifact is reusable. The fingerprint also binds the upstream
commit/archive, dependency lock, resolver version, and image build context. A
global lock protects producers with the same fingerprint. When a complete
receipt already exists, a new attempt is explicitly recorded as a
`derived-cache-hit` and does not rerun. Partial, corrupted, or mismatched content
causes a fail-closed result and never overwrites the existing cache.

`repro sync <RUN_ID>` synchronizes only the run directory, not the entire cache.
When running an expensive stage, `ENERGY_REPRO_CACHE` must therefore be on a
persistent mounted volume, or the complete cache must be backed up separately.

Do not place Hugging Face tokens, W&B keys, or SSH private keys in the cache,
run directory, or Git.

## Two-hour lease operating pattern

The paper estimates that the complete experiment requires approximately 2,000
H100 GPU-hours. A two-hour lease is suitable for:

- New-machine doctor and telemetry smoke tests
- A bounded pipeline smoke test after an audited tiny-step configuration becomes
  available
- A clearly scoped preprocessing task known to finish within the lease
- A known-size portion of teacher artifact generation
- Validating checkpoint, manifest, and synchronization paths

It is generally not suitable for complete teacher generation, complete student
training, the complete 3×3 grid, or evaluation. Do not report an interrupted
long-running task as a complete experiment merely because the lease lasts only
two hours.

Recommended schedule:

1. **Before the lease starts:** place the image and recipe asset set in the
   persistent cache.
2. **T+0 to T+10 minutes:** run the real `doctor` and telemetry smoke test.
3. **After T+10 minutes:** start only one explicit stage and record its run ID.
4. **With approximately 20 minutes remaining:** start no new stage; generate the
   current manifest and perform the first synchronization.
5. **With approximately 5–10 minutes remaining:** stop work that depends on the
   local temporary disk, verify the sync receipt, and synchronize again.

Whenever a new checkpoint has been fully written, you can repeat:

```bash
./repro manifest <RUN_ID>

./repro sync <RUN_ID> \
  --to file:///mnt/persistent/energy-runs \
  --verify
```

A receipt for a snapshot taken while the run is active is explicitly marked
`partial`; only a stable terminal snapshot is marked `complete`. Neither form
deletes checkpoints synchronized by a previous operation.

Upstream saves checkpoints according to `save_steps` by default, but "the file
exists" does not imply "the run can be resumed exactly." The released code's
resume behavior skips to the next epoch, and the mid-epoch optimizer/data
position lacks enough information to demonstrate a complete recovery.
Therefore:

- A completed preprocess or teacher stage can be safely reused when its manifest
  matches.
- The current CLI does not offer automatic student resume because the released
  code lacks enough state to demonstrate exact recovery.
- Even if a checkpoint happens to lie at an epoch boundary, it must pass an
  independent resume-equivalence test before support can be enabled in a future
  version.
- Every mid-epoch checkpoint is treated as non-exact.
- Without a trustworthy boundary, create a new `attempt` from the last complete
  stage instead of overwriting the old run.
- If non-exact resume is explicitly permitted in a future version, the result
  must declare that deviation and must not be aggregated with a complete
  uninterrupted run.

## Recovery after losing a machine

The simplest reliable recovery method is to mount the same
`ENERGY_REPRO_CACHE` and `ENERGY_REPRO_RUNS` on the new machine. Then run:

```bash
./repro doctor --profile h100-portable
./repro fetch-assets --set kd-1b --offline --verify-only
./repro manifest <RUN_ID> --verify
```

If the previous run was synchronized to remote storage, first restore the
complete run directory and all referenced derived artifacts to persistent
directories on the new machine, then run `manifest --verify`. The current CLI
does not include a `restore` command that automatically restores a remote target
as a local run. Do not mistake `sync` for bidirectional synchronization.

After recovery, begin a new attempt from the latest completed stage whenever
possible. For example, if teacher logits were fully generated and validated
against their fingerprint and receipt, the teacher stage need not be rerun; you
can begin the student stage on the new machine. An interrupted student stage
cannot currently be resumed automatically. Preserve the original attempt and
wait for audited resume support, or create a new attempt from the beginning.

## Bundle directory

```text
energy-repro/
├── repro
├── upstream.lock.json
├── assets.lock.json
├── uv.lock
├── pyproject.toml
├── image/
│   ├── Dockerfile
│   └── image.lock.json
├── locks/
│   └── core-py310-cu124.txt
├── profiles/
│   ├── upstream-exact.json
│   ├── paper-strict.json
│   └── h100-portable.json
├── recipes/
│   ├── kd-{1b,7b,13b}.json
│   └── sft-{1b,7b,13b}.json
└── patches/
    └── paper-intent/
        └── series.json
```

Lock files are provenance inputs. They must not be re-resolved or automatically
upgraded on every new machine. Any change in a source, dependency, image, or
asset revision must produce a new auditable bundle version and a new run
attempt.

## Claims that must not currently be made

Results produced with the current bundle may be described as:

- Using the pinned Energy commit and pinned asset revisions
- Running the specified preprocess/teacher/student stage on one H100
- Recording the actual software, hardware, and GPU energy in a manifest
- An H100 GPU-telemetry-comparable reproduction when using `h100-portable`

They must not currently be described as:

- A strict reproduction of the paper's complete 3×3 main results
- A strict reproduction of the paper's five evaluations
- Proof that the current Hugging Face revisions are the authors' original
  revisions
- CPU or total energy from a non-paper host that is equivalent to the paper's
  Xeon/approximately 2TB host
- A run resumed from a mid-epoch checkpoint after interruption that is fully
  identical to an uninterrupted run

This is why `paper-strict` remains fail-closed: a reproducibility tool must keep
its labels honest before it makes the commands run.
