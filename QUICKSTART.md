# New H100 Machine: Minimal Startup Checklist

Prerequisites: the provider's control plane reports `Pass` for GPU
hardware/memory health; the host is Linux x86_64; host Python is 3.10 or newer;
and Docker Engine, NVIDIA Container Toolkit, and Git are installed. Docker must
be usable by your account without `sudo`, and the selected cache, run, and state
directories must exist and be writable. In Colossus, verify the corresponding
health status on the lease page. Do not place SSH, Hugging Face, or W&B
credentials in this directory.

```bash
git clone \
  --branch v0.1.0-energy-a3b76e8 \
  --depth 1 \
  https://github.com/chenxuniu/energy-repro.git
cd energy-repro

export ENERGY_REPRO_PROFILE=h100-portable
export ENERGY_REPRO_CACHE=/mnt/energy-cache
export ENERGY_REPRO_RUNS=/mnt/energy-runs
export ENERGY_REPRO_STATE=/mnt/energy-state

./repro doctor --profile h100-portable --dry-run
./repro bootstrap --profile h100-portable --build-image
./repro doctor --profile h100-portable
./repro smoke --profile h100-portable --kind telemetry
```

Before running a recipe for the first time, prefetch its pinned assets. If the
download is interrupted, repeat the same command to resume it:

```bash
./repro fetch-assets --set kd-1b
./repro fetch-assets --set kd-1b --offline --verify-only
```

Inspect the plan before a formal run, then start only one stage:

```bash
./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage preprocess \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations \
  --dry-run

./repro run \
  --profile h100-portable \
  --recipe kd-1b \
  --stage preprocess \
  --seed 42 \
  --attempt 1 \
  --ack-known-deviations
```

After preprocessing completes successfully, change `--stage` to `teacher` and
then `student`. Every command produces a distinct run ID. After a failure or
interruption, increment `--attempt`; do not overwrite an earlier attempt. When a
complete derived receipt already exists, the producer reports
`derived-cache-hit` and does not recompute or overwrite the artifact.

Before the lease ends:

```bash
./repro manifest <RUN_ID> --verify
./repro sync <RUN_ID> \
  --to file:///mnt/persistent/energy-runs \
  --verify
```

A synchronization performed while a run is active is marked `partial`; only a
stable terminal snapshot is marked `complete`. Expensive teacher artifacts live
under `${ENERGY_REPRO_CACHE}/derived` and are not included in run-only
synchronization. `ENERGY_REPRO_CACHE` itself must be on a persistent volume or
backed up separately.

The complete paper grid requires approximately 2,000 H100 GPU-hours. A two-hour
machine is suitable for environment validation, telemetry smoke testing,
preprocessing, or a single stage known to complete within that window. It is
not sufficient to claim a complete reproduction of the paper.
