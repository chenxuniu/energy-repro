# Short-Lived H100 Pilot Workflow

This procedure is for isolated rental machines whose local disks disappear
when the lease ends. It assumes one H100 and no shared storage.

Every recipe in this document is classified:

```text
workflow-pilot-not-paper-reproduction
```

The runs validate a reproducible workflow; they are not estimates of the
paper's full losses, quality, runtime, or energy.

## 1. Use the immutable release

For a new clone:

```bash
git clone \
  --branch v0.2.0-energy-a3b76e8 \
  --depth 1 \
  https://github.com/chenxuniu/energy-repro.git
cd energy-repro
```

For a machine that already has an older checkout:

```bash
cd ~/energy-repro
git fetch origin --tags
git checkout v0.2.0-energy-a3b76e8
```

Detached `HEAD` is expected for a release tag. Create a branch only if you later
want to commit exported result directories.

## 2. Check the rented host before changing it

```bash
cat /etc/os-release
uname -m
free -h
nvidia-smi -L
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,power.limit,compute_mode,mig.mode.current \
  --format=csv
nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER
nvidia-smi pmon -c 1
```

Stop and return/escalate the node if host `nvidia-smi` fails, the GPU is not an
H100, another compute process owns it, volatile or aggregate uncorrectable ECC
is nonzero, pending retired pages exist, row remapping failed, or the provider
dashboard still reports a hardware/memory failure. This full-80GB profile also
requires MIG to be disabled and compute mode to permit the job (normally
`Default`). Do not spend a short lease replacing the provider's NVIDIA driver.

## 3. Install only the missing container layer

On a supported Ubuntu 22.04, 24.04, or 26.04 amd64 host, run as root:

```bash
./scripts/setup_docker_nvidia_ubuntu.sh
```

The guarded script:

- requires a working H100 and a CUDA-12.4-compatible host driver first;
- stops if provider-managed/conflicting container packages are installed;
- exits without changing host packages or configuration when an existing
  NVIDIA-enabled Docker stack already passes its GPU-container probe (the probe
  may populate Docker's image cache);
- refuses to reconfigure a working but incomplete Docker installation unless
  `ENERGY_REPRO_ALLOW_PROVIDER_DOCKER_MUTATION=1` is set explicitly;
- does not install, remove, or replace an NVIDIA driver;
- installs Docker Engine from Docker's official apt repository;
- installs the pinned NVIDIA Container Toolkit `1.19.1-1`;
- configures Docker and runs NVIDIA's GPU-container `nvidia-smi` sample.

Review any conflict reported by the script instead of removing provider
packages blindly. The commands follow the official
[Docker Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/),
[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and [NVIDIA sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).

## 4. Treat `/mnt` as lease-local scratch

```bash
LEASE_ATTEMPT="$(date -u +%Y%m%d%H%M%S)"
printf '%s\n' \
  'export ENERGY_REPRO_PROFILE=h100-portable' \
  'export ENERGY_REPRO_CACHE=/mnt/energy-cache' \
  'export ENERGY_REPRO_RUNS=/mnt/energy-runs' \
  'export ENERGY_REPRO_STATE=/mnt/energy-state' \
  "export ENERGY_REPRO_ATTEMPT=$LEASE_ATTEMPT" \
  > /root/energy-repro.env
. /root/energy-repro.env

install -d -m 0755 \
  "$ENERGY_REPRO_CACHE" \
  "$ENERGY_REPRO_RUNS" \
  "$ENERGY_REPRO_STATE"
```

It is fine if `/mnt` is the root NVMe rather than a shared mount. It is fast
scratch, not persistence. Nothing there survives unless it is downloaded
before the lease is released. Run `. /root/energy-repro.env` after every SSH
reconnect; otherwise `repro` falls back to repository-local directories.

The timestamp attempt value keeps run IDs unique when results from multiple
rented machines are collected together. Use the same value for every stage in
one repetition. For a retry after failure, create a new value (or increment the
old one) and update `/root/energy-repro.env`.

## 5. Build and validate the pinned runtime

```bash
./repro doctor --profile h100-portable --mode host
./repro bootstrap --profile h100-portable --build-image
./repro doctor --profile h100-portable --mode runtime
./repro smoke --profile h100-portable --kind telemetry
```

Until this project publishes a registry image digest, `doctor` has an expected
top-level `WARN` for exactly this static warning:

```text
no immutable built-image digest is recorded for profile h100-portable
```

Proceed only when every `host.checks` entry and `runtime.status` is `PASS` and
there are no other warnings. Any `FAIL` is a stop condition.

No immutable prebuilt image is published yet, so the first machine builds the
image locally. The resulting image ID and complete build-context hash are
recorded in provenance.

## 6. Run the smallest experiment first

The direct 1B SFT pilot has approximately 7.4 GB of locked model/dataset assets.
It avoids the 32B teacher and has only two stages. On the first machine, budget
additional network and disk for Docker/base-image/Python dependencies and for
the checkpoint/final-model outputs.

```bash
./repro fetch-assets --set sft-original-1b-pilot
./repro fetch-assets \
  --set sft-original-1b-pilot \
  --offline \
  --verify-only
```

Inspect, then run deterministic preprocessing:

```bash
./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage preprocess \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations \
  --dry-run

./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage preprocess \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations
```

After preprocessing reports `PASS`, run the one-epoch student stage:

```bash
./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage student \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations \
  --dry-run

./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage student \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations
```

Copy both printed run IDs. Never reuse a failed attempt directory; increment
`--attempt` instead. If a producer stage (`preprocess` or `teacher`) fails after
it creates a partial derived artifact, a new attempt deliberately remains
blocked until that content-addressed directory is quarantined. Preserve it for
debugging and move the exact resolved root:

```bash
FAILED_RUN_ID='<paste-failed-run-id>'
FAILED_STAGE='preprocess'  # or teacher
METADATA="$ENERGY_REPRO_RUNS/$FAILED_RUN_ID/resolved/$FAILED_STAGE.metadata.json"

DERIVED_ROOT="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["derived_root"])' \
    "$METADATA"
)"
case "$DERIVED_ROOT" in
  /cache/derived/*) ;;
  *) echo "Unsafe derived root: $DERIVED_ROOT" >&2; exit 1 ;;
esac

HOST_DERIVED_ROOT="$ENERGY_REPRO_CACHE/${DERIVED_ROOT#/cache/}"
QUARANTINE="$ENERGY_REPRO_CACHE/quarantine/$FAILED_RUN_ID"
if [[ -e "$QUARANTINE" ]]; then
  echo "Quarantine destination already exists: $QUARANTINE" >&2
  exit 1
fi
install -d -m 0755 "$QUARANTINE"
mv -- "$HOST_DERIVED_ROOT" "$QUARANTINE/"
```

Moving the fingerprint root may also quarantine a valid earlier producer
artifact for the same recipe, so rerun the stages sequentially with the new
attempt number. A failed `student` stage has no shared derived output; only
increment its attempt.

## 7. Export before the lease ends

For each terminal run:

```bash
RUN_ID='<paste-one-run-id>'
mkdir -p "$PWD/results" /root/energy-public-archives

./repro manifest "$RUN_ID" --verify
./repro export-results "$RUN_ID" \
  --output "$PWD/results/$RUN_ID" \
  --archive "/root/energy-public-archives/$RUN_ID.public.tar.gz"
python3 tools/export_results.py \
  --verify-export "$PWD/results/$RUN_ID"
```

Repeat for the preprocess and student run IDs. The result directory is designed
for a public Git repository. The archive is deterministic and convenient for
download.

From the laptop, using the same SSH host/options used to enter the lease:

```bash
mkdir -p "$HOME/energy-public-archives"
scp 'root@<lease-host>:/root/energy-public-archives/*.public.tar.gz' \
  "$HOME/energy-public-archives/"
```

Do not put GitHub or Hugging Face credentials on the rental node merely to
save a small result. On the laptop, unpack each archive into its own run-ID
directory and verify it before committing:

```bash
cd /path/to/local/energy-repro
ARCHIVE_DIR="$HOME/energy-public-archives"

set -- "$ARCHIVE_DIR"/*.public.tar.gz
if [[ ! -e "$1" ]]; then
  echo "No public result archives found in $ARCHIVE_DIR" >&2
  exit 1
fi
for archive do
  run_id="$(basename "$archive" .public.tar.gz)"
  destination="results/$run_id"
  if [[ -e "$destination" ]]; then
    echo "Result destination already exists: $destination" >&2
    exit 1
  fi
  mkdir -p "$destination"
  tar -xzf "$archive" -C "$destination" --strip-components=1
  python3 tools/export_results.py --verify-export "$destination"
done
```

Each archive has a top-level `result/` directory; the per-run destination
prevents two exports from colliding.

If the trained model itself is needed, transfer it separately; it is
intentionally excluded from public exports and may be several gigabytes:

```bash
MODEL_RUN_ID='<paste-student-run-id>'
rsync -av --partial \
  root@<lease-host>:"/mnt/energy-runs/$MODEL_RUN_ID/outputs/student/final_model/" \
  "./private-models/$MODEL_RUN_ID/"
```

## 8. Later pilots

After the direct SFT pilot succeeds, the bounded KD pilot is:

```bash
./repro fetch-assets --set kd-1b
./repro run --profile h100-portable --recipe kd-1b-pilot \
  --stage preprocess --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
./repro run --profile h100-portable --recipe kd-1b-pilot \
  --stage teacher --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
./repro run --profile h100-portable --recipe kd-1b-pilot \
  --stage student --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
```

The bounded synthetic-SFT pilot uses the same large asset footprint:

```bash
./repro fetch-assets --set sft-1b
./repro run --profile h100-portable --recipe sft-1b-pilot \
  --stage preprocess --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
./repro run --profile h100-portable --recipe sft-1b-pilot \
  --stage teacher --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
./repro run --profile h100-portable --recipe sft-1b-pilot \
  --stage student --seed 42 --attempt "$ENERGY_REPRO_ATTEMPT" --ack-known-deviations
```

Run stages sequentially and wait for `PASS` before starting the next one. The
large recipes download approximately 71.8 GB of locked assets, so do not start
them near the end of a lease.

## Lease rule

With 20 minutes remaining, start no new stage. Verify/export every completed
run and begin downloading. With 5–10 minutes remaining, prioritize completed
public bundles and any private model that is actually needed. Local cache and
failed partial artifacts are disposable by design.
