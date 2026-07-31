# New H100 Machine: Minimal Startup Checklist

Use [PILOT.md](PILOT.md) for explanations, health criteria, result download,
and the larger KD/synthetic-SFT pilots.

Clone the immutable release:

```bash
git clone \
  --branch v0.2.0-energy-a3b76e8 \
  --depth 1 \
  https://github.com/chenxuniu/energy-repro.git
cd energy-repro
```

Confirm the host driver/GPU works before installing anything:

```bash
nvidia-smi -L
nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER
```

If host `nvidia-smi` fails or provider hardware/memory health is not `Pass`,
return/escalate the node. Do not replace the driver on a short lease.

On a supported Ubuntu amd64 host without Docker/NVIDIA Container Toolkit, run
as root:

```bash
./scripts/setup_docker_nvidia_ubuntu.sh
```

Use lease-local scratch explicitly:

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

Run `. /root/energy-repro.env` after every SSH reconnect. The timestamp attempt
keeps run IDs from different rented machines distinct.

Build and validate the runtime:

```bash
./repro doctor --profile h100-portable --mode host
./repro bootstrap --profile h100-portable --build-image
./repro doctor --profile h100-portable --mode runtime
./repro smoke --profile h100-portable --kind telemetry
```

The top-level doctor status may be `WARN` only because this release has no
published registry image digest. Require every host check and the runtime
status to be `PASS`; stop on any other warning or any failure.

Run the smallest experiment. Its locked assets are approximately 7.4 GB; the
first image build and model outputs require additional network and disk:

```bash
./repro fetch-assets --set sft-original-1b-pilot
./repro fetch-assets \
  --set sft-original-1b-pilot \
  --offline \
  --verify-only

./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage preprocess \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations

./repro run \
  --profile h100-portable \
  --recipe sft-original-1b-pilot \
  --stage student \
  --seed 42 \
  --attempt "$ENERGY_REPRO_ATTEMPT" \
  --ack-known-deviations
```

These runs are classified
`workflow-pilot-not-paper-reproduction`. Copy each printed run ID, then export
before releasing the machine:

```bash
RUN_ID='<paste-one-run-id>'
mkdir -p "$PWD/results" /root/energy-public-archives

./repro manifest "$RUN_ID" --verify
./repro export-results "$RUN_ID" \
  --output "$PWD/results/$RUN_ID" \
  --archive "/root/energy-public-archives/$RUN_ID.public.tar.gz"
python3 tools/export_results.py --verify-export "$PWD/results/$RUN_ID"
```

Repeat for both stage run IDs and download
`/root/energy-public-archives/` to the laptop. Public exports omit models,
checkpoints, data, raw logs, W&B files, credentials, host identity, GPU UUIDs,
and absolute paths.
