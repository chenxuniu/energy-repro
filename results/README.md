# Public Result Bundles

Each child directory here must be produced by:

```bash
./repro export-results <RUN_ID> --output "results/<RUN_ID>"
```

An exported directory contains sanitized plans/manifests/configs, compact
energy and metric summaries, an export receipt, and `SHA256SUMS`. It must not
contain model weights, checkpoints, raw datasets, assets, W&B files, raw logs,
credentials, host identity, GPU UUIDs, or absolute filesystem paths.

Every workflow-pilot result must retain this classification:

```text
workflow-pilot-not-paper-reproduction
```

Verify a bundle before committing it:

```bash
python3 tools/export_results.py --verify-export "results/<RUN_ID>"
```
