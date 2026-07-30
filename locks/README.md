# Dependency locks

`uv.lock` and `core-py310-cu124.txt` are generated for Linux x86-64 with
Python 3.10.13 and uv 0.6.14:

```bash
uv lock --python 3.10.13
uv export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --format requirements-txt \
  --output-file locks/core-py310-cu124.txt
```

The OCI build installs the exported file with `pip --require-hashes`. Do not
replace the explicit PyTorch cu124 index with a generic extra index, and do not
regenerate the lock on a new machine as part of bootstrap.

The evaluation stack is deliberately not included. The pinned upstream commit
does not declare the package that provides `olmes`, and its MT-Bench-101
dependency follows an unpinned Git branch. Evaluation remains fail-closed until
those inputs and the judge protocol are pinned.
