# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 \
  python:3.10.13-slim-bookworm@sha256:1326d0fd281d283b077fd249e618339a44c9ca5aae6e05cb4f069a087e827922

ARG ENERGY_COMMIT=a3b76e861943bffa7be64aceb1d65993c9142249
ARG ENERGY_SOURCE_SHA256=531d6beeb1c28369e42df295f7d8ea7335540df02da30d17763fbb74b236e048
ARG ENERGY_BUILD_CONTEXT_SHA256
ARG ENERGY_DEPENDENCY_LOCK_SHA256

LABEL org.opencontainers.image.source="https://github.com/StellarLuminosity/Energy" \
      org.opencontainers.image.revision="${ENERGY_COMMIT}" \
      io.energy-repro.source-sha256="${ENERGY_SOURCE_SHA256}" \
      io.energy-repro.build-context-sha256="${ENERGY_BUILD_CONTEXT_SHA256}" \
      io.energy-repro.dependency-lock-sha256="${ENERGY_DEPENDENCY_LOCK_SHA256}" \
      io.energy-repro.source-mode="upstream-exact"

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/opt/energy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_DATASETS_CACHE=/cache/datasets \
    WANDB_MODE=offline \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.4"

RUN python -m venv "${VIRTUAL_ENV}"

COPY locks/core-py310-cu124.txt /tmp/requirements.lock

# uv records the package-to-index binding in uv.lock, while requirements.txt
# carries exact versions and hashes.  The extra index is needed only to make
# the locked +cu124 wheel discoverable; hash mode prevents index substitution.
RUN python -m pip install \
      --no-cache-dir \
      --require-hashes \
      --only-binary=:all: \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      -r /tmp/requirements.lock \
 && python -m pip check

# `./repro bootstrap --build-image` assembles a verified, manifest-checked
# context containing the detached upstream tree and these capsule tools.
COPY upstream/ /opt/energy/
COPY tools/ /opt/energy-repro/tools/

WORKDIR /opt/energy

RUN python -c "import platform, torch, transformers; \
assert platform.python_version() == '3.10.13'; \
assert torch.__version__ == '2.6.0+cu124'; \
assert torch.version.cuda == '12.4'; \
assert transformers.__version__ == '4.51.3'"

CMD ["python", "/opt/energy-repro/tools/runtime_probe.py", "--json"]
