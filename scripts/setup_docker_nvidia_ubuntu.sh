#!/usr/bin/env bash
#
# Guarded host bootstrap for a short-lived Ubuntu H100 node.
#
# This script deliberately does not install or replace the NVIDIA driver and
# does not remove conflicting packages. It exits before changing apt sources if
# the host GPU is unhealthy or provider-managed container packages are present.

set -Eeuo pipefail

readonly NVIDIA_CONTAINER_TOOLKIT_VERSION="1.19.1-1"
readonly MINIMUM_DRIVER_VERSION="525.60.13"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

version_at_least() {
  local observed="$1"
  local minimum="$2"
  [[ "$(printf '%s\n%s\n' "${minimum}" "${observed}" | sort -V | head -n 1)" == "${minimum}" ]]
}

[[ "${EUID}" -eq 0 ]] || fail "run this script as root (or with sudo)"
[[ -r /etc/os-release ]] || fail "/etc/os-release is missing"

# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || fail "only Ubuntu is supported; found ${ID:-unknown}"
case "${VERSION_ID:-}" in
  22.04|24.04|26.04) ;;
  *) fail "supported Ubuntu versions are 22.04, 24.04, and 26.04; found ${VERSION_ID:-unknown}" ;;
esac
[[ "$(dpkg --print-architecture)" == "amd64" ]] || fail "only amd64/x86_64 is supported"

command -v nvidia-smi >/dev/null 2>&1 || fail \
  "nvidia-smi is missing; ask the machine provider to repair the driver"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
[[ "${gpu_name}" == *H100* ]] || fail "expected an H100, found: ${gpu_name}"
version_at_least "${driver_version}" "${MINIMUM_DRIVER_VERSION}" || fail \
  "driver ${driver_version} is below the CUDA 12.4 compatibility floor ${MINIMUM_DRIVER_VERSION}"

printf 'Host GPU check passed: %s, driver %s\n' "${gpu_name}" "${driver_version}"

docker_was_installed=false
if command -v docker >/dev/null 2>&1; then
  docker_was_installed=true
  docker version >/dev/null 2>&1 || fail \
    "Docker is installed but its daemon is unusable. Ask the provider to repair it; this script will not replace a provider stack."

  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    if docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi; then
      printf 'Existing Docker/NVIDIA container runtime is already ready; no host packages or configuration were changed.\n'
      exit 0
    fi
  fi

  [[ "${ENERGY_REPRO_ALLOW_PROVIDER_DOCKER_MUTATION:-0}" == "1" ]] || fail \
    "Docker already works, but its NVIDIA runtime did not pass. This may be provider-managed. Review it first; set ENERGY_REPRO_ALLOW_PROVIDER_DOCKER_MUTATION=1 only if you explicitly authorize toolkit installation, Docker runtime configuration, and daemon restart."
fi

if [[ "${docker_was_installed}" == "false" ]]; then
  conflicting_packages=()
  for package in \
    docker.io docker-compose docker-compose-v2 docker-doc docker-buildx \
    podman-docker containerd runc
  do
    if dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii '; then
      conflicting_packages+=("${package}")
    fi
  done
  if ((${#conflicting_packages[@]})); then
    fail "conflicting/provider-managed packages detected: ${conflicting_packages[*]}. Review them before removal."
  fi

  provider_stack_markers=()
  for package in \
    nvidia-container-toolkit nvidia-container-toolkit-base \
    libnvidia-container-tools libnvidia-container1
  do
    if dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii '; then
      provider_stack_markers+=("${package}")
    fi
  done
  for marker in \
    /etc/docker/daemon.json \
    /etc/apt/sources.list.d/nvidia-container-toolkit.list
  do
    if [[ -e "${marker}" ]]; then
      provider_stack_markers+=("${marker}")
    fi
  done
  if ((${#provider_stack_markers[@]})); then
    fail "an existing provider/partial container stack was detected: ${provider_stack_markers[*]}. Review it instead of overwriting it."
  fi

  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl gnupg2
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  architecture="$(dpkg --print-architecture)"
  ubuntu_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
  {
    printf 'Types: deb\n'
    printf 'URIs: https://download.docker.com/linux/ubuntu\n'
    printf 'Suites: %s\n' "${ubuntu_codename}"
    printf 'Components: stable\n'
    printf 'Architectures: %s\n' "${architecture}"
    printf 'Signed-By: /etc/apt/keyrings/docker.asc\n'
  } >/etc/apt/sources.list.d/docker.sources

  apt-get update
  apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
fi

systemctl enable --now docker
docker version

temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "${temporary_directory}"' EXIT
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  -o "${temporary_directory}/nvidia-container-toolkit.gpgkey"
gpg --batch --yes --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  "${temporary_directory}/nvidia-container-toolkit.gpgkey"
curl -fsSL \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  -o "${temporary_directory}/nvidia-container-toolkit.list"
sed \
  's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  "${temporary_directory}/nvidia-container-toolkit.list" \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y \
  "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}"

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

printf 'Docker Engine and NVIDIA Container Toolkit are ready.\n'
