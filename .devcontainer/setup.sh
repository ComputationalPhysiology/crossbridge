#!/usr/bin/env bash
set -euo pipefail

# The workspace is bind-mounted from the host, so it is owned by a different
# uid than root; without this git (and hence pre-commit) refuses to touch it.
git config --global --add safe.directory "$PWD"

chsh -s "$(which zsh)" root || true
if ! command -v starship &> /dev/null; then
  echo "Installing Starship prompt..."
  curl -sS https://starship.rs/install.sh | sh -s -- -y
fi

# Python comes from the base image; these are just build tools for any
# dependency that has to compile from source.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y install --no-install-recommends \
    build-essential gfortran patch file zstd unzip zip bzip2 xz-utils
apt-get clean
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
python -m pip install -e ".[test,fast,demos,dev]"
pre-commit install

echo "Container setup complete!"
