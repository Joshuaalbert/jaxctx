#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "${python_bin}" >/dev/null; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi

release_env="$(mktemp -d "${TMPDIR:-/tmp}/jaxctx-release.XXXXXX")"
trap 'rm -rf -- "${release_env}"' EXIT

cd "${project_root}"
rm -rf -- dist build src/jaxctx.egg-info

"${python_bin}" -m venv "${release_env}"
"${release_env}/bin/python" -m pip install --upgrade pip build twine
"${release_env}/bin/python" -m build
"${release_env}/bin/python" -m twine check dist/*
"${release_env}/bin/python" -m twine upload dist/*
