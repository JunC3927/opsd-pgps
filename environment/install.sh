#!/usr/bin/env bash
# An independent prefix only. Never modifies an existing Conda/base environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prefix=""
conda_exe="${CONDA_EXE:-}"
python_version="3.12.14"
index_url="https://pypi.org/simple"
conda_channel="conda-forge"
wheel=""
download=0
direct=0
resume=0
cache_dir=""
usage() {
  cat <<'EOF'
Usage: bash environment/install.sh --prefix /your/new/conda/env \
  (--flash-attn-wheel /path/to/historical.whl | --download-flash-attn) [options]

Options:
  --conda PATH                 Conda executable (or use CONDA_EXE / PATH)
  --python-version 3.12.X       Explicit patch override; default 3.12.14
  --index-url URL              PyPI mirror; default https://pypi.org/simple
  --conda-channel URL_OR_NAME   Conda channel; default conda-forge
  --cache-dir DIR              Caches/logs/tmp; default sibling PREFIX-install
  --no-proxy                   Ignore HTTP(S)/ALL proxy environment variables
  --resume                     Continue only a prefix owned by this installer

No GPU is used. No training/service is started. No automatic version substitutions.
EOF
}
while (($#)); do
  case "$1" in
    --prefix) prefix="$2"; shift 2;;
    --conda) conda_exe="$2"; shift 2;;
    --python-version) python_version="$2"; shift 2;;
    --index-url) index_url="$2"; shift 2;;
    --conda-channel) conda_channel="$2"; shift 2;;
    --cache-dir) cache_dir="$2"; shift 2;;
    --flash-attn-wheel) wheel="$2"; shift 2;;
    --download-flash-attn) download=1; shift;;
    --no-proxy) direct=1; shift;;
    --resume) resume=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$prefix" && "$prefix" == /* && "$prefix" != / ]] || { usage; exit 2; }
[[ "$python_version" =~ ^3\.12\.[0-9]+$ ]] || { echo 'Explicit CPython 3.12.X required' >&2; exit 2; }
[[ -n "$wheel" && "$download" == 0 || -z "$wheel" && "$download" == 1 ]] || { usage; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required' >&2; exit 2; }
if [[ -z "$conda_exe" ]]; then conda_exe="$(command -v conda || true)"; fi
[[ -n "$conda_exe" && -x "$conda_exe" ]] || { echo 'Install Conda or provide --conda PATH' >&2; exit 2; }
if [[ "$index_url" == *@* || "$index_url" == *\?* || "$conda_channel" == *@* || "$conda_channel" == *\?* ]]; then
  echo 'Use public mirror URLs without credentials/query tokens' >&2; exit 2
fi
if [[ "$direct" == 1 ]]; then
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_TRUSTED_HOST PIP_CONSTRAINT PIP_REQUIREMENT
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
cache_dir="${cache_dir:-${prefix%/}-install}"
mkdir -p "$cache_dir" "$cache_dir/logs" "$cache_dir/tmp" "$cache_dir/wheels"
cache_dir="$(cd "$cache_dir" && pwd)"
export PIP_CACHE_DIR="$cache_dir/pip-cache" CONDA_PKGS_DIRS="$cache_dir/conda-pkgs"
export XDG_CACHE_HOME="$cache_dir/xdg-cache" TMPDIR="$cache_dir/tmp" TMP="$cache_dir/tmp" TEMP="$cache_dir/tmp"
exec 9>"$cache_dir/install.lock"
flock -n 9 || { echo 'Another installer holds this cache lock' >&2; exit 3; }
claim="$cache_dir/prefix.claim"
if [[ -e "$prefix" ]]; then
  [[ "$resume" == 1 && -f "$claim" && "$(cat "$claim")" == "$prefix" ]] || {
    echo 'Refusing existing prefix; --resume only works for this installer-owned prefix' >&2; exit 3;
  }
else
  [[ ! -e "$claim" || "$(cat "$claim")" == "$prefix" ]] || { echo 'Cache belongs to another prefix' >&2; exit 3; }
  printf '%s\n' "$prefix" > "$claim"
  "$conda_exe" create --yes --prefix "$prefix" --override-channels -c "$conda_channel" \
    "python=$python_version" pip > "$cache_dir/logs/conda-create.log" 2>&1
fi
python="$prefix/bin/python"
[[ -x "$python" ]] || { echo 'Partial Conda creation; inspect log before recovery' >&2; exit 3; }
"$python" -c 'import sys; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve(); assert sys.version.split()[0]==sys.argv[2]' "$prefix" "$python_version"
printf 'Installing pinned bootstrap tools; logs: %s\n' "$cache_dir/logs"
"$python" -m pip install --no-deps --index-url "$index_url" -r "$script_dir/bootstrap.lock.txt" \
  > "$cache_dir/logs/bootstrap.log" 2>&1
printf 'Installing historical package pins...\n'
"$python" -m pip install --no-deps --no-build-isolation --index-url "$index_url" \
  -r "$script_dir/requirements.lock.txt" > "$cache_dir/logs/packages.log" 2>&1
fetch_args=(--output-dir "$cache_dir/wheels")
if [[ "$download" == 1 ]]; then fetch_args+=(--download); else fetch_args+=(--wheel "$wheel"); fi
if [[ "$direct" == 1 ]]; then fetch_args+=(--no-proxy); fi
fa_wheel="$("$python" "$script_dir/fetch_flash_attn.py" "${fetch_args[@]}")"
"$python" -m pip install --no-deps --no-build-isolation "$fa_wheel" > "$cache_dir/logs/flash-attn.log" 2>&1
"$python" -m pip check > "$cache_dir/logs/pip-check.log" 2>&1
"$python" "$script_dir/verify_environment.py" --imports --expected-python "$python_version" \
  --prefix "$prefix" --output "$cache_dir/verification.json" > "$cache_dir/logs/verification.log" 2>&1
"$python" -m pip freeze > "$cache_dir/installed.freeze.txt"
printf 'Environment metadata/import checks passed. Report: %s\n' "$cache_dir/verification.json"
printf 'H100 kernels and six-GPU training are NOT yet verified. Run the separate GPU smoke test.\n'
