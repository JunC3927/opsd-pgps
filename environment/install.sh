#!/usr/bin/env bash
# An independent prefix only. Never modifies an existing Conda/base environment.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prefix=""
conda_exe="${CONDA_EXE:-}"
python_version=""
profile="compatible"
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
  --profile compatible|historical  Default compatible: fixed core, resolved auxiliary tools
  --python-version 3.12[.X]     Default 3.12 (compatible) / 3.12.14 (historical)
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
    --profile) profile="$2"; shift 2;;
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
[[ "$profile" == compatible || "$profile" == historical ]] || { usage; exit 2; }
if [[ -z "$python_version" ]]; then
  if [[ "$profile" == historical ]]; then python_version=3.12.14; else python_version=3.12; fi
fi
[[ "$python_version" =~ ^3\.12(\.[0-9]+)?$ ]] || { echo 'CPython 3.12[.X] required by the FA2 wheel' >&2; exit 2; }
[[ "$profile" != historical || "$python_version" =~ ^3\.12\.[0-9]+$ ]] || { echo 'Historical mode requires an explicit Python patch version' >&2; exit 2; }
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
stage_log=""
trap 'rc=$?; printf "\nInstallation stopped (exit %s). Last stage log: %s\n" "$rc" "${stage_log:-not started}" >&2; exit "$rc"' ERR
run_stage() {
  local name="$1"; shift
  stage_log="$cache_dir/logs/$name.log"
  printf '\n[%s] %s — log: %s\n' "$(date '+%H:%M:%S')" "$name" "$stage_log"
  "$@" 2>&1 | tee -a "$stage_log"
}
printf 'Install profile: %s; Python: %s; environment: %s\n' "$profile" "$python_version" "$prefix"
claim="$cache_dir/prefix.claim"
if [[ -e "$prefix" ]]; then
  [[ "$resume" == 1 && -f "$claim" && "$(cat "$claim")" == "$prefix" ]] || {
    echo 'Refusing existing prefix; --resume only works for this installer-owned prefix' >&2; exit 3;
  }
else
  [[ ! -e "$claim" || "$(cat "$claim")" == "$prefix" ]] || { echo 'Cache belongs to another prefix' >&2; exit 3; }
  printf '%s\n' "$prefix" > "$claim"
  run_stage conda-create "$conda_exe" create --yes --prefix "$prefix" --override-channels -c "$conda_channel" \
    "python=$python_version" pip
fi
python="$prefix/bin/python"
[[ -x "$python" ]] || { echo 'Partial Conda creation; inspect log before recovery' >&2; exit 3; }
"$python" -c 'import sys; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve(); assert sys.version_info[:2]==(3,12); assert sys.argv[2]=="3.12" or sys.version.split()[0]==sys.argv[2]' "$prefix" "$python_version"
if [[ "$profile" == historical ]]; then
  run_stage bootstrap "$python" -m pip install --no-deps --index-url "$index_url" -r "$script_dir/bootstrap.lock.txt"
  run_stage packages "$python" -m pip install --no-deps --no-build-isolation --index-url "$index_url" \
    -r "$script_dir/requirements.lock.txt"
else
  run_stage bootstrap "$python" -m pip install --index-url "$index_url" -r "$script_dir/bootstrap.compatible.txt"
  run_stage packages "$python" -m pip install --no-build-isolation --index-url "$index_url" \
    -r "$script_dir/requirements.compatible.txt"
fi
fetch_args=(--output-dir "$cache_dir/wheels")
if [[ "$download" == 1 ]]; then fetch_args+=(--download); else fetch_args+=(--wheel "$wheel"); fi
if [[ "$direct" == 1 ]]; then fetch_args+=(--no-proxy); fi
run_stage flash-attn-download "$python" "$script_dir/fetch_flash_attn.py" "${fetch_args[@]}"
fa_wheel="$cache_dir/wheels/$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["filename"])' "$script_dir/flash_attn_wheel.json")"
run_stage flash-attn "$python" -m pip install --no-deps --no-build-isolation "$fa_wheel"
run_stage pip-check "$python" -m pip check
run_stage verification "$python" "$script_dir/verify_environment.py" --imports --profile "$profile" --expected-python "$python_version" \
  --prefix "$prefix" --output "$cache_dir/verification.json"
"$python" -m pip freeze > "$cache_dir/installed.freeze.txt"
printf 'Environment metadata/import checks passed. Report: %s\n' "$cache_dir/verification.json"
printf 'H100 kernels and six-GPU training are NOT yet verified. Run the separate GPU smoke test.\n'
