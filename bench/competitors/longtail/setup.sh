#!/bin/bash
# Creates this harness's two venvs with uv, pinned to the exact interpreters and packages the
# published hf-mps and fasterwhisper-cpu numbers were measured with (`uv pip freeze` of those venvs).
# Apple Silicon only (PyTorch MPS). Needs uv (https://docs.astral.sh/uv/) and network access.
#
#   ./setup.sh                          # both venvs: venv-hf (hf-mps), venv-fw (fasterwhisper-cpu)
#   ./setup.sh fasterwhisper-cpu        # just one runtime's venv
#   ./setup.sh --check [runtime ...]    # compare existing venvs with the pins; changes nothing
#
# Safe to re-run: a venv already on the pinned Python is synced in place (packages not listed below
# are removed); a missing or broken venv, or one on any other Python, is recreated. Model weights are
# not fetched here: run.sh downloads them into hf_home/ on its first run, while loading the model,
# before anything is timed.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# hf-mps: transformers + PyTorch MPS.
HF_PYTHON=3.14.3
HF_PINS='
annotated-doc==0.0.5
anyio==4.15.1
cffi==2.1.1
click==8.5.0
filelock==4.1.0
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.13.1
httpx2==2.13.1
huggingface-hub==2.2.0
idna==3.20
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
packaging==26.3
pycparser==3.11
pygments==2.21.0
pyyaml==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
setuptools==81.0.0
shellingham==1.5.4
soundfile==0.14.0
sympy==1.14.0
tokenizers==0.23.3
torch==2.11.0
tqdm==4.70.1
transformers==5.19.0
truststore==0.10.4
typer==0.27.3
typing-extensions==4.16.0
'

# fasterwhisper-cpu: faster-whisper / CTranslate2. Python 3.13 because onnxruntime (a faster-whisper
# dependency) has no cp314 wheel.
FW_PYTHON=3.13.12
FW_PINS='
anyio==4.15.1
av==19.0.1
cffi==2.1.1
click==8.5.0
ctranslate2==4.8.2
faster-whisper==1.2.1
filelock==4.1.0
flatbuffers==25.12.19
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.13.1
httpx2==2.13.1
huggingface-hub==2.2.0
idna==3.20
numpy==2.5.4
onnxruntime==1.31.0
packaging==26.3
protobuf==7.36.2
pycparser==3.11
pyyaml==6.0.3
soundfile==0.14.0
tokenizers==0.23.3
tqdm==4.70.1
truststore==0.10.4
typing-extensions==4.16.0
'

check=0
if [ "${1:-}" = "--check" ]; then check=1; shift; fi
runtimes=("$@")
[ ${#runtimes[@]} -eq 0 ] && runtimes=(hf-mps fasterwhisper-cpu)

if ! command -v uv >/dev/null; then
  echo "setup.sh: uv not found; install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# python_version DIR: the venv's Python version, or nothing if it has no working interpreter.
python_version() {
  "$1/bin/python" -c 'import platform; print(platform.python_version())' </dev/null 2>/dev/null || true
}

status=0
for rt in "${runtimes[@]}"; do
  case "$rt" in
    hf-mps)            name=venv-hf pyver="$HF_PYTHON" pins="$HF_PINS" ;;
    fasterwhisper-cpu) name=venv-fw pyver="$FW_PYTHON" pins="$FW_PINS" ;;
    *) echo "unknown runtime: $rt (expected hf-mps or fasterwhisper-cpu)" >&2; exit 2 ;;
  esac
  dir="$SCRIPT_DIR/$name"
  have="$(python_version "$dir")"

  if [ "$check" -eq 1 ]; then
    if [ "$have" != "$pyver" ]; then
      echo "$name: Python ${have:-not found}, pinned $pyver" >&2
      status=1
    elif diff <(printf '%s\n' "$pins" | sed '/^$/d' | sort) \
              <(uv pip freeze --python "$dir/bin/python" 2>/dev/null | sort) >&2; then
      echo "$name: matches the pins (Python $pyver)" >&2
    else
      echo "$name: packages differ from the pins (< pinned, > installed)" >&2
      status=1
    fi
    continue
  fi

  echo "== $rt: $name, CPython $pyver" >&2
  if [ "$have" != "$pyver" ]; then
    [ -e "$dir" ] && echo "recreating $name (found Python ${have:-none})" >&2
    # uv-managed CPython (python-build-standalone), like the venvs the numbers came from.
    uv venv --clear --no-project --managed-python --python "cpython@$pyver" "$dir" </dev/null || {
      echo "setup.sh: could not create $name with CPython $pyver; an older uv may not know" \
           "this version (try: uv self update)" >&2
      exit 1
    }
  fi
  # Exactly these packages (no resolution); --strict then checks nothing is missing.
  uv pip sync --strict --python "$dir/bin/python" - <<<"$pins"
done
exit "$status"
