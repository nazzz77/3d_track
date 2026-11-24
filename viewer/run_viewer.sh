#!/bin/bash
# Start the comparison viewer; no GPU needed.
#
#   bash viewer/run_viewer.sh                # all clips, opens on the first one
#   bash viewer/run_viewer.sh --clip pen     # any option of viewer/track3d_viewer.py
#   bash viewer/run_viewer.sh --check        # validate every result file without a server
#
# The first run creates .venv in the repository root with the pinned packages of
# viewer/requirements.txt (Python 3.12 or newer). Override with PYTHON=/path/to/python3.12,
# or reuse another environment with VENV=/path/to/venv. The viewer needs preprocessed_data/
# and <method>/results/ from the data archive (see README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
REQ="$ROOT/viewer/requirements.txt"

version_ok() {  # true if the interpreter is Python 3.12 or newer
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import viser, numpy, PIL" 2>/dev/null; then
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    for candidate in python3.13 python3.12 python3; do
      if command -v "$candidate" > /dev/null && version_ok "$candidate"; then PY="$candidate"; break; fi
    done
  fi
  if [ -n "$PY" ]; then
    echo "== Creating $VENV with $("$PY" --version)"
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -r "$REQ"
  elif command -v uv > /dev/null; then
    echo "== No Python 3.12+ found; creating $VENV with uv"
    uv venv -q --python 3.12 "$VENV"
    uv pip install -q --python "$VENV/bin/python" -r "$REQ"
  else
    echo "The viewer needs Python 3.12 or newer. Install it (e.g. 'brew install python@3.12' or" >&2
    echo "'conda create -n viewer python=3.12'), or install uv, then run this script again." >&2
    echo "You can also point to an interpreter with PYTHON=/path/to/python3.12." >&2
    exit 1
  fi
fi

if [ ! -d "$ROOT/preprocessed_data" ] || ! ls "$ROOT"/*/results > /dev/null 2>&1; then
  echo "preprocessed_data/ or <method>/results/ is missing in $ROOT." >&2
  echo "Download track3d_data.zip (see README.md) and unzip it in the repository root." >&2
  exit 1
fi

cd "$ROOT"
exec "$VENV/bin/python" viewer/track3d_viewer.py "$@"
