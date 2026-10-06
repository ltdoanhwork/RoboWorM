#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/DreamDojo-svg/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
SVG_ROOT="$ROOT_DIR/video_gen_physics/methods/sparse_attention/Sparse-VideoGen"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Set PYTHON_BIN=/path/to/python3.12 and rerun." >&2
  exit 1
fi

if [ ! -d "$SVG_ROOT" ]; then
  echo "Missing Sparse-VideoGen checkout at $SVG_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$ROOT_DIR/video_gen_physics/models/dreamgen/requirements.txt"
python -m pip install -e "$SVG_ROOT"
python -m pip install flash-attn --no-build-isolation
python -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
python -m pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

pushd "$SVG_ROOT/svg/kernels" >/dev/null
bash setup.sh
popd >/dev/null

cat <<EOF
SVG environment created at: $ENV_DIR

Activate it with:
  source "$ENV_DIR/bin/activate"

Then run DreamGen with:
  export PYTHONPATH="./models/dreamgen"
EOF
