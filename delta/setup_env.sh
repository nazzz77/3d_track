#!/bin/bash
# One-time setup for DELTA. Run on a Zaratan login node or a Runpod pod:
#
#   cd /scratch/zt1/project/<PROJECT>/user/<UMD_ID>/track3d/delta
#   bash setup_env.sh
#
# It installs under delta/work/ (override with DELTA_WORK=/some/dir):
#   miniforge/      Miniforge (conda-forge only, as recommended by UMD HPC)
#   env/            Python 3.10, PyTorch 2.2.0 (CUDA 12.1), DELTA requirements and ffmpeg
#   DELTA/          pinned tracker, UniDepth submodule and checkpoints/densetrack3d.pth
#   hf_cache/       UniDepth V2 model config and weights
#   torch_cache/    DINOv2 weights if requested by the UniDepth config
# Running it again skips finished steps.
set -euo pipefail

DELTA_COMMIT=3367cda1c74d19e73296165f9826b213211678dd  # must match run_delta.py
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${DELTA_WORK:-$HERE/work}"

echo "== Checking the preprocessed clips in $ROOT/preprocessed_data"
[ -f "$ROOT/common/track3d_data.py" ] || { echo "Missing $ROOT/common/track3d_data.py; copy the track3d folder first"; exit 1; }
for clip in pour_lid pen stir shoelace paper_fold shirt_fold; do
  for file in clip.npz meta.json; do
    [ -f "$ROOT/preprocessed_data/$clip/$file" ] || { echo "Missing $ROOT/preprocessed_data/$clip/$file"; exit 1; }
  done
done

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
PY="$WORK/env/bin/python"
REPO="$WORK/DELTA"
UNIDEPTH="$REPO/submodules/UniDepth"
if [ ! -x "$WORK/miniforge/bin/conda" ]; then
  echo "== Installing Miniforge"
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm -f "$WORK/miniforge.sh"
fi

if [ ! -x "$PY" ]; then
  echo "== Creating the Python 3.10 env"
  "$WORK/miniforge/bin/conda" create -y -q -p "$WORK/env" --override-channels -c conda-forge python=3.10 pip ffmpeg
fi

if [ ! -d "$REPO/.git" ]; then
  echo "== Cloning DELTA"
  git clone -q --no-recurse-submodules https://github.com/snap-research/DELTA_densetrack3d.git "$REPO"
fi
git -C "$REPO" checkout -q "$DELTA_COMMIT"
git -C "$REPO" submodule update --init --checkout -- submodules/UniDepth

if [ ! -f "$WORK/env/.delta_pip_ok" ]; then
  echo "== Installing PyTorch 2.2.0 + requirements (the longest step)"
  export PIP_CACHE_DIR="$WORK/pip_cache"  # keep the pip cache out of the 10 GB home quota
  printf 'torch==2.2.0\ntorchvision==0.17.0\ntorchaudio==2.2.0\n' > "$WORK/constraints.txt"
  "$PY" -m pip install -q pip==24.0  # pytorch_lightning 1.6.0 uses legacy dependency metadata
  "$PY" -m pip install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
  "$PY" -m pip install -q -r "$REPO/requirements.txt" -c "$WORK/constraints.txt"
  "$PY" -m pip install -q --only-binary=xformers xformers==0.0.24 gdown==5.2.0 -c "$WORK/constraints.txt"
  touch "$WORK/env/.delta_pip_ok"
fi

# The submodule imports the Chamfer metric even for inference. Load its optional KNN
# backend only when the metric runs, so a login node needs no CUDA compiler or PyTorch3D.
cd "$UNIDEPTH"
"$PY" - <<'PY'
from pathlib import Path

path = Path("unidepth/utils/evaluation_depth.py")
source = path.read_text()
eager = """from unidepth.utils.chamfer_distance import ChamferDistance

chamfer_cls = ChamferDistance()"""
lazy = """def chamfer_cls(*args, **kwargs):
    from unidepth.utils.chamfer_distance import ChamferDistance
    return ChamferDistance()(*args, **kwargs)"""
if eager in source:
    path.write_text(source.replace(eager, lazy))
elif lazy not in source:
    raise SystemExit(f"Unexpected UniDepth source in {path}, check the submodule revision")
PY

CKPT="$REPO/checkpoints/densetrack3d.pth"
URL="https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view"
if [ ! -s "$CKPT" ]; then
  echo "== Downloading the DELTA checkpoint"
  mkdir -p "$REPO/checkpoints"
  if "$WORK/env/bin/gdown" --fuzzy "$URL" -O "$CKPT.part" && [ -s "$CKPT.part" ]; then
    mv "$CKPT.part" "$CKPT"
  else
    echo "Google Drive download failed. Download $URL manually, save it as $CKPT and rerun setup_env.sh." >&2
    exit 1
  fi
fi

cd "$REPO"
export PYTHONPATH="$UNIDEPTH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
if [ ! -f "$HF_HOME/.unidepth_ok" ] || [ ! -f "$TORCH_HOME/.unidepth_ok" ]; then
  echo "== Caching UniDepth V2 and its backbone on the CPU"
  mkdir -p "$HF_HOME" "$TORCH_HOME"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"
  CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
from unidepth.models import UniDepthV2

UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval()
PY
  touch "$HF_HOME/.unidepth_ok" "$TORCH_HOME/.unidepth_ok"
fi

echo "== Checking imports"
"$PY" - <<'PY'
import torch
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.predictor.predictor import Predictor3D
from unidepth.models import UniDepthV2

print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("DenseTrack3D, Predictor3D and UniDepthV2 imports ok")
PY
"$WORK/env/bin/ffmpeg" -version > /dev/null
echo "ffmpeg ok"
touch "$WORK/env/.delta_ok"

echo
echo "Setup finished. Submit the job from this folder:"
echo "  cd $HERE && sbatch run_delta.sbatch"
