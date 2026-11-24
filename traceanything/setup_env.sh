#!/bin/bash
# Prepare TraceAnything on a Linux login node or a Runpod pod.
# Run from this folder with bash setup_env.sh. Override work/ with TRACEANYTHING_WORK.
set -euo pipefail

TRACEANYTHING_COMMIT=54677b5e7bf11510c2e8c917a509988ad379f8eb
HF_REVISION=51f3ea4398de240d5fe1109b6f8f8f3c7fe506a9
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${TRACEANYTHING_WORK:-$HERE/work}"

echo "== Checking the preprocessed clips in $ROOT/preprocessed_data"
[ -f "$ROOT/common/track3d_data.py" ] || { echo "Missing $ROOT/common/track3d_data.py"; exit 1; }
for clip in pour_lid pen stir shoelace paper_fold shirt_fold; do
  for file in clip.npz meta.json; do
    [ -f "$ROOT/preprocessed_data/$clip/$file" ] || { echo "Missing $ROOT/preprocessed_data/$clip/$file"; exit 1; }
  done
done

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
PY="$WORK/env/bin/python"
REPO="$WORK/TraceAnything"
export CONDA_PKGS_DIRS="$WORK/conda_pkgs" PIP_CACHE_DIR="$WORK/pip_cache"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache" MODELSCOPE_CACHE="$WORK/modelscope_cache"
export XDG_CACHE_HOME="$WORK/xdg_cache" PYTHONNOUSERSITE=1
if [ ! -x "$WORK/miniforge/bin/conda" ]; then
  echo "== Installing Miniforge"
  curl -fLsS --retry 3 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm -f "$WORK/miniforge.sh"
fi

if [ ! -x "$PY" ]; then
  echo "== Creating the Python 3.10 environment"
  "$WORK/miniforge/bin/conda" create -y -q -p "$WORK/env" --override-channels -c conda-forge \
    python=3.10 pip 'ffmpeg=*=gpl_*' libgl libglib
fi
export LD_LIBRARY_PATH="$WORK/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -d "$REPO/.git" ]; then
  echo "== Cloning TraceAnything"
  git clone -q https://github.com/ByteDance-Seed/TraceAnything.git "$REPO"
fi
git -C "$REPO" checkout -q "$TRACEANYTHING_COMMIT"

PIP_OK="$WORK/env/.traceanything_pip_ok"
if [ ! -f "$PIP_OK" ]; then
  echo "== Installing the CUDA 12.1 wheels and the README dependencies"
  printf 'torch==2.5.1\ntorchvision==0.20.1\n' > "$WORK/constraints.txt"
  "$PY" -m pip install -q -c "$WORK/constraints.txt" --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1
  "$PY" -m pip install -q -c "$WORK/constraints.txt" numpy==1.26.4 einops==0.8.1 omegaconf==2.3.0 \
    pillow==10.4.0 opencv-python==4.10.0.84 viser==0.2.23 imageio==2.36.1 matplotlib==3.9.4
  touch "$PIP_OK"
fi

cd "$REPO"
CKPT="$REPO/checkpoints/trace_anything.pt"
if [ ! -s "$CKPT" ]; then
  echo "== Downloading trace_anything.pt at $HF_REVISION"
  mkdir -p "$REPO/checkpoints"
  curl -fLsS --retry 3 \
    "https://huggingface.co/depth-anything/trace-anything/resolve/$HF_REVISION/trace_anything.pt" -o "$CKPT.part"
  [ -s "$CKPT.part" ] || { echo "The checkpoint download was empty" >&2; exit 1; }
  mv "$CKPT.part" "$CKPT"
fi

# The inference builder constructs the bundled network and reads this one checkpoint.
# The YAML training targets are unused; RoPE has a bundled fallback. The flash_attention
# branch calls PyTorch SDPA directly. Neither Fast3R nor a flash-attn wheel is needed.
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
WARM="$WORK/env/.traceanything_cpu_ok"
if [ ! "$WARM" -nt "$CKPT" ] || [ ! "$WARM" -nt "$PIP_OK" ]; then
  echo "== Building the official model and loading its weights on the CPU"
  CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
import torch
from scripts.infer import _build_model_from_cfg, _load_cfg

_build_model_from_cfg(_load_cfg("configs/eval.yaml"), "checkpoints/trace_anything.pt", torch.device("cpu"))
PY
  touch "$WARM"
fi

echo "== Checking offline imports and ffmpeg"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 "$PY" - <<'PY'
import cv2
import einops
import imageio
import matplotlib
import torch
import torchvision
import viser
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from scripts.infer import _build_model_from_cfg, _resize_long_side
from trace_anything.trace_anything import evaluate_bspline_conf

assert torch.version.cuda == "12.1", "Expected the CUDA 12.1 torch wheel"
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
PY
"$WORK/env/bin/ffmpeg" -loglevel error -f lavfi -i color=s=16x16 -frames:v 1 -c:v libx264 -f null -
touch "$WORK/env/.traceanything_ok"
echo "Setup finished. Submit with: cd $HERE && sbatch run_traceanything.sbatch"
