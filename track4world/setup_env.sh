#!/bin/bash
# Prepare Track4World on a Linux login node or Runpod. Set TRACK4WORLD_WORK
# to keep the environment, checkouts, downloads and caches on scratch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${TRACK4WORLD_WORK:-$HERE/work}"
MODEL_COMMIT=1c2a0f814d8352d2750527b70a6c546e056c2c67
UTILS_COMMIT=2072c024c73f7c0f83e0da23eef5f2d9ac575249
PI3_COMMIT=a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99
CKPT_REV=93ae34410efd05d1b7abfdb0f749bc93f7655404

echo "== Checking preprocessed clips"
for file in "$ROOT/common/track3d_data.py" "$ROOT"/preprocessed_data/{pour_lid,pen,stir,shoelace,paper_fold,shirt_fold}/{clip.npz,meta.json}; do
  [ -f "$file" ] || { echo "Missing $file" >&2; exit 1; }
done

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
REPO="$WORK/Track4World"
EXTERNAL="$REPO/track4world/nets/external"
PY="$WORK/env/bin/python"
export PIP_CACHE_DIR="$WORK/pip_cache" CONDA_PKGS_DIRS="$WORK/conda_pkgs"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
export HF_HUB_CACHE="$HF_HOME/hub" MODELSCOPE_CACHE="$WORK/modelscope_cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

if [ ! -x "$WORK/miniforge/bin/conda" ]; then
  echo "== Installing Miniforge"
  curl -fLsS --retry 3 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm -f "$WORK/miniforge.sh"
fi
if [ ! -x "$PY" ]; then
  echo "== Creating the Python 3.11 environment"
  "$WORK/miniforge/bin/conda" create -y -q -p "$WORK/env" --override-channels -c conda-forge python=3.11 pip ffmpeg
fi
export PATH="$WORK/env/bin:$PATH"
rm -f "$WORK/env/.track4world_ok"

# The only registered submodule is Open-d4rt, used by a separate evaluation.
if [ ! -d "$REPO/.git" ]; then
  git clone -q --no-recurse-submodules https://github.com/TencentARC/Track4World.git "$REPO"
fi
git -C "$REPO" checkout -q --detach "$MODEL_COMMIT"
if [ ! -d "$REPO/utils3d/.git" ]; then
  git clone -q https://github.com/jiah-cloud/utils3d.git "$REPO/utils3d"
fi
git -C "$REPO/utils3d" checkout -q --detach "$UTILS_COMMIT"

# model.py imports pi3.models.pi3 even when constructing the DA3 backbone.
# Only its Python package is needed. DA3 never loads the Pi3 weights.
if [ ! -f "$EXTERNAL/pi3/.pinned-$PI3_COMMIT" ]; then
  PI3_REPO="$WORK/Pi3"
  if [ ! -d "$PI3_REPO/.git" ]; then
    git clone -q --no-checkout --filter=blob:none https://github.com/yyfz/Pi3.git "$PI3_REPO"
  fi
  git -C "$PI3_REPO" sparse-checkout set --cone pi3
  git -C "$PI3_REPO" checkout -q --detach "$PI3_COMMIT"
  mkdir -p "$EXTERNAL/pi3"
  cp -a "$PI3_REPO/pi3/." "$EXTERNAL/pi3/"
  touch "$EXTERNAL/pi3/.pinned-$PI3_COMMIT"
fi

if [ ! -f "$WORK/env/.track4world_pip_ok" ]; then
  echo "== Installing CUDA 12.1 PyTorch and the pinned repository requirements"
  printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n' > "$WORK/constraints.txt"
  "$PY" -m pip install -q -c "$WORK/constraints.txt" --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  "$PY" -m pip install -q -c "$WORK/constraints.txt" -r "$REPO/requirements.txt" addict==2.4.0
  touch "$WORK/env/.track4world_pip_ok"
fi

CKPT="$REPO/checkpoints/track4world_da3.pth"
if [ ! -s "$CKPT" ]; then
  echo "== Downloading the pinned DA3 tracker checkpoint"
  mkdir -p "$REPO/checkpoints"
  curl -fLsS --retry 3 "https://huggingface.co/TencentARC/Track4World/resolve/$CKPT_REV/track4world_da3.pth" -o "$CKPT.part"
  [ -s "$CKPT.part" ]
  mv "$CKPT.part" "$CKPT"
fi

cd "$REPO"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE"
if [ ! -f "$HF_HOME/.track4world_cached" ] || [ ! -f "$TORCH_HOME/.track4world_cached" ]; then
  echo "== Building the runtime model on CPU to populate the offline caches"
  CUDA_VISIBLE_DEVICES="" "$PY" - "$REPO" "$CKPT" <<'PY'
import sys
from pathlib import Path
from run_track4world import build_model

build_model(Path(sys.argv[1]), Path(sys.argv[2]))
PY
  touch "$HF_HOME/.track4world_cached" "$TORCH_HOME/.track4world_cached"
fi

echo "== Checking offline imports, cached DA3 files and ffmpeg"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" "$PY" - "$REPO" <<'PY'
import sys
from pathlib import Path
from run_track4world import configure_repo

configure_repo(Path(sys.argv[1]))
import cv2
import torch
import utils3d
from huggingface_hub import hf_hub_download
from track4world.nets.model import Track4World

for name in ("config.json", "model.safetensors"):
    hf_hub_download("depth-anything/DA3NESTED-GIANT-LARGE-1.1", name, local_files_only=True)
assert callable(utils3d.torch.depth_to_points) and callable(utils3d.torch.unproject_cv)
print("Track4World, utils3d and OpenCV imports ok. PyTorch", torch.__version__, "CUDA", torch.version.cuda)
PY
"$WORK/env/bin/ffmpeg" -version > /dev/null
touch "$WORK/env/.track4world_ok"
echo "Setup finished. From $HERE run sbatch run_track4world.sbatch or bash run_track4world.sbatch."
