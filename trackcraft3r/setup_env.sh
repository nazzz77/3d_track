#!/bin/bash
# Prepare TrackCraft3R and its separate DA3 environment on a Linux login node or pod.
# Run with bash setup_env.sh. TRACKCRAFT3R_WORK defaults to work/ next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
echo "Checking the preprocessed clips in $ROOT/preprocessed_data"
for file in "$ROOT/common/track3d_data.py" "$ROOT"/preprocessed_data/{pour_lid,pen,stir,shoelace,paper_fold,shirt_fold}/{clip.npz,meta.json}; do
  [ -f "$file" ] || { echo "Missing $file" >&2; exit 1; }
done

WORK="${TRACKCRAFT3R_WORK:-$HERE/work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
REPO="$WORK/TrackCraft3r"
DA3="$WORK/Depth-Anything-3"
PY="$WORK/env/bin/python"
DA3_PY="$WORK/da3_env/bin/python"
TRACK_COMMIT=21e8fcaf4b6375b3044cead210d5808e1d81760b
DA3_COMMIT=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
CKPT_REV=e397c1050bd540062cbd6642ba06a344ca80f116
CKPT_DIR="$WORK/checkpoint"

export PIP_CACHE_DIR="$WORK/pip_cache" CONDA_PKGS_DIRS="$WORK/conda_pkgs"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
export HF_HUB_CACHE="$HF_HOME/hub" MODELSCOPE_CACHE="$WORK/wan_models"
export XDG_CACHE_HOME="$WORK/cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE" "$CKPT_DIR"

if [ ! -x "$WORK/miniforge/bin/conda" ]; then
  echo "Installing Miniforge"
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm -f "$WORK/miniforge.sh"
fi
for env_dir in "$WORK/env" "$WORK/da3_env"; do
  if [ ! -x "$env_dir/bin/python" ]; then
    "$WORK/miniforge/bin/conda" create -y -q -p "$env_dir" --override-channels -c conda-forge python=3.10 pip ffmpeg libgl
  fi
done

checkout() {
  local url="$1" dir="$2" revision="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone -q "$url" "$dir"
  fi
  git -C "$dir" checkout -q "$revision"
}
checkout https://github.com/cvlab-kaist/TrackCraft3r.git "$REPO" "$TRACK_COMMIT"
checkout https://github.com/ByteDance-Seed/depth-anything-3.git "$DA3" "$DA3_COMMIT"

CU121=https://download.pytorch.org/whl/cu121
if [ ! -f "$WORK/env/.trackcraft3r_pip_ok" ]; then
  echo "Installing TrackCraft3R with PyTorch 2.4.1"
  # setup.py imports pkg_resources. Use the installed build backend for the editable install.
  cat > "$WORK/constraints.txt" <<'CONSTRAINTS'
torch==2.4.1
torchvision==0.19.1
setuptools<81
transformers==4.51.3
peft==0.15.2
numpy<2
CONSTRAINTS
  export PIP_CONSTRAINT="$WORK/constraints.txt"
  "$PY" -m pip install -q 'setuptools<81' 'numpy<2' wheel
  "$PY" -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url "$CU121"
  "$PY" -m pip install -q --no-build-isolation -e "$REPO" -r "$REPO/requirements.txt" \
    accelerate huggingface_hub 'imageio[ffmpeg]' opencv-python-headless==4.11.0.86
  touch "$WORK/env/.trackcraft3r_pip_ok"
fi
if [ ! -f "$WORK/da3_env/.da3_pip_ok" ]; then
  echo "Installing DA3 with PyTorch 2.5.1"
  printf 'torch==2.5.1\ntorchvision==0.20.1\nxformers==0.0.28.post3\n' > "$WORK/da3_constraints.txt"
  export PIP_CONSTRAINT="$WORK/da3_constraints.txt"
  "$DA3_PY" -m pip install -q torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 --index-url "$CU121"
  "$DA3_PY" -m pip install -q -e "$DA3"
  touch "$WORK/da3_env/.da3_pip_ok"
fi
# DA3's package metadata omits addict, which depth_anything_3.model.da3 imports.
# Checked outside the marker so an env from an earlier run gets it too.
"$DA3_PY" -c "import addict" 2>/dev/null || "$DA3_PY" -m pip install -q addict==2.4.0

cd "$REPO"
# DA3 poses may already include the homogeneous row. Copy just R and t when
# the official preprocessor pads the poses, for both 3x4 and 4x4 arrays.
"$PY" - <<'PY'
from pathlib import Path

path = Path("scripts/preprocess_da3.py")
source = path.read_text()
old = "    extr[:, :3, :] = extr_3x4\n"
new = "    extr[:, :3, :] = extr_3x4[:, :3, :]\n"
if old in source:
    path.write_text(source.replace(old, new, 1))
elif new not in source:
    raise SystemExit(f"Unexpected pose padding in {path}, check the TrackCraft3R revision")
PY

if [ ! -f "$MODELSCOPE_CACHE/.wan_ok" ]; then
  "$PY" scripts/download_wan_1.3B.py --target "$MODELSCOPE_CACHE"
  # The official downloader omits the google/umt5-xxl tokenizer used for the empty prompt.
  "$PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model = "Wan-AI/Wan2.1-T2V-1.3B"
snapshot_download(model, local_dir=Path(os.environ["MODELSCOPE_CACHE"]) / model,
                  allow_patterns=["google/*"])
PY
  touch "$MODELSCOPE_CACHE/.wan_ok"
fi
if [ ! -s "$CKPT_DIR/model.safetensors" ] || [ ! -f "$CKPT_DIR/.$CKPT_REV" ]; then
  "$PY" - "$CKPT_DIR" "$CKPT_REV" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download("trackcraft3r/checkpoint", "model.safetensors",
                local_dir=sys.argv[1], revision=sys.argv[2])
PY
  touch "$CKPT_DIR/.$CKPT_REV"
fi

if [ ! -f "$HF_HOME/.da3_cpu_ok" ] || [ ! -f "$TORCH_HOME/.da3_cpu_ok" ]; then
  echo "Caching DA3NESTED-GIANT-LARGE and constructing DA3 on the CPU"
  CUDA_VISIBLE_DEVICES="" "$DA3_PY" - <<'PY'
from depth_anything_3.api import DepthAnything3

DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cpu").eval()
PY
  touch "$HF_HOME/.da3_cpu_ok" "$TORCH_HOME/.da3_cpu_ok"
fi

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
if [ ! -f "$MODELSCOPE_CACHE/.predictor_cpu_ok" ]; then
  echo "Building the released TrackCraft3R predictor on the CPU with offline caches"
  CUDA_VISIBLE_DEVICES="" "$PY" - "$CKPT_DIR/model.safetensors" <<'PY'
import sys
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor

WanSceneFlowPredictor(checkpoint_path=sys.argv[1], device="cpu")
PY
  touch "$MODELSCOPE_CACHE/.predictor_cpu_ok"
fi

echo "Checking both environments"
"$PY" - <<'PY'
import cv2, imageio, torch, torchvision
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor

print("TrackCraft3R", torch.__version__, "CUDA", torch.version.cuda)
PY
"$DA3_PY" - <<'PY'
import cv2, scipy, torch, torchvision, xformers.ops
from depth_anything_3.api import DepthAnything3
from huggingface_hub import hf_hub_download

for filename in ("config.json", "model.safetensors"):
    hf_hub_download("depth-anything/DA3NESTED-GIANT-LARGE", filename, local_files_only=True)
print("DA3", torch.__version__, "CUDA", torch.version.cuda)
PY
"$WORK/env/bin/ffmpeg" -version > /dev/null
touch "$WORK/env/.trackcraft3r_ok"
echo "Setup finished. Run sbatch run_trackcraft3r.sbatch or bash run_trackcraft3r.sbatch from $HERE"
