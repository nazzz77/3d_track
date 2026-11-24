#!/bin/bash
# One-time setup for SpatialTrackerV2. Run it on a Zaratan LOGIN node (compute nodes have no
# internet access):
#
#   cd /scratch/zt1/project/<PROJECT>/user/<UMD_ID>/track3d/spatracker_v2
#   bash setup_env.sh
#
# It installs, under spatracker_v2/work/ (about 15 GB; override with SPA_WORK=/some/dir):
#   miniforge/      Miniforge (conda-forge only, as recommended by UMD HPC)
#   env/            Python 3.11 + PyTorch 2.4.1 (CUDA 12.4) + the official requirements + ffmpeg
#   SpaTrackerV2/   the official code at a pinned commit
#   hf_cache/       the two model checkpoints, so the job can run offline
# Running it again is safe: finished steps are skipped. Takes about 15-30 minutes.
set -euo pipefail

SPA_COMMIT=7e12274c52077860cebfe007a6290777db43b63c  # must match run_spatracker.py
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${SPA_WORK:-$HERE/work}"
PY="$WORK/env/bin/python"
mkdir -p "$WORK"

echo "== Checking the preprocessed clips in $ROOT/preprocessed_data"
[ -f "$ROOT/common/track3d_data.py" ] || { echo "Missing $ROOT/common/track3d_data.py; copy the track3d folder first"; exit 1; }
for clip in pour_lid pen stir shoelace paper_fold shirt_fold; do
  [ -f "$ROOT/preprocessed_data/$clip/clip.npz" ] || { echo "Missing $ROOT/preprocessed_data/$clip/clip.npz"; exit 1; }
done

if [ ! -x "$WORK/miniforge/bin/conda" ]; then
  echo "== Installing Miniforge"
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm -f "$WORK/miniforge.sh"
fi

if [ ! -x "$PY" ]; then
  echo "== Creating the Python 3.11 env"
  "$WORK/miniforge/bin/conda" create -y -q -p "$WORK/env" -c conda-forge python=3.11 ffmpeg
fi

if [ ! -d "$WORK/SpaTrackerV2/.git" ]; then
  echo "== Cloning SpatialTrackerV2"
  git clone -q https://github.com/henry123-boy/SpaTrackerV2.git "$WORK/SpaTrackerV2"
fi
git -C "$WORK/SpaTrackerV2" checkout -q "$SPA_COMMIT"

if [ ! -f "$WORK/env/.spatrack_ok" ]; then
  echo "== Installing PyTorch 2.4.1 + requirements (the longest step)"
  export PIP_CACHE_DIR="$WORK/pip_cache"  # keep the pip cache out of the 10 GB home quota
  printf "torch==2.4.1\ntorchvision==0.19.1\ntorchaudio==2.4.1\n" > "$WORK/constraints.txt"
  "$PY" -m pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
  "$PY" -m pip install -q -r "$WORK/SpaTrackerV2/requirements.txt" -c "$WORK/constraints.txt"
  "$PY" -m pip install -q pandas -c "$WORK/constraints.txt"
  touch "$WORK/env/.spatrack_ok"
fi

echo "== Downloading the model checkpoints to $WORK/hf_cache"
export HF_HOME="$WORK/hf_cache"
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("Yuxihenry/SpatialTrackerV2_Front", "Yuxihenry/SpatialTrackerV2-Offline"):
    print(repo, "->", snapshot_download(repo))
EOF

echo "== Checking imports"
cd "$WORK/SpaTrackerV2"
"$PY" -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
"$PY" -c "import pycolmap, sys; sys.path.insert(0, '.'); \
from models.SpaTrackV2.models.predictor import Predictor; \
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track; \
print('SpatialTrackerV2 import ok')"
command -v "$WORK/env/bin/ffmpeg" > /dev/null && echo "ffmpeg ok"

echo
echo "Setup finished. Submit the job from this folder:"
echo "  cd $HERE && sbatch run_spatracker.sbatch"
