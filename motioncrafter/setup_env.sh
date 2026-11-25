#!/bin/bash
# Run once on the Linux pod or a Zaratan login node.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
echo "Checking the preprocessed clips in $ROOT/preprocessed_data"
test -f "$ROOT/common/track3d_data.py"
for clip in pour_lid pen stir shoelace paper_fold shirt_fold; do
  for name in meta.json clip.npz; do
    file="$ROOT/preprocessed_data/$clip/$name"
    [[ -s "$file" ]] || { echo "Missing $file" >&2; exit 1; }
  done
done

WORK="${MOTIONCRAFTER_WORK:-$HERE/work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
PY="$WORK/env/bin/python"
REPO="$WORK/MotionCrafter"
COMMIT=9cb4e9679f5f34e249945544052464ef46324bc2
export PIP_CACHE_DIR="$WORK/pip_cache" CONDA_PKGS_DIRS="$WORK/conda_pkgs"
export HF_HOME="$WORK/hf_cache" HF_HUB_CACHE="$WORK/hf_cache/hub" TORCH_HOME="$WORK/torch_cache"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$WORK/miniforge/bin/conda" ]]; then
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$WORK/miniforge.sh"
  bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
  rm "$WORK/miniforge.sh"
fi
if [[ ! -x "$PY" ]]; then
  "$WORK/miniforge/bin/conda" create -y -p "$WORK/env" --override-channels -c conda-forge \
    python=3.11 pip=25.0.1 'ffmpeg=7.1.1=gpl*'
fi
if [[ ! -d "$REPO/.git" ]]; then
  git clone --no-recurse-submodules https://github.com/TencentARC/MotionCrafter.git "$REPO"
fi
git -C "$REPO" checkout -q --detach "$COMMIT"
git -C "$REPO" diff --quiet HEAD -- run.py motioncrafter || { echo "Modified inference code in $REPO" >&2; exit 1; }
cd "$REPO"

# run.py imports h5py even for NPZ, and attempts to enable xformers. The CPU
# decord wheel decodes the input. The conda ffmpeg executable encodes it.
# Skip matplotlib, mediapy, OpenEXR, pandas, pytorch3d, kornia, opencv-python,
# moderngl, piqp, gin-config, wandb and wandb-osh (training, data, eval or GUI).
# Skip imageio, imageio-ffmpeg and imageio-pyav (optional resized-video export).
# The default Euler schedule does not use scipy's optional beta distribution.
export PIP_CONSTRAINT="$WORK/constraints.txt"
cat > "$PIP_CONSTRAINT" <<'PINS'
torch==2.6.0
torchvision==0.21.0
xformers==0.0.29.post3
numpy==1.26.4
pillow==11.1.0
diffusers==0.32.2
transformers==4.48.3
accelerate==1.3.0
huggingface-hub==0.28.1
safetensors==0.5.2
decord==0.6.0
h5py==3.12.1
fire==0.7.0
tokenizers==0.21.0
regex==2024.11.6
requests==2.32.3
urllib3==2.3.0
certifi==2025.1.31
charset-normalizer==3.4.1
idna==3.10
filelock==3.17.0
fsspec==2025.2.0
packaging==24.2
importlib-metadata==8.6.1
zipp==3.21.0
psutil==6.1.1
termcolor==2.5.0
tqdm==4.67.1
PyYAML==6.0.2
typing-extensions==4.12.2
networkx==3.4.2
jinja2==3.1.5
MarkupSafe==3.0.2
sympy==1.13.1
mpmath==1.3.0
PINS
if [[ ! -f "$WORK/env/.motioncrafter_pip_ok" ]]; then
  # Use the CUDA index only for these three binaries. Resolve their pinned
  # dependencies on PyPI, which also has the normal Linux utility wheels.
  "$PY" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3
  "$PY" -m pip install torch torchvision xformers diffusers transformers accelerate huggingface-hub \
    safetensors decord h5py fire numpy pillow
  # The decord 0.6.0 wheel records the tag cp36-cp36m in its WHEEL metadata, so pip check
  # reports it as unsupported although it imports fine (tested below). Ignore only that line.
  if ! check="$("$PY" -m pip check 2>&1)"; then
    if grep -v '^decord 0\.6\.0 is not supported on this platform$' <<< "$check" | grep -q .; then
      echo "$check" >&2; exit 1
    fi
  fi
  touch "$WORK/env/.motioncrafter_pip_ok"
fi

# The unified VAE constructor opens SD 1.5 with cache_dir="cache" relative to
# the official checkout. Point it at the same pinned Hub cache as run.py.
mkdir -p "$HF_HUB_CACHE"
if [[ ! -e cache && ! -L cache ]]; then ln -s "$HF_HUB_CACHE" cache; fi
[[ cache -ef "$HF_HUB_CACHE" ]] || { echo "Expected $REPO/cache to point to $HF_HUB_CACHE" >&2; exit 1; }

# Share repository IDs and revisions with the runner's provenance check.
export PYTHONPATH="$HERE"
"$PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from run_motioncrafter import CHECKPOINTS, MC, SVD, SD15, verify_cache

hub = Path(os.environ["HF_HUB_CACHE"])
files = {
    MC: ["unet_diff/*", "geometry_motion_vae/*"],
    SVD: ["model_index.json", "scheduler/scheduler_config.json",
          "feature_extractor/preprocessor_config.json", "image_encoder/config.json",
          "image_encoder/model.fp16.safetensors", "vae/config.json",
          "vae/diffusion_pytorch_model.fp16.safetensors"],
    SD15: ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"],
}
def complete(repo, snapshot):
    if repo == MC:
        for component in ("unet_diff", "geometry_motion_vae"):
            folder = snapshot / component
            if not (folder / "config.json").is_file() or not any(
                p.is_file() and p.suffix in (".bin", ".safetensors") for p in folder.glob("*")
            ):
                return False
        return True
    return all((snapshot / name).is_file() for name in files[repo])

for repo, revision in CHECKPOINTS.items():
    model_cache = hub / ("models--" + repo.replace("/", "--"))
    ref = model_cache / "refs/main"
    snapshot = model_cache / "snapshots" / revision
    marker = model_cache / ".motioncrafter_download_ok"
    if ref.exists() and ref.read_text().strip() != revision:
        raise SystemExit(f"Conflicting cached main for {repo}: {ref.read_text().strip()} != {revision}")
    if not (marker.is_file() and ref.is_file() and complete(repo, snapshot)):
        path = Path(snapshot_download(repo, revision=revision, cache_dir=hub, allow_patterns=files[repo]))
        if path != snapshot or not complete(repo, snapshot):
            raise SystemExit(f"Incomplete pinned download for {repo} at {path}")
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)
        marker.touch()
    resolved = Path(snapshot_download(repo, revision="main", cache_dir=hub, local_files_only=True))
    if resolved.name != revision:
        raise SystemExit(f"Offline main for {repo} resolved to {resolved}")
verify_cache(hub)
PY

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
"$WORK/env/bin/ffmpeg" -hide_banner -h encoder=libx264rgb 2>&1 | grep 'Encoder libx264rgb' > /dev/null
"$PY" - <<'PY'
# Import torch first, as run.py does: importing decord before torchvision segfaults.
import torch, torchvision, xformers.ops
import decord, h5py
import run
if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected the CUDA 12.4 torch wheel, found {torch.version.cuda}")
print("MotionCrafter imports OK, torch", torch.__version__, "CUDA", torch.version.cuda)
PY
if [[ ! -f "$WORK/env/.motioncrafter_ok" ]]; then
  echo "Checking all model loads on CPU with the network disabled"
  CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
import os
import torch
from run_motioncrafter import MC, SVD
from motioncrafter import MotionCrafterDiffPipeline, UNetSpatioTemporalConditionModelVid2vid, UnifyAutoencoderKL
cache = os.environ["HF_HUB_CACHE"]
unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
    MC, subfolder="unet_diff", cache_dir=cache, torch_dtype=torch.float16, low_cpu_mem_usage=True)
UnifyAutoencoderKL.from_pretrained(MC, subfolder="geometry_motion_vae", cache_dir=cache,
                                   torch_dtype=torch.float32, low_cpu_mem_usage=True)
MotionCrafterDiffPipeline.from_pretrained(SVD, unet=unet, cache_dir=cache,
                                          torch_dtype=torch.float16, variant="fp16")
PY
  touch "$WORK/env/.motioncrafter_ok"
fi
echo "Setup finished. Run bash $HERE/run_motioncrafter.sbatch or submit it with sbatch."
