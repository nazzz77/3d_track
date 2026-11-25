#!/bin/bash
# Run on a Linux login node or Runpod before submitting the offline job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
WORK="${ZEROMSF_WORK:-$HERE/work}"
COMMIT=b4ea29f8c7c2c6aeb4c91d27f8fa9311f6786f79
MODEL=zmsf_shift+dynamicreplica+vkitti+kubric+pointodyssey+spring_adap_241106
DRIVE_ID=1F0A1uicevtLvSnZlrSzOW0VREohbsUcT

echo "Checking preprocessed clips in $ROOT/preprocessed_data"
for file in "$ROOT/common/track3d_data.py" "$ROOT"/preprocessed_data/{pour_lid,pen,stir,shoelace,paper_fold,shirt_fold}/{clip.npz,meta.json}; do
    [ -s "$file" ] || { echo "Missing $file" >&2; exit 1; }
done

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
ENV="$WORK/env"
PY="$ENV/bin/python"
REPO="$WORK/zero-msf"
export PIP_CACHE_DIR="$WORK/pip_cache" CONDA_PKGS_DIRS="$WORK/conda_pkgs"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
export XDG_CACHE_HOME="$WORK/cache" MPLCONFIGDIR="$WORK/mpl_cache" MPLBACKEND=Agg
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"
rm -f "$ENV/.zeromsf_ok"

if [ ! -x "$WORK/miniforge/bin/conda" ]; then
    curl --fail --location --retry 3 \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o "$WORK/miniforge.sh"
    bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge"
    rm "$WORK/miniforge.sh"
fi
if [ ! -x "$PY" ]; then
    # Open3D's eager import needs the GL runtime even on a headless node.
    "$WORK/miniforge/bin/conda" create -y -p "$ENV" --override-channels -c conda-forge \
        python=3.10.16 pip=25.0.1 libgl=1.7.0 libgomp=14.2.0
fi
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Initialize recursively after pinning the parent so Git fetches its recorded gitlinks.
github=(git -c 'url.https://github.com/.insteadOf=git@github.com:')
if [ ! -d "$REPO/.git" ]; then
    "${github[@]}" clone --no-recurse-submodules https://github.com/NVlabs/zero-msf.git "$REPO"
fi
cd "$REPO"
git checkout --quiet --detach "$COMMIT"
"${github[@]}" submodule update --init --recursive --checkout
for entry in \
    zmsf/mast3r:f5209afc300cec36239a7ac992263f36847bbba0 \
    zmsf/mast3r/dust3r:3cc8c88c413bb9e34c41db0e0eef99c2ee010b12 \
    zmsf/mast3r/dust3r/croco:d7de0705845239092414480bd829228723bf20de; do
    [ "$(git -C "${entry%:*}" rev-parse HEAD)" = "${entry#*:}" ] || {
        echo "Wrong submodule revision for ${entry%:*}" >&2; exit 1;
    }
done

if [ ! -f "$ENV/.zeromsf_pip_ok" ]; then
    # Pin the CUDA wheels across both installs. Constraints on transitive dependencies
    # (including Open3D's Python dependencies) do not request extra packages.
    cat > "$WORK/constraints.txt" <<'PINS'
torch==2.6.0
torchvision==0.21.0
numpy==1.26.4
pillow==11.1.0
filelock==3.17.0
typing-extensions==4.12.2
networkx==3.4.2
jinja2==3.1.5
fsspec==2024.12.0
sympy==1.13.1
mpmath==1.3.0
setuptools==75.8.0
requests==2.32.3
certifi==2025.1.31
charset-normalizer==3.4.1
idna==3.10
urllib3==2.3.0
beautifulsoup4==4.13.3
soupsieve==2.6
pysocks==1.7.1
contourpy==1.3.1
cycler==0.12.1
fonttools==4.56.0
kiwisolver==1.4.8
pyparsing==3.2.1
python-dateutil==2.9.0.post0
six==1.17.0
dash==2.18.2
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
flask==3.0.3
werkzeug==3.0.6
markupsafe==3.0.2
itsdangerous==2.2.0
click==8.1.8
blinker==1.9.0
plotly==5.24.1
tenacity==9.0.0
importlib-metadata==8.6.1
zipp==3.21.0
retrying==1.3.4
nest-asyncio==1.6.0
configargparse==1.7
addict==2.4.0
pyquaternion==0.9.9
pandas==2.2.3
pytz==2025.1
tzdata==2025.1
scikit-learn==1.6.1
joblib==1.4.2
threadpoolctl==3.5.0
nbformat==5.10.4
fastjsonschema==2.21.1
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
referencing==0.36.2
rpds-py==0.22.3
attrs==25.1.0
jupyter-core==5.7.2
platformdirs==4.3.6
traitlets==5.14.3
ipywidgets==8.1.5
comm==0.2.2
widgetsnbextension==4.0.13
jupyterlab-widgets==3.0.13
ipython==8.32.0
decorator==5.1.1
exceptiongroup==1.2.2
jedi==0.19.2
parso==0.8.4
matplotlib-inline==0.1.7
pexpect==4.9.0
ptyprocess==0.7.0
prompt-toolkit==3.0.50
wcwidth==0.2.13
pygments==2.19.1
stack-data==0.6.3
asttokens==3.0.0
executing==2.2.0
pure-eval==0.2.3
PINS
    "$PY" -m pip install -c "$WORK/constraints.txt" torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
    # zmsf eagerly imports Open3D and pyplot. DUSt3R localization imports cv2
    # and quaternion even though the camera-space demo never calls PnP.
    # Use CPU Open3D and headless OpenCV. The Hub mixin needs only huggingface-hub.
    # scikit-image, plyfile and iopath have no imports on this path. tqdm is used
    # by the Hub and gdown. viser belongs to the local viewer. Skip the optional
    # poselib, pycolmap and tensorboard backends. Without compiled curope, CroCo
    # uses its slower PyTorch RoPE implementation.
    "$PY" -m pip install -c "$WORK/constraints.txt" \
        numpy==1.26.4 Pillow==11.1.0 PyYAML==6.0.2 huggingface-hub==0.28.1 \
        einops==0.8.1 scipy==1.13.1 packaging==24.2 opencv-python-headless==4.10.0.84 \
        numpy-quaternion==2024.0.7 open3d-cpu==0.19.0 matplotlib==3.9.4 \
        tqdm==4.67.1 gdown==5.2.0
    "$PY" -m pip check
    touch "$ENV/.zeromsf_pip_ok"
fi

echo "Checking official imports"
HF_HUB_OFFLINE=1 "$PY" - <<'PY'
import torch
import torchvision
from demo import instantiate_component, prepare_input_data, run_inference
from zmsf import AsymmetricMASt3R_flow
from zmsf.eval_models import St3RModel
assert torch.__version__.split("+")[0] == "2.6.0" and torch.version.cuda == "12.4"
assert torchvision.__version__.split("+")[0] == "0.21.0"
print("ZeroMSF imports OK with torch", torch.__version__)
PY

# The YAML sets pretrained to null. The strict predictor state_dict below is the
# entire model. The demo never calls from_pretrained or downloads a backbone.
CKPT="$REPO/checkpoints/${MODEL}_last.ckpt"
META="$CKPT.json"
if [ ! -s "$CKPT" ] || [ ! -s "$META" ] || [ "$CKPT" -nt "$META" ]; then
    candidate="$CKPT"
    if [ ! -s "$CKPT" ]; then
        candidate="$CKPT.part"
        "$ENV/bin/gdown" --no-cookies "$DRIVE_ID" -O "$candidate"
    fi
    CUDA_VISIBLE_DEVICES="" "$PY" - "$candidate" "$CKPT" "$DRIVE_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch
import yaml

src, dst = map(Path, sys.argv[1:3])
config = yaml.safe_load((Path("zmsf/configs") / (dst.name.removesuffix("_last.ckpt") + ".yaml")).read_text())
assert Path(config["init_args"]["weight_path"]).resolve() == dst
assert config["init_args"]["pretrained"] is None
with src.open("rb") as stream:
    prefix = stream.read(4096).lstrip().lower()
    if not prefix or b"<html" in prefix or prefix.startswith(b"<!doctype html"):
        raise ValueError(f"Expected a checkpoint at {src}, got HTML or an empty file")
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b""):
        digest.update(block)
state = torch.load(src, map_location="cpu", weights_only=True)["state_dict"]
if not any(k.startswith("predictor.") for k in state):
    raise ValueError("Missing predictor weights")
record = dict(file=dst.name, drive_id=sys.argv[3], size_bytes=src.stat().st_size, sha256=digest.hexdigest())
if src != dst:
    src.replace(dst)
manifest = Path(str(dst) + ".json")
tmp = Path(str(manifest) + ".tmp")
tmp.write_text(json.dumps(record, indent=2) + "\n")
tmp.replace(manifest)
PY
fi
cat "$META"
touch "$ENV/.zeromsf_ok"
echo "Setup finished. Run bash run_zeromsf.sbatch or submit from $HERE."
