#!/bin/bash
# Submit from track3d/traceanything after running setup_env.sh on the login node.
# On an 80 GB Runpod GPU, use bash run_traceanything.sbatch. Both accept runner arguments.
# The official example with 40 views calls for at least 48 GB of GPU memory.
#SBATCH --job-name=traceanything
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=traceanything_%j.log
#SBATCH --mail-type=END,FAIL
##SBATCH --account=<ALLOCATION>
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${TRACEANYTHING_WORK:-$HERE/work}"
[ -f "$HERE/run_traceanything.py" ] || { echo "Submit from the traceanything folder"; exit 1; }
[ -f "$WORK/env/.traceanything_ok" ] || { echo "Run setup_env.sh first to prepare $WORK"; exit 1; }
WORK="$(cd "$WORK" && pwd)"

export PATH="$WORK/env/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
export LD_LIBRARY_PATH="$WORK/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache" MODELSCOPE_CACHE="$WORK/modelscope_cache"
export XDG_CACHE_HOME="$WORK/xdg_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "$HERE/run_traceanything.py" --repo-dir "$WORK/TraceAnything" "$@"
echo "Finished $(date)"
