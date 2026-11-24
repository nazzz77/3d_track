#!/bin/bash
# Submit from trackcraft3r/ on Zaratan. On Runpod use bash run_trackcraft3r.sbatch.
# Both forms pass arguments through, e.g. --clips pen --modes rgbd_fixed.
#SBATCH --job-name=trackcraft3r
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=trackcraft3r_%j.log
#SBATCH --mail-type=END,FAIL
##SBATCH --account=<ALLOCATION>
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${TRACKCRAFT3R_WORK:-$HERE/work}"
[ -f "$HERE/run_trackcraft3r.py" ] || { echo "Submit from the trackcraft3r folder" >&2; exit 1; }
[ -f "$WORK/env/.trackcraft3r_ok" ] || { echo "Run setup_env.sh first for $WORK" >&2; exit 1; }
export TRACKCRAFT3R_WORK="$(cd "$WORK" && pwd)"
WORK="$TRACKCRAFT3R_WORK"
export PATH="$WORK/env/bin:$PATH"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache" MODELSCOPE_CACHE="$WORK/wan_models"
export HF_HUB_CACHE="$HF_HOME/hub" XDG_CACHE_HOME="$WORK/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$WORK/env/bin/python" "$HERE/run_trackcraft3r.py" "$@"
echo "Finished $(date)"
