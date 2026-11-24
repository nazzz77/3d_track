#!/bin/bash
# Submit from track4world/ on Zaratan, or run with bash on Runpod.
# Both forms forward arguments, e.g. --clips pen. Resubmission skips completed runs.
#SBATCH --job-name=track4world
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=track4world_%j.log
#SBATCH --mail-type=END,FAIL
##SBATCH --account=<ALLOCATION>
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${TRACK4WORLD_WORK:-$HERE/work}"
[ -f "$HERE/run_track4world.py" ] || { echo "Submit from the track4world folder" >&2; exit 1; }
[ -f "$WORK/env/.track4world_ok" ] || { echo "Run setup_env.sh first for $WORK" >&2; exit 1; }
WORK="$(cd "$WORK" && pwd)"
export PATH="$WORK/env/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
export HF_HUB_CACHE="$HF_HOME/hub" MODELSCOPE_CACHE="$WORK/modelscope_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$WORK/env/bin/python" "$HERE/run_track4world.py" --repo-dir "$WORK/Track4World" "$@"
echo "Finished $(date)"
