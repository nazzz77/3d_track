#!/bin/bash
# Submit from motioncrafter/ with sbatch, or invoke with bash on Runpod.
#SBATCH --job-name=motioncrafter
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=motioncrafter_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Slurm may execute a spool copy of this script.
if [[ ! -f "$HERE/run_motioncrafter.py" ]]; then HERE="${SLURM_SUBMIT_DIR:-$HERE}"; fi
[[ -f "$HERE/run_motioncrafter.py" ]] || { echo "Submit from the motioncrafter folder" >&2; exit 1; }
WORK="${MOTIONCRAFTER_WORK:-$HERE/work}"
[[ -f "$WORK/env/.motioncrafter_ok" ]] || { echo "Run setup_env.sh first for $WORK" >&2; exit 1; }
export MOTIONCRAFTER_WORK="$(cd "$WORK" && pwd)"
export PATH="$MOTIONCRAFTER_WORK/env/bin:$PATH"
export HF_HOME="$MOTIONCRAFTER_WORK/hf_cache" HF_HUB_CACHE="$MOTIONCRAFTER_WORK/hf_cache/hub"
export TORCH_HOME="$MOTIONCRAFTER_WORK/torch_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "$HERE/run_motioncrafter.py" "$@"
echo "Finished $(date)"
