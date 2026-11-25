#!/bin/bash
# Submit from zeromsf/ with sbatch, or use bash run_zeromsf.sbatch on Runpod.
# Example: bash run_zeromsf.sbatch --clips pour_lid pen
#SBATCH --job-name=zeromsf
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=zeromsf_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${ZEROMSF_WORK:-$HERE/work}"
[ -f "$HERE/run_zeromsf.py" ] || { echo "Submit from the zeromsf folder" >&2; exit 1; }
[ -f "$WORK/env/.zeromsf_ok" ] || { echo "Run setup_env.sh first for $WORK" >&2; exit 1; }
WORK="$(cd "$WORK" && pwd)"
export PATH="$WORK/env/bin:$PATH" LD_LIBRARY_PATH="$WORK/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache" XDG_CACHE_HOME="$WORK/cache"
export MPLCONFIGDIR="$WORK/mpl_cache" MPLBACKEND=Agg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "$HERE/run_zeromsf.py" --repo-dir "$WORK/zero-msf" "$@"
echo "Finished $(date)"
