#!/bin/bash
# Slurm job for UMD Zaratan: run DELTA on all 6 clips (settings rgb and rgbd_fixed).
# Run setup_env.sh once on a login node first, then submit from this folder:
#
#   cd /scratch/zt1/project/<PROJECT>/user/<UMD_ID>/track3d/delta
#   sbatch run_delta.sbatch
#
# On Runpod, use bash run_delta.sbatch. Both forms pass arguments to run_delta.py.
# Watch the Slurm job with `squeue -u $USER` and `tail -f delta_<JOBID>.log`.
# If it fails or times out, submit it again: finished runs are skipped.
#
# GPU: one H100 (80 GB, 144 SU/hour). An A100 (40 GB, 48 SU/hour) may run out of
# memory on long clips. To use an A100, change the --gpus line to a100:1.
#SBATCH --job-name=delta
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=delta_%j.log
#SBATCH --mail-type=END,FAIL
# If you belong to more than one allocation (check with `sbalance`), uncomment and fill in:
##SBATCH --account=<ALLOCATION>
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${DELTA_WORK:-$HERE/work}"
[ -f "$HERE/run_delta.py" ] || { echo "Submit this job from the delta folder"; exit 1; }
[ -f "$WORK/env/.delta_ok" ] || { echo "Environment not ready in $WORK; run setup_env.sh on a login node first"; exit 1; }
WORK="$(cd "$WORK" && pwd)"

export PATH="$WORK/env/bin:$PATH"     # python and ffmpeg from the env
export HF_HOME="$WORK/hf_cache" TORCH_HOME="$WORK/torch_cache"
export HF_HUB_OFFLINE=1               # compute nodes have no internet access
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
# Pods can show every core on the host while capping CPU time. Match the reference job's
# thread limit so model loading stays within the CPUs granted.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "$HERE/run_delta.py" --repo-dir "$WORK/DELTA" "$@"
echo "Finished $(date)"
