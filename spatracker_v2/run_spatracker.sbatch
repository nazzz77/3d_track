#!/bin/bash
# Slurm job for UMD Zaratan: run SpatialTrackerV2 on all 6 clips (settings rgb and rgbd_fixed).
# Run setup_env.sh once on a login node first, then submit from this folder:
#
#   cd /scratch/zt1/project/<PROJECT>/user/<UMD_ID>/track3d/spatracker_v2
#   sbatch run_spatracker.sbatch
#
# Watch it with `squeue -u $USER` and `tail -f spatracker_v2_<JOBID>.log`.
# If it fails or times out, submit it again: finished runs are skipped.
#
# GPU: one H100 (80 GB, 144 SU/hour). The rgb setting runs a large front model over up to 185
# frames; on an A100 (40 GB, 48 SU/hour) long clips may run out of memory and fall back to fewer
# frames. To use an A100 anyway, change the --gpus line to a100:1.
#SBATCH --job-name=spatracker_v2
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=spatracker_v2_%j.log
#SBATCH --mail-type=END,FAIL
# If you belong to more than one allocation (check with `sbalance`), uncomment and fill in:
##SBATCH --account=<ALLOCATION>
set -euo pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORK="${SPA_WORK:-$HERE/work}"
[ -f "$HERE/run_spatracker.py" ] || { echo "Submit this job from the spatracker_v2 folder"; exit 1; }
[ -f "$WORK/env/.spatrack_ok" ] || { echo "Environment not found in $WORK; run setup_env.sh on a login node first"; exit 1; }

export PATH="$WORK/env/bin:$PATH"     # python and ffmpeg from the env
export HF_HOME="$WORK/hf_cache"       # checkpoints downloaded by setup_env.sh
export HF_HUB_OFFLINE=1               # compute nodes have no internet access
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
# Match CPU threads to the CPUs actually granted. Containers (e.g. Runpod) can show every core of
# the host while capping CPU time, and hundreds of OpenMP threads then stall model loading.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ${SLURM_JOB_ID:-(not under Slurm)} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "$HERE/run_spatracker.py" --repo-dir "$WORK/SpaTrackerV2" "$@"
echo "Finished $(date)"
