#!/bin/bash
# =============================================================================
# SWE-bench SIF pre-pull SLURM array wrapper.
# =============================================================================
# One array task per instance. Each task invokes the CLI with
# --index $SLURM_ARRAY_TASK_ID, which resolves to exactly one NormalizedRow
# from the chosen dataset and pulls that instance's docker image into
# $CACHE_DIR/<instance_id>.sif.
#
# The CLI is resume-safe (skips when the SIF already exists at >=
# --min-sif-bytes) and uses fcntl.flock + atomic rename, so overlapping
# array tasks on the same index are safe.
#
# Usage:
#   # SWE-bench Lite = 300 rows, so --array=0-299%8 (8 concurrent pulls)
#   DATASET_NAME=swe_bench_lite CACHE_DIR=/scratch/m000069-pm05/miaolu/swebench_sifs \
#       sbatch --array=0-299%8 recipe/mini_swe/scripts/submit_swebench_pull.sh
#
#   # Smoke test: just 3 instances
#   DATASET_NAME=swe_bench_lite CACHE_DIR=/scratch/m000069-pm05/miaolu/swebench_sifs \
#       sbatch --array=0-2 recipe/mini_swe/scripts/submit_swebench_pull.sh
# =============================================================================

#SBATCH --job-name=mswe-sif-pull
#SBATCH -A marlowe-m000069-pm05
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# NOTE: the submitter must pass --array=0-N via the sbatch command line.

set -euo pipefail

DATASET=${DATASET_NAME:-swe_bench_lite}
CACHE=${CACHE_DIR:-/scratch/m000069-pm05/miaolu/swebench_sifs}
SPLIT=${SPLIT:-}
TIMEOUT=${TIMEOUT_PER_PULL:-1800}
MIN_BYTES=${MIN_SIF_BYTES:-10000000}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-N." >&2
    exit 2
fi

echo "=============================================="
echo "mswe-sif-pull -- job starting"
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:            ${SLURMD_NODENAME:-unknown}"
echo "Dataset:         ${DATASET}"
echo "Cache dir:       ${CACHE}"
echo "Split override:  ${SPLIT:-<adapter default>}"
echo "Timeout/pull:    ${TIMEOUT}s"
echo "Min SIF bytes:   ${MIN_BYTES}"
echo "=============================================="

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/m000069-pm05/miaolu/conda_env/verl-agent

UPSTREAM_VERL="/scratch/m000069-pm05/miaolu/verl"
cd "${UPSTREAM_VERL}"

export PYTHONPATH="${UPSTREAM_VERL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/m000069/miaolu/.cache/huggingface}

mkdir -p "${CACHE}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CACHE}/.apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

CLI_ARGS=(
    --dataset-name "${DATASET}"
    --cache-dir "${CACHE}"
    --index "${SLURM_ARRAY_TASK_ID}"
    --apptainer-cachedir "${APPTAINER_CACHEDIR}"
    --timeout-per-pull "${TIMEOUT}"
    --min-sif-bytes "${MIN_BYTES}"
)
if [[ -n "${SPLIT}" ]]; then
    CLI_ARGS+=(--split "${SPLIT}")
fi

set +e
python -m recipe.mini_swe.scripts.swebench_pull "${CLI_ARGS[@]}"
EXIT_CODE=$?
set -e

echo "=============================================="
echo "mswe-sif-pull exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
