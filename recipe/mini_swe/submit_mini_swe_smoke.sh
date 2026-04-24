#!/bin/bash
# =============================================================================
# Mini-swe-agent x SWE-bench smoke test on 2 GPUs.
# =============================================================================
# Wraps ``recipe/mini_swe/scripts/run_mini_swe.sh`` with cleanup, fail-fast
# env / SIF checks, and stub train/val parquets. Mirrors the structure of
# ``recipe/alfworld/submit_alfworld_qwen3_1.7b_val.sh``.
#
# Usage:
#   sbatch recipe/mini_swe/submit_mini_swe_smoke.sh
# =============================================================================

#SBATCH --job-name=mswe-smoke
#SBATCH -A marlowe-m000069-pm05
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

echo "=============================================="
echo "mini-swe-agent x SWE-bench smoke -- job starting"
echo "Job ID:     ${SLURM_JOB_ID:-unknown}"
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "=============================================="

# -- env activation ----------------------------------------------------------
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/m000069-pm05/miaolu/conda_env/verl-agent

UPSTREAM_VERL="/scratch/m000069-pm05/miaolu/verl"
MINI_SWE_RECIPE="${UPSTREAM_VERL}/recipe/mini_swe"
cd "${UPSTREAM_VERL}"

export PYTHONPATH="${UPSTREAM_VERL}:${PYTHONPATH:-}"
export HF_HOME=/scratch/m000069/miaolu/.cache/huggingface
export SIF_CACHE_DIR=${SIF_CACHE_DIR:-/scratch/m000069-pm05/miaolu/swebench_sifs}
export PYTHONUNBUFFERED=1
export RAY_agent_register_timeout_ms=300000
# Smoke run: we only have 1 SIF (django__django-11099) in cache, so train
# batch must be 1 or DataLoader will be empty. rollout_n=2 still gives 2
# trajectories per step, enough for GRPO to compute an advantage.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export ROLLOUT_N=${ROLLOUT_N:-2}

# Trajectory dump sink -- ON by default so we can forensically inspect failed
# rollouts. Each rollout writes a per-turn JSONL: {turn, text, actions, outputs,
# returncodes} plus a trailing summary. Disk cost is small (~1 MB per rollout).
export MSWE_TRAJ_DUMP_DIR="${MSWE_TRAJ_DUMP_DIR:-${UPSTREAM_VERL}/trajectories/mini_swe_smoke_${SLURM_JOB_ID:-adhoc}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${MSWE_TRAJ_DUMP_DIR}"
echo "Trajectory dump dir: ${MSWE_TRAJ_DUMP_DIR}"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "SIF_CACHE_DIR:   ${SIF_CACHE_DIR}"

# -- preflight: apptainer available ------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found on PATH"
    exit 1
fi
apptainer --version

# -- preflight: zombie instance cleanup --------------------------------------
echo "--- zombie instance cleanup (pre-run) ---"
apptainer instance list 2>/dev/null | awk '$1 ~ /^mswe_/ {print $1}' \
    | xargs -r -n1 apptainer instance stop || true

# -- preflight: SIF cache must be non-empty ----------------------------------
if [ ! -d "${SIF_CACHE_DIR}" ]; then
    echo "ERROR: SIF_CACHE_DIR does not exist: ${SIF_CACHE_DIR}"
    exit 1
fi
SIF_COUNT=$(find "${SIF_CACHE_DIR}" -maxdepth 1 -name '*.sif' -type f 2>/dev/null | wc -l)
if [ "${SIF_COUNT}" -lt 1 ]; then
    echo "ERROR: no .sif files in SIF_CACHE_DIR=${SIF_CACHE_DIR}"
    echo "Run recipe/mini_swe/scripts/submit_swebench_pull.sh first."
    exit 1
fi
echo "SIF count in cache: ${SIF_COUNT}"

# -- preflight: fail-fast imports --------------------------------------------
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.mini_swe.agent_loop import MiniSweAgentLoop; print('MiniSweAgentLoop OK')"
python -c "from recipe.mini_swe.dataset import SweBenchDataset; print('SweBenchDataset OK')"
python -c "import vllm, torch, transformers, ray; print(f'vllm={vllm.__version__} torch={torch.__version__} transformers={transformers.__version__} ray={ray.__version__}')"

# -- preflight: ray cleanup --------------------------------------------------
echo "--- ray pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"

# -- stub parquets (non-null sentinels for data.{train,val}_files) ----------
[ -f /tmp/swebench_stub_train.parquet ] || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/swebench_stub_train.parquet')"
[ -f /tmp/swebench_stub_val.parquet ]   || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/swebench_stub_val.parquet')"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

# -- run ---------------------------------------------------------------------
set +e
bash "${MINI_SWE_RECIPE}/scripts/run_mini_swe.sh"
EXIT_CODE=$?
set -e

# -- postflight: zombie instance cleanup -------------------------------------
echo "--- zombie instance cleanup (post-run) ---"
apptainer instance list 2>/dev/null | awk '$1 ~ /^mswe_/ {print $1}' \
    | xargs -r -n1 apptainer instance stop || true

echo "=============================================="
echo "mini-swe-agent smoke exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
