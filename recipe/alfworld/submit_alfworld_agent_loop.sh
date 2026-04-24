#!/bin/bash
# =============================================================================
# ALFWorld agent-loop smoke test on Marlowe -- MULTI-GPU (2 GPU, TP=2)
# =============================================================================
# Mirrors verl-agent/submit_agent_loop_smoke_multigpu.sh but runs upstream
# verl's AlfWorldAgentLoop (recipe/alfworld/). Uses TP=2 so rollout_world_size
# is 2 and only one FSDP rank binds the ZMQ sender socket (avoids the
# EADDRINUSE race diagnosed on 284527/284529).
#
# Usage:
#   sbatch recipe/alfworld/submit_alfworld_agent_loop.sh
# =============================================================================

#SBATCH --job-name=alfworld-al-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=1:30:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld agent-loop smoke (MULTI-GPU) -- job starting"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Conda env (verl-agent env -- bundled verl 0.3.1.dev is shadowed by our
# upstream-verl PYTHONPATH prepend below)
# -----------------------------------------------------------------------------
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/m000069-pm05/miaolu/conda_env/verl-agent

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
UPSTREAM_VERL="/scratch/m000069-pm05/miaolu/verl"
ALFWORLD_RECIPE="${UPSTREAM_VERL}/recipe/alfworld"
cd "${UPSTREAM_VERL}"

# Prepend UPSTREAM verl so `import verl` resolves to it, not verl-agent bundled.
export PYTHONPATH="${UPSTREAM_VERL}:${PYTHONPATH:-}"

# HF model/dataset cache
export HF_HOME=/scratch/m000069/miaolu/.cache/huggingface

# ALFWorld data root
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}

# Real-time stdout
export PYTHONUNBUFFERED=1

# Ray agent registration timeout (shared FS can be slow)
export RAY_agent_register_timeout_ms=300000

# WandB
export WANDB_ENTITY=miaolu-stanford-university
export WANDB_PROJECT=verl_agent_alfworld

# Local JSONL trajectory dumper (opt-in via env var; zero overhead otherwise).
# One <request_id>.jsonl file per episode lands here, for downstream SFT curation.
export ALFWORLD_TRAJ_DUMP_DIR="${UPSTREAM_VERL}/trajectories/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ALFWORLD_TRAJ_DUMP_DIR}"
echo "Trajectory dump dir: ${ALFWORLD_TRAJ_DUMP_DIR}"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "ALFWORLD_DATA:   ${ALFWORLD_DATA}"
echo "PYTHONPATH head: ${PYTHONPATH%%:*}"

# -----------------------------------------------------------------------------
# Cleanup stale IPC/ray state from prior runs that crashed on this node
# -----------------------------------------------------------------------------
echo "--- pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ls /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null && echo "WARN: sockets still present" || echo "zmq sockets: clean"
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"
echo "-----------------------"

# -----------------------------------------------------------------------------
# Fail fast if verl resolves to the wrong copy, or recipe/data is missing.
# -----------------------------------------------------------------------------
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop; print('AlfWorldAgentLoop OK')"
python -c "from recipe.alfworld.alfworld_dataset import AlfWorldDataset; print('AlfWorldDataset OK')"
python -c "import vllm, torch, transformers, ray; print(f'vllm={vllm.__version__} torch={torch.__version__} transformers={transformers.__version__} ray={ray.__version__}')"

# Gamefiles sanity
GAMEFILES_ROOT="${ALFWORLD_DATA}/json_2.1.1/train"
if [ ! -d "${GAMEFILES_ROOT}" ]; then
    echo "ERROR: ALFWorld train gamefiles root missing: ${GAMEFILES_ROOT}"
    exit 1
fi
GAMEFILE_COUNT=$(ls "${GAMEFILES_ROOT}" 2>/dev/null | wc -l)
[ "${GAMEFILE_COUNT}" -gt 0 ] || { echo "ERROR: gamefiles empty"; exit 1; }
echo "alfworld gamefile count: ${GAMEFILE_COUNT}"
echo "alfworld data: OK (${GAMEFILES_ROOT})"

# Create sentinel stub parquets so data.train_files / data.val_files path
# checks don't fail. AlfWorldDataset ignores the content.
mkdir -p /tmp/alfworld_stub
if [ ! -f /tmp/alfworld_stub_train.parquet ]; then
    python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_train.parquet')"
fi
if [ ! -f /tmp/alfworld_stub_val.parquet ]; then
    python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_val.parquet')"
fi

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

bash "${ALFWORLD_RECIPE}/run_alfworld.sh"

EXIT_CODE=$?
echo "=============================================="
echo "ALFWorld smoke exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
