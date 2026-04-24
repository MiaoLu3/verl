#!/bin/bash
# =============================================================================
# Standalone Qwen3-32B ALFWorld valid_seen eval — reuses AlfWorldAgentLoop +
# AlfWorldEnvPool but bypasses FSDP/main_ppo to dodge the 32B weight-duplication
# OOM (actor FSDP + vllm both holding 32B shards simultaneously).
#
# Launches `python -m recipe.alfworld.eval_standalone` with TP=4 on 4xH100,
# pool_size=1 (vllm engine is single-threaded anyway), valid_seen split,
# enable_thinking=True (default; Qwen3 chat template decides).
# =============================================================================

#SBATCH --job-name=alfworld-qwen3-32b-eval-standalone
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=3:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld Qwen3-32B STANDALONE eval — job starting"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "=============================================="

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/m000069-pm05/miaolu/conda_env/verl-agent

UPSTREAM_VERL="/scratch/m000069-pm05/miaolu/verl"
ALFWORLD_RECIPE="${UPSTREAM_VERL}/recipe/alfworld"
cd "${UPSTREAM_VERL}"

export PYTHONPATH="${UPSTREAM_VERL}:${PYTHONPATH:-}"
export HF_HOME=/scratch/m000069/miaolu/.cache/huggingface
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export PYTHONUNBUFFERED=1
export RAY_agent_register_timeout_ms=300000
export VLLM_USE_V1=1
# vLLM multiproc: must be "spawn" to avoid CUDA fork issue. Without this,
# imports (ray, torch, etc) touch CUDA before vLLM forks its workers, causing:
# "Cannot re-initialize CUDA in forked subprocess"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Local JSONL trajectory dumper — AlfWorldAgentLoop picks this up.
TS="$(date +%Y%m%d_%H%M%S)"
export ALFWORLD_TRAJ_DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen3_32b_eval_standalone_${TS}"
mkdir -p "${ALFWORLD_TRAJ_DUMP_DIR}"
echo "Trajectory dump dir: ${ALFWORLD_TRAJ_DUMP_DIR}"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "ALFWORLD_DATA:   ${ALFWORLD_DATA}"

# --- pre-run cleanup -------------------------------------------------
echo "--- pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"

# --- fail-fast imports -----------------------------------------------
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop; print('AlfWorldAgentLoop OK')"
python -c "from recipe.alfworld.alfworld_dataset import AlfWorldDataset; print('AlfWorldDataset OK')"
python -c "from recipe.alfworld.eval_standalone import StandaloneServerManager; print('StandaloneServerManager OK')"
python -c "import vllm, torch, transformers, ray; print(f'vllm={vllm.__version__} torch={torch.__version__} transformers={transformers.__version__} ray={ray.__version__}')"

GAMEFILES_ROOT="${ALFWORLD_DATA}/json_2.1.1/valid_seen"
GAMEFILE_COUNT=$(find "${GAMEFILES_ROOT}" -name 'game.tw-pddl' 2>/dev/null | wc -l)
[ "${GAMEFILE_COUNT}" -gt 0 ] || { echo "ERROR: valid_seen gamefiles empty at ${GAMEFILES_ROOT}"; exit 1; }
echo "valid_seen gamefile count: ${GAMEFILE_COUNT}"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

SUMMARY_PATH="${ALFWORLD_TRAJ_DUMP_DIR}/summary.json"

python -m recipe.alfworld.eval_standalone \
    --model_path "Qwen/Qwen3-32B" \
    --tp 4 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager \
    --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split valid_seen \
    --pool_size 8 \
    --seed_base 1042 \
    --history_length 0 \
    --max_steps 50 \
    --prompt_length 4096 \
    --response_length 12288 \
    --max_assistant_turns 50 \
    --max_user_turns 50 \
    --max_tokens_per_turn 4096 \
    --temperature 0.4 \
    --top_p 1.0 \
    --top_k -1 \
    --dump_dir "${ALFWORLD_TRAJ_DUMP_DIR}" \
    --summary_path "${SUMMARY_PATH}" \
    --num_cpus 16

EXIT_CODE=$?
echo "=============================================="
echo "Summary path: ${SUMMARY_PATH}"
echo "Trajectory dir: ${ALFWORLD_TRAJ_DUMP_DIR}"
echo "ALFWorld Qwen3-32B standalone eval exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
