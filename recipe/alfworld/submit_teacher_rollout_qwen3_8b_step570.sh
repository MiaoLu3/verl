#!/bin/bash
# Teacher rollout: Qwen3-8B RL ckpt (global_step_570, val=0.957) over the
# train split, 8 rollouts per game, T=0.4 (default), 2 GPUs tp=2, concurrency=128.
# Output goes to teacher_rollouts/qwen3_8b_rl_step570_T0.4/.

#SBATCH --job-name=alfworld-teacher-qwen3-8b-step570
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=4:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld TEACHER ROLLOUT — Qwen3-8B step 570"
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CKPT_PATH="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_8b_rl_step570"
RUN_DIR="${UPSTREAM_VERL}/teacher_rollouts/qwen3_8b_rl_step570_T0.4"

# Single fixed run dir (re-runnable / resumable). If you want a new fresh run,
# add a timestamp suffix yourself before sbatch.
mkdir -p "${RUN_DIR}"
echo "Teacher rollout run dir: ${RUN_DIR}"
echo "Teacher ckpt path:       ${CKPT_PATH}"

ls "${CKPT_PATH}" >/dev/null 2>&1 \
    || { echo "ERROR: ckpt path not readable: ${CKPT_PATH}"; exit 1; }

# pre-run cleanup
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true

# fail-fast imports
python -c "import verl; print('verl', verl.__file__)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"
python -c "from recipe.alfworld.alfworld_agent_loop import _extract_gamefile_id; print('agent_loop OK')"

GAMEFILES_ROOT="${ALFWORLD_DATA}/json_2.1.1/train"
GAMEFILE_COUNT=$(find "${GAMEFILES_ROOT}" -name 'game.tw-pddl' 2>/dev/null | wc -l)
[ "${GAMEFILE_COUNT}" -gt 0 ] || { echo "ERROR: train gamefiles empty at ${GAMEFILES_ROOT}"; exit 1; }
echo "alfworld train gamefile count: ${GAMEFILE_COUNT}"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

T_START=$(date +%s)

python -m recipe.alfworld.eval_standalone \
    --model_path "${CKPT_PATH}" \
    --tp 2 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split train \
    --pool_size 128 --concurrency 128 \
    --rollouts_per_game 8 \
    --seed_base 7000 \
    --history_length 0 --max_steps 50 \
    --prompt_length 4096 --response_length 12288 \
    --max_assistant_turns 50 --max_user_turns 50 \
    --max_tokens_per_turn 4096 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --teacher_rollout_dir "${RUN_DIR}" \
    --ckpt_step 570 \
    --num_cpus 16

EXIT_CODE=$?
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo "=============================================="
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m)"
echo "Run dir:    ${RUN_DIR}"
JSONL_COUNT=$(find "${RUN_DIR}/by_task_type" -name '*.jsonl' 2>/dev/null | wc -l)
echo "JSONL trajectory count: ${JSONL_COUNT} / planned 28424"
echo "Teacher rollout exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
