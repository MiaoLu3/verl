#!/bin/bash
# Eval the SFT-finetuned Qwen3-0.6B on ALFWorld valid_seen.
# Uses our standalone async vllm eval pipeline.

#SBATCH --job-name=alfworld-eval-qwen3-0.6b-sft-v1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=1:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld Qwen3-0.6B SFT-v1 eval"
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

CKPT_PATH="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_0.6b_sft_v1"
# NB: SFT ckpt layout is `global_step_<N>/{model_world_size_*.pt,huggingface/}`
# (no `/actor` subdir, unlike the RL ckpts). model_merger merge takes the
# global_step dir directly:
#   python -m verl.model_merger merge --backend fsdp \
#     --local_dir checkpoints/sft/.../global_step_780 \
#     --target_dir <CKPT_PATH> --use_cpu_initialization

TS="$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen3_0.6b_sft_v1_eval_${SLURM_JOB_ID}_${TS}"
mkdir -p "${DUMP_DIR}"
echo "Dump dir: ${DUMP_DIR}"
echo "Ckpt path: ${CKPT_PATH}"

ls "${CKPT_PATH}" >/dev/null 2>&1 \
    || { echo "ERROR: ckpt not readable: ${CKPT_PATH}"; exit 1; }

# pre-run cleanup
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true

# fail-fast
python -c "import verl; print('verl', verl.__file__)"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"

T_START=$(date +%s)

python -m recipe.alfworld.eval_standalone \
    --model_path "${CKPT_PATH}" \
    --tp 1 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split valid_seen \
    --pool_size 16 --concurrency 16 \
    --max_samples -1 \
    --seed_base 1042 \
    --history_length 0 --max_steps 50 \
    --prompt_length 4096 --response_length 12288 \
    --max_assistant_turns 50 --max_user_turns 50 \
    --max_tokens_per_turn 4096 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --dump_dir "${DUMP_DIR}" \
    --num_cpus 8

EXIT_CODE=$?
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo "=============================================="
echo "Wall time: ${ELAPSED}s ($((ELAPSED / 60))m)"
echo "Dump dir: ${DUMP_DIR}"
JSONL_COUNT=$(find "${DUMP_DIR}" -name '*.jsonl' 2>/dev/null | wc -l)
echo "JSONL trajectory count: ${JSONL_COUNT}"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
