#!/bin/bash
# Smoke test for the teacher-rollout dump format.
# 1xH100, Qwen3-0.6B, 16 episodes, concurrency=8, train split,
# rollouts_per_game=2 (so each of 8 unique games gets 2 trajectories).

#SBATCH --job-name=alfworld-teacher-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=0:30:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld TEACHER ROLLOUT smoke test"
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

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/scratch/m000069-pm05/miaolu/verl/teacher_rollouts/smoke_${SLURM_JOB_ID}_${TS}"
mkdir -p "${RUN_DIR}"
echo "Teacher rollout dir: ${RUN_DIR}"

# pre-run cleanup
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true

# Fail-fast imports
python -c "import verl; print('verl', verl.__file__)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"
python -c "from recipe.alfworld.alfworld_agent_loop import _extract_gamefile_id, _extract_task_type, _next_rollout_idx; print('helpers OK')"
python -c "from recipe.alfworld.jsonl_to_parquet import load_record; print('parquet converter OK')"

T_START=$(date +%s)

python -m recipe.alfworld.eval_standalone \
    --model_path Qwen/Qwen3-0.6B \
    --tp 1 --max_model_len 16384 --gpu_memory_utilization 0.7 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split train \
    --pool_size 1 --concurrency 8 \
    --max_samples 8 --rollouts_per_game 2 \
    --seed_base 9000 \
    --history_length 0 --max_steps 10 \
    --prompt_length 4096 --response_length 8192 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --max_tokens_per_turn 1024 \
    --teacher_rollout_dir "${RUN_DIR}" \
    --num_cpus 8

EXIT_CODE=$?
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo "=============================================="
echo "Wall time: ${ELAPSED}s"
echo "Run dir: ${RUN_DIR}"
echo "metadata.json:" "$(cat ${RUN_DIR}/metadata.json | python -c 'import sys,json;d=json.load(sys.stdin);print(d.get(\"started_at\"),d.get(\"total_episodes_planned\"))' 2>/dev/null || echo MISSING)"
echo "summary.json:" "$(cat ${RUN_DIR}/summary.json | python -c 'import sys,json;d=json.load(sys.stdin);print(d.get(\"num_episodes\"),d.get(\"wins\"))' 2>/dev/null || echo MISSING)"

# Tree output
echo "--- run dir tree ---"
find "${RUN_DIR}" -maxdepth 3 -type f | sort | head -40
JSONL_COUNT=$(find "${RUN_DIR}/by_task_type" -name '*.jsonl' 2>/dev/null | wc -l)
echo "JSONL trajectory files (under by_task_type/): ${JSONL_COUNT}"
echo "expected: 8 unique games * 2 rollouts = 16"

# Validate one JSONL has the expected new fields
SAMPLE_JSONL="$(find ${RUN_DIR}/by_task_type -name '*.jsonl' | head -1)"
if [ -n "${SAMPLE_JSONL}" ]; then
    echo "--- field check on ${SAMPLE_JSONL} ---"
    python -c "
import json,sys
d=json.load(open('${SAMPLE_JSONL}'))
required=['gamefile_id','task_type','rollout_idx','messages','tokens','lengths','model_path','ckpt_step','sampling_params']
missing=[k for k in required if k not in d]
print('missing fields:', missing or 'none')
print('gamefile_id:', d['gamefile_id'])
print('task_type:', d['task_type'])
print('rollout_idx:', d['rollout_idx'])
print('messages count:', len(d['messages']))
print('tokens.input_ids len:', len(d['tokens']['input_ids']))
print('tokens.loss_mask len:', len(d['tokens']['loss_mask']))
print('tokens.loss_mask sum:', sum(d['tokens']['loss_mask']))
print('lengths:', d['lengths'])
print('won:', d['won'])
print('sampling_params:', d['sampling_params'])
sys.exit(0 if not missing else 1)
"
fi

# Convert to parquet smoke
echo "--- jsonl_to_parquet smoke ---"
python -m recipe.alfworld.jsonl_to_parquet \
    --teacher_dir "${RUN_DIR}" \
    --out "${RUN_DIR}/sft_data_all.parquet" \
    --keep all || true
ls -la "${RUN_DIR}/sft_data_all.parquet" 2>/dev/null || echo "parquet missing"

echo "=============================================="
echo "Teacher smoke exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
