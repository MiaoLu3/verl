#!/bin/bash
# Smoke test for the concurrent eval_standalone refactor.
# 1xH100, Qwen3-0.6B, 8 episodes, 8 concurrency, train split.

#SBATCH --job-name=alfworld-eval-smoke
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
echo "ALFWorld concurrent eval SMOKE TEST"
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
SMOKE_DIR="/scratch/m000069-pm05/miaolu/verl/trajectories/teacher_smoke_${SLURM_JOB_ID}_${TS}"
mkdir -p "${SMOKE_DIR}"
echo "Smoke dump dir: ${SMOKE_DIR}"

# pre-run cleanup
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true

python -c "import verl; print('verl', verl.__file__)"
python -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)"
python -c "from vllm.v1.engine.async_llm import AsyncLLM; print('AsyncLLM OK')"
python -c "from recipe.alfworld.eval_standalone import StandaloneServerManager; print('StandaloneServerManager OK')"

SUMMARY_PATH="${SMOKE_DIR}/summary.json"

T_START=$(date +%s)
python -m recipe.alfworld.eval_standalone \
    --model_path Qwen/Qwen3-0.6B \
    --tp 1 --max_model_len 16384 --gpu_memory_utilization 0.7 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split valid_seen --pool_size 8 --concurrency 8 \
    --max_samples 8 --seed_base 9000 \
    --history_length 0 --max_steps 10 \
    --prompt_length 4096 --response_length 8192 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --max_tokens_per_turn 1024 \
    --dump_dir "${SMOKE_DIR}" \
    --summary_path "${SUMMARY_PATH}" \
    --num_cpus 8

EXIT_CODE=$?
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo "=============================================="
echo "Wall time: ${ELAPSED}s"
echo "Summary: ${SUMMARY_PATH}"
echo "Trajectory dump dir: ${SMOKE_DIR}"
ls -la "${SMOKE_DIR}" || true
JSONL_COUNT=$(find "${SMOKE_DIR}" -name '*.jsonl' 2>/dev/null | wc -l)
echo "JSONL trajectory files: ${JSONL_COUNT}"
if [ -f "${SUMMARY_PATH}" ]; then
    python - <<PY
import json
s = json.load(open("${SUMMARY_PATH}"))
print(f"summary num_episodes={s['num_episodes']} wins={s['wins']} sr={s['success_rate']}")
print(f"len(results)={len(s.get('results', []))}")
PY
fi
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
