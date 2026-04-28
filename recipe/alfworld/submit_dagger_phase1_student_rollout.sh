#!/bin/bash
# =============================================================================
# DAgger Phase 1: roll the A-trained student on S35x64 part B's 64 games.
#
# Student ckpt is the same one that was used to generate the A-direct SFT
# baseline on the OCI cluster:
#   merged_hf/qwen3_0.6b_sft_S35x64_fwon8_prop_v1_A_1r_16ep_oci_v1
#
# We use --teacher_rollout_dir for the dumper format only (full tokens +
# parsed messages per turn) -- the rolling model is the student, not the
# teacher. Phase 2 then re-runs the teacher on each (student-state) prefix
# to emit the expert action label.
#
# Settings:
#   - split=train (B's games come from the train split)
#   - --gamefile_filter_json  -> 64 gids of S35x64 part B
#   - rollouts_per_game=1     -> matches SFT data convention
#   - T=0.4, top_p=1.0        -> matches SFT data generation
#   - concurrency=32 pool=32  -> matches recent eval throughput
#
# Usage:
#   sbatch recipe/alfworld/submit_dagger_phase1_student_rollout.sh
# =============================================================================

#SBATCH --job-name=dagger-p1-student-rollout-qwen3-0.6b-A-on-B
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=02:00:00

set -euo pipefail

echo "=============================================="
echo "DAgger Phase 1: student rollout (Qwen3-0.6B A-SFT on S35x64 part B)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-local}"
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
export TOKENIZERS_PARALLELISM=true
export RAY_agent_register_timeout_ms=300000
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# A-SFT student ckpt (1-rollout-per-game, 16 epochs, OCI). User scp's this in.
STUDENT_CKPT="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_0.6b_sft_S35x64_fwon8_prop_v1_A_1r_16ep_oci_v1"
FILTER_JSON="${UPSTREAM_VERL}/sft_data/splits/S35x64_fwon8_prop_v1/B_gids.json"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${UPSTREAM_VERL}/student_rollouts/dagger_qwen3_0.6b_S35x64_A_1r_16ep_on_B_${SLURM_JOB_ID:-local}_${TS}"
mkdir -p "${RUN_DIR}"

cleanup_runtime() {
    rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
    ray stop --force 2>&1 | tail -5 || true
    rm -rf /tmp/ray 2>/dev/null || true
}

# Fail-fast: ckpt + filter file must exist
test -d "${STUDENT_CKPT}"  || { echo "ERROR: student ckpt missing: ${STUDENT_CKPT}"; exit 1; }
test -f "${FILTER_JSON}"   || { echo "ERROR: filter json missing:  ${FILTER_JSON}"; exit 1; }

# Sanity-print the filter file
python -c "
import json
with open('${FILTER_JSON}') as f:
    d = json.load(f)
gids = d['gamefile_ids'] if isinstance(d, dict) else d
print(f'[phase1] filter_json has {len(gids)} gids; first 3: {sorted(gids)[:3]}')
"

python -c "import verl; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

cleanup_runtime

# ---- Phase 1: roll the A-SFT student on B's 64 games ----
echo "---> [phase1] student=${STUDENT_CKPT}"
echo "---> [phase1] filter=${FILTER_JSON}"
echo "---> [phase1] dump =${RUN_DIR}"
python -m recipe.alfworld.eval_standalone \
    --model_path "${STUDENT_CKPT}" \
    --tp 1 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split train \
    --gamefile_filter_json "${FILTER_JSON}" \
    --pool_size 32 --concurrency 32 \
    --max_samples -1 \
    --seed_base 1042 \
    --history_length 0 --max_steps 50 \
    --prompt_length 4096 --response_length 12288 \
    --max_assistant_turns 50 --max_user_turns 50 \
    --max_tokens_per_turn 4096 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --teacher_rollout_dir "${RUN_DIR}" \
    --rollouts_per_game 1 \
    --num_cpus 16
cleanup_runtime

EXIT_CODE=$?
echo "=============================================="
echo "DAgger Phase 1 exited with code ${EXIT_CODE}"
echo "Student rollouts: ${RUN_DIR}"
echo "  layout: <run>/by_task_type/<task>/<gid>__rollout_0.jsonl  (64 files expected)"
echo "  metadata.json + summary.json at run root"
echo "=============================================="
exit ${EXIT_CODE}
