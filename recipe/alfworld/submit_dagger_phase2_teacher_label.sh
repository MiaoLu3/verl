#!/bin/bash
# =============================================================================
# DAgger Phase 2: relabel student-visited states with the teacher's expert
# action.
#
# Reads a Phase-1 student-rollout dump and emits a single-turn SFT parquet
# whose rows are (student_state_prefix, teacher_action_at_that_state).
# Output schema is a strict superset of what PretokenizedSFTDataset wants
# (tokens_input_ids + tokens_loss_mask), so it slots straight into the SFT
# trainer.
#
# Usage (after editing the two _DIR paths below):
#   sbatch recipe/alfworld/submit_dagger_phase2_teacher_label.sh
# =============================================================================

#SBATCH --job-name=dagger-p2-teacher-label-qwen3-8b-step570
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=02:00:00

set -euo pipefail

echo "=============================================="
echo "DAgger Phase 2: teacher-label student states"
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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- inputs / outputs ----
# Point STUDENT_RUN_DIR at the Phase-1 dump (auto-pick the newest one if not
# overridden). User can override with: STUDENT_RUN_DIR=/path sbatch ...
STUDENT_RUN_DIR="${STUDENT_RUN_DIR:-$(ls -dt ${UPSTREAM_VERL}/student_rollouts/dagger_qwen3_0.6b_S35x64_A_1r_16ep_on_B_* 2>/dev/null | head -1)}"
TEACHER_MODEL="${TEACHER_MODEL:-${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_8b_rl_step570}"
CKPT_STEP="${CKPT_STEP:-570}"

OUT_DIR="${UPSTREAM_VERL}/sft_data/dagger"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_PARQUET="${OUT_DIR}/qwen3_0.6b_A_1r_16ep_on_B_step570_T0.4_${SLURM_JOB_ID:-local}_${TS}.parquet"

test -d "${STUDENT_RUN_DIR}"  || { echo "ERROR: student_run_dir missing: ${STUDENT_RUN_DIR}"; exit 1; }
test -d "${TEACHER_MODEL}"    || { echo "ERROR: teacher_model missing:  ${TEACHER_MODEL}";   exit 1; }

# How many student JSONLs are there?
N_TRAJ=$(find "${STUDENT_RUN_DIR}/by_task_type" -name '*.jsonl' 2>/dev/null | wc -l)
echo "[phase2] student_run_dir=${STUDENT_RUN_DIR}"
echo "[phase2] student trajectories found: ${N_TRAJ}"
echo "[phase2] teacher_model=${TEACHER_MODEL}  ckpt_step=${CKPT_STEP}"
echo "[phase2] out_parquet=${OUT_PARQUET}"

python -c "import verl; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.alfworld.dagger_teacher_label import parse_args; print('dagger_teacher_label OK')"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

cleanup_runtime() {
    rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
    rm -rf /tmp/ray 2>/dev/null || true
}
cleanup_runtime

# ---- run labeler ----
python -m recipe.alfworld.dagger_teacher_label \
    --student_run_dir "${STUDENT_RUN_DIR}" \
    --teacher_model   "${TEACHER_MODEL}" \
    --ckpt_step       "${CKPT_STEP}" \
    --out_parquet     "${OUT_PARQUET}" \
    --keep all \
    --tp 4 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager --dtype bfloat16 \
    --concurrency 32 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --max_tokens_per_turn 4096 \
    --enable_thinking true

EXIT_CODE=$?
echo "=============================================="
echo "DAgger Phase 2 exited with code ${EXIT_CODE}"
echo "Output parquet : ${OUT_PARQUET}"
echo "Manifest       : ${OUT_PARQUET%.parquet}.manifest.json"
echo "=============================================="
exit ${EXIT_CODE}
