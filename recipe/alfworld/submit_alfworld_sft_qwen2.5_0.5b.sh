#!/bin/bash
# =============================================================================
# Pilot SFT of Qwen2.5-0.5B-Instruct on teacher trajectories from Qwen3-8B RL
# step570. Because Qwen2.5 has a different tokenizer/template than Qwen3, we
# CANNOT use the pre-tokenized fast path -- token IDs in our dump are tied to
# Qwen3's vocab (e.g. <think> = special id 151667 in Qwen3 but a 7-token plain
# string in Qwen2.5). Instead we feed verl's default MultiTurnSFTDataset the
# `messages` column and let it re-tokenize with the Qwen2.5-0.5B tokenizer.
#
# Note that Qwen2.5's chat template does NOT strip <think>...</think> from
# non-last assistant turns (no native think handling), so train context
# matches inference context exactly.
#
# Usage:
#   sbatch recipe/alfworld/submit_alfworld_sft_qwen2.5_0.5b.sh
# =============================================================================

#SBATCH --job-name=alfworld-sft-qwen2.5-0.5b
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=04:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld SFT Qwen2.5-0.5B-Instruct pilot"
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
export WANDB_ENTITY=miaolu-stanford-university
export WANDB_PROJECT=verl_agent_alfworld_sft

NUM_GPUS=${NUM_GPUS:-2}
# Messages-only parquet (no tokens columns) so verl's default
# MultiTurnSFTDataset re-tokenizes from `messages` with the Qwen2.5 tokenizer.
TRAIN_FILE="${UPSTREAM_VERL}/sft_data/qwen3_8b_rl_step570_T0.4_won_messages.parquet"
CKPT_DIR="${UPSTREAM_VERL}/checkpoints/sft/qwen2.5_0.5b_alfworld_v1"
EXP_NAME="sft_qwen2.5_0.5b_alfworld_pilot_v1"
PROJECT_NAME="verl_agent_alfworld_sft"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "Train parquet:   ${TRAIN_FILE}"
echo "Checkpoint dir:  ${CKPT_DIR}"

mkdir -p "${CKPT_DIR}"

# Cleanup safety net
echo "--- pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"

# Fail-fast imports
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from verl.trainer.sft_trainer import SFTTrainer; print('SFTTrainer OK')"
python -c "from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset; print('MultiTurnSFTDataset OK')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); print('Qwen2.5 tokenizer OK')"
test -f "${TRAIN_FILE}" || { echo "ERROR: train parquet missing: ${TRAIN_FILE}"; exit 1; }

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

# -----------------------------------------------------------------------------
# Hyperparameters (mirrored from Qwen3-0.6B SFT pilot):
#   - max_length=16384 matches teacher rollout cap. Qwen2.5 re-tokenization
#     of the same messages may give a slightly different length than Qwen3,
#     but should still fit (no special think tokens means slightly LONGER
#     sequences -- 7 tokens for "<think>" vs 1 for Qwen3). truncation=right
#     is a safety net (was 'error' for the Qwen3 path).
#   - apply_chat_template_kwargs left empty -- Qwen2.5 doesn't accept
#     enable_thinking, and the messages already have <think> as plain text.
#   - ignore_input_ids_mismatch=True is a defensive flag; not strictly
#     required for Qwen2.5 (no think-stripping) but keeps things robust if
#     the chat template ever changes.
# -----------------------------------------------------------------------------

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=null \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    data.truncation=right \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    data.num_workers=4 \
    data.messages_key=messages \
    data.tools_key=tools \
    data.enable_thinking_key=enable_thinking \
    data.enable_thinking_default=false \
    data.ignore_input_ids_mismatch=True \
    model=hf_model \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    model.trust_remote_code=True \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1 \
    engine.param_offload=False \
    engine.optimizer_offload=False \
    engine.ulysses_sequence_parallel_size=1 \
    optim=fsdp \
    optim.lr=8e-6 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.01 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=2 \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=-1 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS}

EXIT_CODE=$?
echo "=============================================="
echo "ALFWorld SFT Qwen2.5-0.5B-Instruct exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
