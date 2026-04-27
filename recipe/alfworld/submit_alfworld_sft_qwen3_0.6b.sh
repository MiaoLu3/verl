#!/bin/bash
# =============================================================================
# Pilot SFT of Qwen3-0.6B on pre-tokenized teacher trajectories produced by
# the qwen3_8b RL step570 (T=0.4) won-only rollout dump.
#
# The data was tokenized once during rollout (so token tapes match exactly
# what the rollout model saw), and `recipe/alfworld/pretokenized_sft_dataset.py`
# bypasses verl's chat-template re-tokenization. We use verl's FSDP SPMD SFT
# trainer (`python -m verl.trainer.sft_trainer` via torchrun) -- 0.6B fits
# trivially on H100, so no offload, no LoRA.
#
# Usage:
#   sbatch recipe/alfworld/submit_alfworld_sft_qwen3_0.6b.sh
# =============================================================================

#SBATCH --job-name=alfworld-sft-qwen3-0.6b
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
echo "ALFWorld SFT Qwen3-0.6B pilot -- job starting"
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
TRAIN_FILE="${UPSTREAM_VERL}/sft_data/qwen3_8b_rl_step570_T0.4_won.parquet"
CKPT_DIR="${UPSTREAM_VERL}/checkpoints/sft/qwen3_0.6b_alfworld_v1"
EXP_NAME="sft_qwen3_0.6b_alfworld_pilot_v1"
PROJECT_NAME="verl_agent_alfworld_sft"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "Train parquet:   ${TRAIN_FILE}"
echo "Checkpoint dir:  ${CKPT_DIR}"

mkdir -p "${CKPT_DIR}"

# Cleanup safety net (no rollout/Ray here, but keep mirror with RL submit scripts)
echo "--- pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"

# Fail-fast imports
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from verl.trainer.sft_trainer import SFTTrainer; print('SFTTrainer OK')"
python -c "from recipe.alfworld.pretokenized_sft_dataset import PretokenizedSFTDataset; print('PretokenizedSFTDataset OK')"
python -c "import torch, transformers; print(f'torch={torch.__version__} transformers={transformers.__version__}')"
test -f "${TRAIN_FILE}" || { echo "ERROR: train parquet missing: ${TRAIN_FILE}"; exit 1; }

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

# -----------------------------------------------------------------------------
# Hyperparameters (see rationale in commit notes / agent report):
#   - max_length=16384 matches RL rollout cap, no truncation expected.
#   - pad_mode=no_padding => SFTTensorCollator emits NestedTensors so the
#     5-10k-token rows are not padded to 16k (massive throughput win).
#   - use_dynamic_bsz=True with max_token_len_per_gpu=24576 lets the engine
#     pack ~2-3 mid-length samples per microbatch on H100 SXM (80GB).
#   - LR 8e-6, cosine, 5% warmup, 2 epochs, AdamW [0.9, 0.95], wd 0.01,
#     clip_grad 1.0 -- standard small-LM SFT on agentic data.
#   - global batch 64 with 2 GPUs => 32/GPU; with dynamic packing this is
#     bounded by token budget rather than micro_batch_size_per_gpu.
#   - FSDP2, no param/optim offload (0.6B fits with room to spare).
# -----------------------------------------------------------------------------

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=null \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    data.truncation=error \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    data.num_workers=4 \
    data.custom_cls.path="${ALFWORLD_RECIPE}/pretokenized_sft_dataset.py" \
    data.custom_cls.name=PretokenizedSFTDataset \
    model=hf_model \
    model.path=Qwen/Qwen3-0.6B \
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
echo "ALFWorld SFT Qwen3-0.6B exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
