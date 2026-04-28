#!/bin/bash
# =============================================================================
# Single-shot SFT of Qwen3-0.6B on a 2-rollout-per-game subsample of
# S19x128 part A (S19x128_fwon8_prop_v1/A_2rollout.parquet): 128 games ×
# 2 rollouts = 256 trajectories. Runs at total_epochs=8 to match the
# user's external 8-epoch baseline on the 1rollout split.
#
# Hyperparameters (mirror the user's external 8-epoch 1rollout run):
#   bs=16, lr=8e-6 cosine, min_lr_ratio=0.1, warmup_steps_ratio=0.05,
#   total_epochs=8.
# Step counts: 256 / 16 = 16 steps/epoch × 8 epochs = 128 total steps,
# warmup ≈ round(0.05 * 128) = 6 steps. Same total step count as
# the bs=64 main-v1 A+B+C+D run, but on much less data (256 vs 4124 traj).
#
# Eval bumped to --pool_size 32 --concurrency 32 (was 16/16) per user
# request — current 16/16 was too slow on weak-student episodes that
# usually hit max_steps (~100s/episode).
#
# Usage:
#   sbatch recipe/alfworld/submit_alfworld_sft_qwen3_0.6b_S19x128_A_2rollout_8ep.sh
# =============================================================================

#SBATCH --job-name=alfworld-sft-qwen3-0.6b-S19x128-A_2rollout-8ep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=03:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld SFT Qwen3-0.6B S19x128 A_2rollout 8ep"
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
export RAY_agent_register_timeout_ms=300000
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=${NUM_GPUS:-2}
TRAIN_FILE="${UPSTREAM_VERL}/sft_data/splits/S19x128_fwon8_prop_v1/A_2rollout.parquet"
EXP_NAME="sft_qwen3_0.6b_alfworld_S19x128_A_2rollout_8ep_v1"
CKPT_DIR="${UPSTREAM_VERL}/checkpoints/sft/qwen3_0.6b_S19x128_A_2rollout_8ep_v1"
MERGED="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_0.6b_sft_S19x128_A_2rollout_8ep_v1"
PROJECT_NAME="verl_agent_alfworld_sft"
RESULTS_DIR="${UPSTREAM_VERL}/results/sft_qwen3_0.6b_S19x128_A_2rollout_8ep_v1_${SLURM_JOB_ID:-local}"
mkdir -p "${RESULTS_DIR}"

cleanup_runtime() {
    rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
    ray stop --force 2>&1 | tail -5 || true
    rm -rf /tmp/ray 2>/dev/null || true
}

# Fail-fast imports + data check
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from verl.trainer.sft_trainer import SFTTrainer; print('SFTTrainer OK')"
python -c "from recipe.alfworld.pretokenized_sft_dataset import PretokenizedSFTDataset; print('PretokenizedSFTDataset OK')"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"
python -c "from recipe.alfworld.compute_pass_at_k import pass_at_k; print('pass@k OK')"
test -f "${TRAIN_FILE}" || { echo "ERROR: train parquet missing: ${TRAIN_FILE}"; exit 1; }

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"
cleanup_runtime

# ---- 1. SFT ----
echo "---> [SFT] A_2rollout 256 traj, bs=16, ep=8 → 128 steps"
torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=null \
    data.train_batch_size=16 \
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
    trainer.total_epochs=8 \
    trainer.save_freq=after_each_epoch \
    trainer.test_freq=-1 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS}

STEP=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")
echo "---> SFT done. final step=${STEP}"
cleanup_runtime

# ---- 2. Merge ----
rm -rf "${MERGED}"
echo "---> [merge] step_${STEP} -> ${MERGED}"
python -m verl.model_merger merge --backend fsdp \
    --local_dir "${CKPT_DIR}/global_step_${STEP}" \
    --target_dir "${MERGED}" \
    --use_cpu_initialization

# ---- 3. Eval (8 rollouts/game, bumped concurrency) ----
TS="$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen3_0.6b_sft_S19x128_A_2rollout_8ep_v1_eval_${SLURM_JOB_ID:-local}_${TS}"
mkdir -p "${DUMP_DIR}"
echo "---> [eval] rollouts=8 pool=32 concurrency=32 dump=${DUMP_DIR}"
cleanup_runtime
python -m recipe.alfworld.eval_standalone \
    --model_path "${MERGED}" \
    --tp 1 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager --dtype bfloat16 \
    --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
    --split valid_seen \
    --pool_size 32 --concurrency 32 \
    --max_samples -1 \
    --seed_base 1042 \
    --history_length 0 --max_steps 50 \
    --prompt_length 4096 --response_length 12288 \
    --max_assistant_turns 50 --max_user_turns 50 \
    --max_tokens_per_turn 4096 \
    --temperature 0.4 --top_p 1.0 --top_k -1 \
    --dump_dir "${DUMP_DIR}" \
    --rollouts_per_game 8 \
    --num_cpus 16
cleanup_runtime

# ---- 4. pass@k ----
echo "---> [pass@k] computing"
python -m recipe.alfworld.compute_pass_at_k \
    --dump_dir "${DUMP_DIR}" \
    --rollouts_per_game 8 \
    --out_json "${RESULTS_DIR}/A_2rollout.json"

# annotate
python -c "
import json
p = '${RESULTS_DIR}/A_2rollout.json'
with open(p) as f: d = json.load(f)
d['subset_tag'] = 'A_2rollout'
d['subset_desc'] = 'A_2rollout (S19x128/A 128 games × 2 rollouts = 256 traj)'
d['scheme'] = 'S19x128_fwon8_prop_v1'
d['epochs'] = 8
d['batch_size'] = 16
d['final_step'] = ${STEP}
d['ckpt_dir'] = '${CKPT_DIR}'
d['merged_hf'] = '${MERGED}'
d['eval_dump_dir'] = '${DUMP_DIR}'
with open(p, 'w') as f: json.dump(d, f, indent=2)
print(f'[meta] annotated {p}')
"

EXIT_CODE=$?
echo "=============================================="
echo "ALFWorld SFT+eval Qwen3-0.6B A_2rollout 8ep exited with code ${EXIT_CODE}"
echo "Results: ${RESULTS_DIR}/A_2rollout.json"
echo "=============================================="
exit ${EXIT_CODE}
