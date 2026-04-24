#!/bin/bash
# Inner launcher: invoke main_ppo with ALFWorld agent-loop overrides.
# Intended to be called from the SLURM wrapper submit_alfworld_agent_loop.sh
# after env activation and cleanup.
set -x
ulimit -n 65535

PROJECT_DIR="/scratch/m000069-pm05/miaolu/verl"
CONFIG_PATH="${PROJECT_DIR}/recipe/alfworld/configs"
ALFWORLD_RECIPE_DIR="${PROJECT_DIR}/recipe/alfworld"

# HF / ALFWorld caches
export HF_HOME=${HF_HOME:-/scratch/m000069/miaolu/.cache/huggingface}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export VLLM_USE_V1=1

# Smoke defaults; override on the CLI via $@ if needed.
# Bumped (bsz 8 -> 16, rollout_n 4 -> 8) for richer trajectory coverage:
# 16 * 8 = 128 episodes/step * 2 steps = 256 trajectories per smoke.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-8}

cd "${PROJECT_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='alfworld_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    data.custom_cls.path="${ALFWORLD_RECIPE_DIR}/alfworld_dataset.py" \
    data.custom_cls.name=AlfWorldDataset \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${CONFIG_PATH}/alfworld_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='agent_loop_qwen2.5_1.5b_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    $@
