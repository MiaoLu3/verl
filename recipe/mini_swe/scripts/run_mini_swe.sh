#!/bin/bash
# Inner launcher: invoke main_ppo with mini-swe-agent x SWE-bench
# agent-loop overrides. Intended to be called from the SLURM wrapper
# ``submit_mini_swe_smoke.sh`` after env activation, SIF-cache verification,
# and zombie-instance cleanup.
set -x
ulimit -n 65535

PROJECT_DIR="/scratch/m000069-pm05/miaolu/verl"
CONFIG_PATH="${PROJECT_DIR}/recipe/mini_swe/configs"
MINI_SWE_RECIPE_DIR="${PROJECT_DIR}/recipe/mini_swe"

# HF / SWE-bench caches
export HF_HOME=${HF_HOME:-/scratch/m000069/miaolu/.cache/huggingface}
export SIF_CACHE_DIR=${SIF_CACHE_DIR:-/scratch/m000069-pm05/miaolu/swebench_sifs}
export VLLM_USE_V1=1

# Smoke-run defaults; overridable on the CLI via $@ or via env.
MSWE_MODEL_PATH=${MSWE_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-2}
MSWE_MAX_TURNS=${MSWE_MAX_TURNS:-75}
export MSWE_MAX_TURNS

cd "${PROJECT_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='mini_swe_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    data.custom_cls.path="${MINI_SWE_RECIPE_DIR}/dataset.py" \
    data.custom_cls.name=SweBenchDataset \
    actor_rollout_ref.model.path="${MSWE_MODEL_PATH}" \
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
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MSWE_MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MSWE_MAX_TURNS} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${CONFIG_PATH}/agent_loops.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=mini_swe \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS:-2} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_agent_mini_swe' \
    trainer.experiment_name='mini_swe_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=2 \
    trainer.total_epochs=1 \
    $@
