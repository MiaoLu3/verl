#!/bin/bash
# =============================================================================
# ALFWorld val-only on Qwen3-4B using upstream agent_loop framework.
# =============================================================================
# Re-uses submit_alfworld_agent_loop.sh's setup (cleanup, env, fail-fast,
# dump dir, etc) but overrides model path to Qwen3-4B and adds Qwen3-specific
# chat template kwargs so the model generates its own <think> tags instead of
# relying on the builtin thinking mode.
#
# Pairs with T8.5's val-only flow (history_length=0, do_sample=True, T=0.4,
# val_only=True) -- see run_alfworld.sh + configs/alfworld_agent_loop.yaml.
#
# Usage:
#   sbatch recipe/alfworld/submit_alfworld_qwen3_4b_val.sh
# =============================================================================

#SBATCH --job-name=alfworld-qwen3-0.6b-rl-wt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=24:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld Qwen3-4B val-only -- job starting"
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
export WANDB_ENTITY=miaolu-stanford-university
export WANDB_PROJECT=verl_agent_alfworld

# Local JSONL trajectory dumper
export ALFWORLD_TRAJ_DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen3_0.6b_rl_wt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ALFWORLD_TRAJ_DUMP_DIR}"
echo "Trajectory dump dir: ${ALFWORLD_TRAJ_DUMP_DIR}"

echo "Working dir:     $(pwd)"
echo "Python:          $(which python)"
echo "HF_HOME:         ${HF_HOME}"
echo "ALFWORLD_DATA:   ${ALFWORLD_DATA}"

# Cleanup
echo "--- pre-run cleanup ---"
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true
echo "ray: clean"

# Fail-fast checks
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop; print('AlfWorldAgentLoop OK')"
python -c "from recipe.alfworld.alfworld_dataset import AlfWorldDataset; print('AlfWorldDataset OK')"
python -c "import vllm, torch, transformers, ray; print(f'vllm={vllm.__version__} torch={torch.__version__} transformers={transformers.__version__} ray={ray.__version__}')"

GAMEFILES_ROOT="${ALFWORLD_DATA}/json_2.1.1/train"
GAMEFILE_COUNT=$(ls "${GAMEFILES_ROOT}" 2>/dev/null | wc -l)
[ "${GAMEFILE_COUNT}" -gt 0 ] || { echo "ERROR: gamefiles empty"; exit 1; }
echo "alfworld gamefile count: ${GAMEFILE_COUNT}"

mkdir -p /tmp/alfworld_stub
[ -f /tmp/alfworld_stub_train.parquet ] || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_train.parquet')"
[ -f /tmp/alfworld_stub_val.parquet ]   || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_val.parquet')"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=============================================="

# Run with Qwen3-4B override. Leave enable_thinking at default True:
# counter-intuitively, enable_thinking=False prefills <think></think> into
# the assistant prompt, which double-closes against our template's <think>
# instruction. Default True produces a plain "<|im_start|>assistant\n"
# prefix and lets the model emit its own <think>...</think><action>...
bash "${ALFWORLD_RECIPE}/run_alfworld.sh" \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.val_only=False \
    trainer.total_training_steps=null \
    trainer.total_epochs=2 \
    trainer.test_freq=5 \
    trainer.save_freq=15 \
    trainer.resume_mode=auto \
    trainer.experiment_name=agent_loop_qwen3_0.6b_rl_with_think

EXIT_CODE=$?
echo "=============================================="
echo "ALFWorld Qwen3-4B val-only exited with code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
