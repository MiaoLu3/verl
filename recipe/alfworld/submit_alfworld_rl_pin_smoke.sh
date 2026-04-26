#!/bin/bash
# Smoke test for the new pinned-gamefile semantics in RL training mode.
#
# What this verifies: with rollout_n=8 and the same dataset row replicated 8
# times in a batch, all 8 rollouts pin to that row's gamefile (instead of
# letting the env's shuffled_cycle pick 8 unrelated games as the old code
# did). After the run finishes, the script reads every dumped JSONL and
# checks: every (gamefile_id) appears exactly rollout_n=8 times.
#
# Uses Qwen3-0.6B + 2 prompts + 8 rollouts = 16 episodes for one training
# step, on 2 GPUs. Total wall ~ 5-15 min.

#SBATCH --job-name=alfworld-rl-pin-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=0:45:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld RL pinned-gamefile SMOKE"
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
export WANDB_ENTITY=miaolu-stanford-university
export WANDB_PROJECT=verl_agent_alfworld
export WANDB_MODE=offline

TS="$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${UPSTREAM_VERL}/trajectories/rl_pin_smoke_${SLURM_JOB_ID}_${TS}"
mkdir -p "${DUMP_DIR}"
export ALFWORLD_TRAJ_DUMP_DIR="${DUMP_DIR}"
echo "Dump dir: ${DUMP_DIR}"

# pre-run cleanup
rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
ray stop --force 2>&1 | tail -5 || true
rm -rf /tmp/ray 2>/dev/null || true

# fail-fast imports
python -c "import verl; print('verl', verl.__file__)"
python -c "from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop; print('AlfWorldAgentLoop OK')"
python -c "from recipe.alfworld.alfworld_env_wrapper import AlfWorldSingleEnv; print('AlfWorldSingleEnv OK')"

# stub parquet files (verl expects them)
mkdir -p /tmp/alfworld_stub
[ -f /tmp/alfworld_stub_train.parquet ] || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_train.parquet')"
[ -f /tmp/alfworld_stub_val.parquet ]   || python -c "import pandas as pd; pd.DataFrame({'x':[0]}).to_parquet('/tmp/alfworld_stub_val.parquet')"

# Tiny RL training step: 2 prompts × 8 rollouts = 16 episodes per step.
# We do exactly 1 training step + skip val to keep it fast.
TRAIN_BATCH_SIZE=2
ROLLOUT_N=8

bash "${ALFWORLD_RECIPE}/run_alfworld.sh" \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    trainer.n_gpus_per_node=2 \
    trainer.val_only=False \
    trainer.val_before_train=False \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.experiment_name=rl_pin_smoke_${SLURM_JOB_ID}

EXIT_CODE=$?

echo "=============================================="
echo "Verifying pinning correctness..."
echo "=============================================="

python <<EOF
import json, os, sys, glob
from collections import defaultdict
dump_dir = "${DUMP_DIR}"

# RL training layout (now unified with teacher): step_<N>/by_task_type/<task>/<gid>__rollout_<idx>.jsonl
files = sorted(glob.glob(os.path.join(dump_dir, "step_*/by_task_type/*/*.jsonl")))
print(f"found {len(files)} jsonl files under {dump_dir}")
if not files:
    sys.exit("FAIL: no trajectory files dumped")

per_gid = defaultdict(list)
for fp in files:
    try:
        d = json.load(open(fp))
    except Exception as e:
        print(f"  skip {fp}: {e}")
        continue
    gid = d.get("gamefile_id", "unknown")
    per_gid[gid].append(d)

print(f"unique gamefile_ids: {len(per_gid)}")
print("per-gid rollout counts:")
for gid, ts in sorted(per_gid.items()):
    print(f"  {gid}: {len(ts)} rollouts")

expected_rollout_n = ${ROLLOUT_N}
expected_unique = ${TRAIN_BATCH_SIZE}

ok = True
if len(per_gid) != expected_unique:
    print(f"FAIL: expected {expected_unique} unique games, got {len(per_gid)}")
    ok = False
for gid, ts in per_gid.items():
    if len(ts) != expected_rollout_n:
        print(f"FAIL: gamefile {gid} has {len(ts)} rollouts, expected {expected_rollout_n}")
        ok = False
if ok:
    print(f"PASS: {len(per_gid)} unique gamefile_ids each with exactly {expected_rollout_n} rollouts ✓")
    print("Pinning verified — same-context GRPO group is real.")
sys.exit(0 if ok else 1)
EOF

VERIFY_RC=$?

echo "=============================================="
echo "RL pin smoke exited with run_code=${EXIT_CODE} verify_code=${VERIFY_RC}"
echo "Dump dir: ${DUMP_DIR}"
echo "=============================================="
exit ${VERIFY_RC}
