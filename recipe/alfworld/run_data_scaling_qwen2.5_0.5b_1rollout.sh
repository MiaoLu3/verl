#!/bin/bash
# =============================================================================
# Sequential data-scaling experiment for Qwen2.5-0.5B-Instruct SFT —
# 1-rollout-per-game variant, messages-only split.
# Same gid partition as `S19x128_fwon8_prop_v1` (the main v1 run), but each
# part is subsampled to one rollout per game (deterministic, seed=42), and
# we read from the messages-only mirror (Qwen2.5 has a different tokenizer
# than Qwen3, so we re-tokenize from `messages` rather than re-using the
# pre-tokenized Qwen3 path).
#
# Pairs with `run_data_scaling_qwen2.5_0.5b.sh` (the main v1 run with all
# 8 rollouts/game) for an ablation: "more games, fewer rollouts each" vs
# "all 8 rollouts per game".
#
# For each cumulative subset {A, A+B, A+B+C, A+B+C+D, A+B+C+D+E}:
#   128 games / 1 rollout per part. Cumulative traj counts: 128, 256, 384,
#   512, 640.
#
# Step counts at bs=64, 2 epochs:
#   A         128 traj  →  4 steps
#   A+B+C+D+E 640       → 20 steps
#
# Eval cost is independent of training-data size (8 rollouts × 140 games each),
# so wall time is dominated by eval: ~30-50 min × 5 ≈ 4-5h. 8h SLURM budget.
#
# Same Qwen2.5-specific flags vs Qwen3 variant:
#   - default MultiTurnSFTDataset (no custom_cls)
#   - data.truncation=right, ignore_input_ids_mismatch=True
#   - data.enable_thinking_default=false
#
# Usage:
#   sbatch recipe/alfworld/run_data_scaling_qwen2.5_0.5b_1rollout.sh
# =============================================================================

#SBATCH --job-name=alfworld-data-scaling-qwen2.5-0.5b-S19x128-1rollout
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=32
#SBATCH --mem=720G
#SBATCH -A marlowe-m000069-pm05
#SBATCH --qos=medium
#SBATCH --time=08:00:00

set -euo pipefail

echo "=============================================="
echo "ALFWorld data-scaling Qwen2.5-0.5B-Instruct on S19x128 (1 rollout/game)"
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

# ---- experiment knobs ----
SCHEME="S19x128_fwon8_prop_v1"
SCHEME_VARIANT="1rollout"  # data dir suffix: ${SCHEME}_${SCHEME_VARIANT}_messages
PARTS=(A B C D E)
NUM_GPUS=${NUM_GPUS:-2}
EPOCHS=${EPOCHS:-2}
BS=${BS:-64}
ROLLOUTS=${ROLLOUTS:-8}
EXP_TAG=${EXP_TAG:-1r_v1}

PROJECT_NAME="verl_agent_alfworld_sft"
SCHEME_DIR="${UPSTREAM_VERL}/sft_data/splits/${SCHEME}_${SCHEME_VARIANT}_messages"
RESULTS_DIR="${UPSTREAM_VERL}/results/data_scaling_qwen2.5_0.5b_${SCHEME}_${EXP_TAG}_${SLURM_JOB_ID:-local}"
export RESULTS_DIR
mkdir -p "${RESULTS_DIR}"
echo "results dir: ${RESULTS_DIR}"

# Fail-fast imports
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from verl.trainer.sft_trainer import SFTTrainer; print('SFTTrainer OK')"
python -c "from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset; print('MultiTurnSFTDataset OK')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); print('Qwen2.5 tokenizer OK')"
python -c "from recipe.alfworld.eval_standalone import parse_args; print('eval_standalone OK')"
python -c "from recipe.alfworld.compute_pass_at_k import pass_at_k; print('pass@k OK')"
for p in "${PARTS[@]}"; do
    test -f "${SCHEME_DIR}/${p}.parquet" || { echo "ERROR: missing ${SCHEME_DIR}/${p}.parquet"; exit 1; }
done

cleanup_runtime() {
    rm -f /tmp/rl-colocate-zmq-GPU-*.sock 2>/dev/null || true
    ray stop --force 2>&1 | tail -5 || true
    rm -rf /tmp/ray 2>/dev/null || true
}

NUM_SUBSETS=${#PARTS[@]}
for ((i=1; i<=NUM_SUBSETS; i++)); do
    SUBSET=("${PARTS[@]:0:i}")
    SUBSET_TAG=$(IFS=_; echo "${SUBSET[*]}")
    SUBSET_DESC=$(IFS=+; echo "${SUBSET[*]}")
    echo ""
    echo "=============================================="
    echo "[$i/$NUM_SUBSETS] Subset ${SUBSET_DESC}"
    echo "=============================================="

    EXP_NAME="sft_qwen2.5_0.5b_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    CKPT_DIR="${UPSTREAM_VERL}/checkpoints/sft/qwen2.5_0.5b_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    MERGED="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen2.5_0.5b_sft_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    TS="$(date +%Y%m%d_%H%M%S)"
    DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen2.5_0.5b_sft_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}_eval_${SLURM_JOB_ID:-local}_${TS}"
    PER_RUN_RESULT="${RESULTS_DIR}/${SUBSET_TAG}.json"

    # Build hydra-list of train_files. Hydra accepts: data.train_files=[a,b,c]
    TF_LIST=""
    for p in "${SUBSET[@]}"; do
        f="${SCHEME_DIR}/${p}.parquet"
        TF_LIST="${TF_LIST}${TF_LIST:+,}${f}"
    done
    echo "train_files: [${TF_LIST}]"

    rm -rf "${CKPT_DIR}"
    mkdir -p "${CKPT_DIR}"

    cleanup_runtime
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head

    # ---- 1. SFT ----
    echo "---> [SFT] subset=${SUBSET_DESC} bs=${BS} epochs=${EPOCHS}"
    torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
        -m verl.trainer.sft_trainer \
        data.train_files="[${TF_LIST}]" \
        data.val_files=null \
        data.train_batch_size=${BS} \
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
        trainer.total_epochs=${EPOCHS} \
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

    # ---- 3. Eval (8 rollouts/game) ----
    mkdir -p "${DUMP_DIR}"
    echo "---> [eval] rollouts=${ROLLOUTS} dump=${DUMP_DIR}"
    cleanup_runtime
    python -m recipe.alfworld.eval_standalone \
        --model_path "${MERGED}" \
        --tp 1 \
        --max_model_len 16384 \
        --gpu_memory_utilization 0.85 \
        --enforce_eager --dtype bfloat16 \
        --alf_config_path "${ALFWORLD_RECIPE}/config_tw.yaml" \
        --split valid_seen \
        --pool_size 16 --concurrency 16 \
        --max_samples -1 \
        --seed_base 1042 \
        --history_length 0 --max_steps 50 \
        --prompt_length 4096 --response_length 12288 \
        --max_assistant_turns 50 --max_user_turns 50 \
        --max_tokens_per_turn 4096 \
        --temperature 0.4 --top_p 1.0 --top_k -1 \
        --dump_dir "${DUMP_DIR}" \
        --rollouts_per_game ${ROLLOUTS} \
        --num_cpus 8

    cleanup_runtime

    # ---- 4. pass@k ----
    echo "---> [pass@k] computing for ${SUBSET_DESC}"
    python -m recipe.alfworld.compute_pass_at_k \
        --dump_dir "${DUMP_DIR}" \
        --rollouts_per_game ${ROLLOUTS} \
        --out_json "${PER_RUN_RESULT}"

    # Annotate result JSON with subset metadata
    python -c "
import json, os
p = '${PER_RUN_RESULT}'
with open(p) as f: d = json.load(f)
d['subset_tag'] = '${SUBSET_TAG}'
d['subset_desc'] = '${SUBSET_DESC}'
d['parts'] = '${SUBSET[*]}'.split()
d['scheme'] = '${SCHEME}'
d['epochs'] = ${EPOCHS}
d['batch_size'] = ${BS}
d['final_step'] = ${STEP}
d['ckpt_dir'] = '${CKPT_DIR}'
d['merged_hf'] = '${MERGED}'
d['eval_dump_dir'] = '${DUMP_DIR}'
with open(p, 'w') as f: json.dump(d, f, indent=2)
print(f'[meta] annotated {p}')
"

    echo "[$i/$NUM_SUBSETS] DONE: ${SUBSET_DESC}  →  ${PER_RUN_RESULT}"
done

# ---- final cross-subset summary ----
echo ""
echo "=============================================="
echo "Cross-subset summary"
echo "=============================================="
python - <<'PY'
import glob, json, os
results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir:
    import sys; sys.exit("RESULTS_DIR not set")
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
files = [f for f in files if os.path.basename(f) != "summary.json"]
rows = []
for f in files:
    with open(f) as fh: d = json.load(fh)
    rows.append({
        "subset": d.get("subset_desc"),
        "n_parts": len(d.get("parts", [])),
        "epochs": d.get("epochs"),
        "bs": d.get("batch_size"),
        "step": d.get("final_step"),
        "n_games": d.get("n_games"),
        **d.get("overall", {}),
    })
import json
print(json.dumps(rows, indent=2))
out = os.path.join(results_dir, "summary.json")
with open(out, "w") as fh: json.dump(rows, fh, indent=2)
print(f"\n[summary] written to {out}")
print()
header = "subset           parts  step  pass@1  pass@2  pass@4  pass@8"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{str(r['subset'] or ''):<14}  "
          f"{r['n_parts']:>5}  "
          f"{r['step']:>4}  "
          f"{r.get('pass@1', 0):.3f}  "
          f"{r.get('pass@2', 0):.3f}  "
          f"{r.get('pass@4', 0):.3f}  "
          f"{r.get('pass@8', 0):.3f}")
PY

echo ""
echo "=============================================="
echo "All done. Results in: ${RESULTS_DIR}"
echo "=============================================="
