#!/bin/bash
# =============================================================================
# Sequential data-scaling experiment for Qwen3-0.6B SFT — 1-rollout-per-game
# variant. Same gid partition as `S19x128_fwon8_prop_v1` (the main v1 run),
# but each part is subsampled to one rollout per game (deterministic, seed=42).
# Pairs with the v1 main run for an ablation: "more games, fewer rollouts each"
# vs "all 8 rollouts per game".
#
# For each cumulative subset {A, A+B, A+B+C, A+B+C+D, A+B+C+D+E}:
#   128 games / 1 rollout per part. Cumulative traj counts: 128, 256, 384,
#   512, 640.
#
# Step counts at bs=64, 2 epochs (small — that's the point):
#   A         128 traj  →  4 steps
#   A+B       256       →  8 steps
#   A+B+C     384       → 12 steps
#   A+B+C+D   512       → 16 steps
#   A+B+C+D+E 640       → 20 steps
#
# Eval cost is independent of training-data size (8 rollouts × 140 games each),
# so wall time is dominated by eval: ~30-45 min × 5 ≈ 4h. 8h SLURM budget.
#
# Usage:
#   sbatch recipe/alfworld/run_data_scaling_qwen3_0.6b_1rollout.sh
# =============================================================================

#SBATCH --job-name=alfworld-data-scaling-qwen3-0.6b-S19x128-1rollout
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
echo "ALFWorld data-scaling Qwen3-0.6B on S19x128 (1 rollout/game)"
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
SCHEME_VARIANT="1rollout"  # data dir suffix: ${SCHEME}_${SCHEME_VARIANT}
PARTS=(A B C D E)
NUM_GPUS=${NUM_GPUS:-2}
EPOCHS=${EPOCHS:-2}
BS=${BS:-64}
ROLLOUTS=${ROLLOUTS:-8}
EXP_TAG=${EXP_TAG:-1r_v1}

PROJECT_NAME="verl_agent_alfworld_sft"
SCHEME_DIR="${UPSTREAM_VERL}/sft_data/splits/${SCHEME}_${SCHEME_VARIANT}"
RESULTS_DIR="${UPSTREAM_VERL}/results/data_scaling_qwen3_0.6b_${SCHEME}_${EXP_TAG}_${SLURM_JOB_ID:-local}"
export RESULTS_DIR
mkdir -p "${RESULTS_DIR}"
echo "results dir: ${RESULTS_DIR}"

# Fail-fast imports
python -c "import verl; assert verl.__file__.startswith('${UPSTREAM_VERL}/verl/'), f'wrong verl: {verl.__file__}'; print(f'verl OK: {verl.__file__}')"
python -c "from verl.trainer.sft_trainer import SFTTrainer; print('SFTTrainer OK')"
python -c "from recipe.alfworld.pretokenized_sft_dataset import PretokenizedSFTDataset; print('PretokenizedSFTDataset OK')"
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

    EXP_NAME="sft_qwen3_0.6b_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    CKPT_DIR="${UPSTREAM_VERL}/checkpoints/sft/qwen3_0.6b_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    MERGED="${UPSTREAM_VERL}/checkpoints/merged_hf/qwen3_0.6b_sft_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}"
    TS="$(date +%Y%m%d_%H%M%S)"
    DUMP_DIR="${UPSTREAM_VERL}/trajectories/qwen3_0.6b_sft_${SCHEME}_${SUBSET_TAG}_${EXP_TAG}_eval_${SLURM_JOB_ID:-local}_${TS}"
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
