#!/usr/bin/env bash
# Fast LoRA fine-tuning job: 2 nodes (8x H100), no gradient checkpointing, 300 steps.
# Targets ~2 hours wall time. Use for quick iteration / sanity checks.
# Usage: bash scripts/slurm_train_lora_fast.sh [run_name]
set -euo pipefail

RUN_NAME="${1:-lora_wan22_fast_$(date +%Y%m%d_%H%M%S)}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
LORA_DIR="${PROJECT_DIR}/third_party/lora_finetuning"
SCRATCH_BASE="/n/netscratch/ydu_lab/Lab/bcupps"
OUTPUT_DIR="${SCRATCH_BASE}/results/lora_runs/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

cd "${LORA_DIR}"

python -m main \
  +name="${RUN_NAME}" \
  experiment=exp_video \
  algorithm=wan_t2v_A14B_lora \
  dataset=wisa_physics \
  cluster=fas_bcupps \
  "experiment.num_nodes=2" \
  "cluster.params.num_gpus=4" \
  "cluster.params.time=4:00:00" \
  "algorithm.gradient_checkpointing_rate=0" \
  "experiment.training.max_steps=300" \
  "experiment.training.lr=1e-4" \
  "experiment.training.optim.accumulate_grad_batches=4" \
  "experiment.training.checkpointing.every_n_train_steps=50" \
  "algorithm.model.tuned_ckpt_path=null" \
  2>&1 | tee "${OUTPUT_DIR}/submit.log"

echo "Run name: ${RUN_NAME}"
echo "Logs and checkpoints will appear in: ${OUTPUT_DIR}"
