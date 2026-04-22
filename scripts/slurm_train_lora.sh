#!/usr/bin/env bash
# Submit a LoRA fine-tuning job for Wan2.2-T2V-A14B on the Physics/WISA dataset.
# Usage: bash scripts/slurm_train_lora.sh [run_name]
set -euo pipefail

RUN_NAME="${1:-lora_wan22_$(date +%Y%m%d_%H%M%S)}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
LORA_DIR="${PROJECT_DIR}/third_party/lora_finetuning"
SCRATCH_BASE="/n/netscratch/ydu_lab/Lab/bcupps"
OUTPUT_DIR="${SCRATCH_BASE}/results/lora_runs/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

cd "${LORA_DIR}"

# The training command. Submits via the built-in cluster launcher (cluster=fas_bcupps).
# To run locally instead: remove cluster=fas_bcupps and run directly.
python -m main \
  +name="${RUN_NAME}" \
  experiment=exp_video \
  algorithm=wan_t2v_A14B_lora \
  dataset=wisa_physics \
  cluster=fas_bcupps \
  "experiment.training.checkpointing.every_n_train_steps=500" \
  "experiment.training.max_steps=3000" \
  "experiment.training.lr=5e-5" \
  "experiment.training.optim.accumulate_grad_batches=4" \
  "algorithm.model.tuned_ckpt_path=null" \
  2>&1 | tee "${OUTPUT_DIR}/submit.log"

echo "Run name: ${RUN_NAME}"
echo "Logs and checkpoints will appear in: ${OUTPUT_DIR}"
