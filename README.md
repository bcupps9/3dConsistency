# 3dConsistency

Physics-consistency evaluation pipeline for:
- `Wan2.2`
- `Wan2.1`
- `large-video-planner (LVP)`

The current production flow is **layout-driven inference**:
1. build dataset manifests,
2. prepare standardized run layout,
3. submit one Slurm job that runs all models/tasks/datasets,
4. monitor progress and collect outputs from `RUN_ROOT`.

## Directory Conventions
- Project root: `~/projects/3dConsistency`
- Scratch results root: `/n/netscratch/ydu_lab/Lab/bcupps/results`
- Scratch raw data root: `/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw`

## One-Time Setup (Remote)
```bash
cd ~/projects/3dConsistency
git pull origin main
git submodule update --init --recursive
```

Create model environments:
```bash
bash scripts/bootstrap_remote_env.sh      # Wan2.2 env (.mamba/wan2)
bash scripts/bootstrap_wan21_env.sh       # Wan2.1 env (.mamba/wan21)
bash scripts/bootstrap_lvp_env.sh         # LVP env (default: ei_world_model)
```

## Data Download

### WISA-80K (recommended via Slurm)
```bash
export HF_TOKEN=<your_hf_token>
sbatch \
  -p serial_requeue \
  --cpus-per-task=8 \
  --mem=32G \
  --time=48:00:00 \
  --output=/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/download-%j.out \
  --export=ALL,REPO_ID=qihoo360/WISA-80K,REPO_TYPE=dataset,DEST=/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K,ALLOW_PATTERNS='data/**',HF_ENV_NAME=hf_dl \
  scripts/slurm_download_hf_dataset.sh "$HF_TOKEN"
```

Expected files:
- `/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K/data/wisa-80k.json`
- `/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K/data/videos/*.mp4`

### Physics-IQ
```bash
cd /n/netscratch/ydu_lab/Lab/bcupps/datasets/raw
python -m pip install -U gsutil
printf "30\n" | python ~/projects/3dConsistency/third_party/physics-IQ-Benchmark/code/download_physics_iq_data.py
```

Ensure descriptions CSV exists at:
- `/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/physics-IQ-benchmark/descriptions.csv`

## Build Manifests

```bash
RUN_ID=20260218_batch1
RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/$RUN_ID
MANIFEST_DIR=$RUN_ROOT/manifests
mkdir -p "$MANIFEST_DIR"
```

### Physics-IQ manifest
```bash
python scripts/build_physics_iq_manifest.py \
  --descriptions-csv /n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/physics-IQ-benchmark/descriptions.csv \
  --video-search-roots /n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/physics-IQ-benchmark \
  --output-manifest "$MANIFEST_DIR/physics_iq.jsonl" \
  --take-filter take-1 \
  --limit 100 \
  --debug-misses 10
```

### WISA manifest
```bash
python scripts/build_wisa_manifest.py \
  --json-path /n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K/data/wisa-80k.json \
  --video-root /n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K/data/videos \
  --output-manifest "$MANIFEST_DIR/wisa80k.jsonl" \
  --limit 100
```

## Prepare Inference Layout

If `ffmpeg` is not on PATH, install in your active env and pass `--ffmpeg-bin`.

```bash
python scripts/prepare_inference_layout.py \
  --manifest "$MANIFEST_DIR/physics_iq.jsonl" \
  --run-root "$RUN_ROOT" \
  --dataset-name physics_iq \
  --materialize-mode symlink

python scripts/prepare_inference_layout.py \
  --manifest "$MANIFEST_DIR/wisa80k.jsonl" \
  --run-root "$RUN_ROOT" \
  --dataset-name wisa80k \
  --materialize-mode symlink
```

Output structure:
- `RUN_ROOT/datasets/<dataset>/samples/<sample_id>/...`
- `RUN_ROOT/wan22/<dataset>/{t2v,i2v}/...`
- `RUN_ROOT/wan21/<dataset>/{t2v,i2v}/...`
- `RUN_ROOT/lvp/<dataset>/{t2v,i2v}/...`

## Submit Inference Jobs

### Smoke (all models, both datasets, both tasks, 1 sample)
```bash
RUN_ID=20260218_smoke
RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/$RUN_ID

MAX_SAMPLES=1 \
GPU_PARTITION=gpu_h200 \
GRES='gpu:nvidia_h200:1' \
MEMORY=32G \
CPUS_PER_TASK=2 \
WALLTIME=00:30:00 \
DATASET_NAMES=physics_iq,wisa80k \
TARGETS=wan22,wan21,lvp \
TASKS=t2v,i2v \
RUN_ROOT=$RUN_ROOT \
LOCAL_SUBMIT=1 \
bash scripts/submit_slurm_layout.sh $RUN_ID
```

### Full batch
```bash
RUN_ID=20260218_full
RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/$RUN_ID

GPU_PARTITION=gpu_h200 \
GRES='gpu:nvidia_h200:1' \
MEMORY=64G \
CPUS_PER_TASK=4 \
WALLTIME=12:00:00 \
DATASET_NAMES=physics_iq,wisa80k \
TARGETS=wan22,wan21,lvp \
TASKS=t2v,i2v \
RUN_ROOT=$RUN_ROOT \
LOCAL_SUBMIT=1 \
bash scripts/submit_slurm_layout.sh $RUN_ID
```

Notes:
- Partial progress is preserved. If a job fails late, already-written outputs remain.
- Set `CONTINUE_ON_ERROR=1` to skip bad samples and continue.
- LVP uses `wan_i2v` inference path for both task slots; `t2v` runs with reduced history guidance.

## Monitor / Cancel
```bash
squeue -u $USER
squeue --start -j <JOBID>
scontrol show job <JOBID> | egrep "JobState=|Reason=|Partition=|ReqTRES=|NumCPUs=|MinMemory|TimeLimit"
```

Logs:
```bash
tail -f /n/netscratch/ydu_lab/Lab/bcupps/results/<run_id>/slurm-<jobid>.out
```

Cancel:
```bash
scancel <JOBID>
```

You can log out; Slurm jobs continue running.

## Quick Output Sanity Check
```bash
RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/<run_id>
python - <<'PY'
import os, glob
run_root = os.environ["RUN_ROOT"]
for model in ["wan22", "wan21", "lvp"]:
    for dataset in ["physics_iq", "wisa80k"]:
        for task in ["t2v", "i2v"]:
            manifest = f"{run_root}/{model}/{dataset}/{task}/inputs/manifest.jsonl"
            outputs = f"{run_root}/{model}/{dataset}/{task}/outputs"
            expected = sum(1 for _ in open(manifest)) if os.path.isfile(manifest) else 0
            got = len(glob.glob(f"{outputs}/*.mp4")) if os.path.isdir(outputs) else 0
            print(model, dataset, task, "expected", expected, "got", got)
PY
```

## Submodule Reminder
If you edit code inside submodules (for example `third_party/large-video-planner`), you must:
1. commit and push inside that submodule repo,
2. commit and push the parent repo pointer update.
