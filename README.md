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

With one GPU, about 20 minutes for a video

With Muliple:
RUN_ID=20260218_batch1
RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/$RUN_ID

for m in wan22 wan21 lvp; do
  for d in physics_iq wisa80k; do
    for t in t2v i2v; do
      DATASET_NAMES=$d TARGETS=$m TASKS=$t \
      RUN_ROOT="$RUN_ROOT" \
      GPU_PARTITION=gpu_h200 GRES='gpu:nvidia_h200:1' \
      CPUS_PER_TASK=8 MEMORY=200G WALLTIME=24:00:00 \
      MAX_SAMPLES=0 SKIP_EXISTING=1 CONTINUE_ON_ERROR=1 LOCAL_SUBMIT=1 \
      bash scripts/submit_slurm_layout.sh "$RUN_ID"
    done
  done
done

Submitted batch job 61061000
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=wan22 tasks=t2v
Submitted batch job 61061002
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=wan22 tasks=i2v
Submitted batch job 61061004
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=wan22 tasks=t2v
Submitted batch job 61061009
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=wan22 tasks=i2v
Submitted batch job 61061022
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=wan21 tasks=t2v
Submitted batch job 61061027
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=wan21 tasks=i2v
Submitted batch job 61061028
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=wan21 tasks=t2v
Submitted batch job 61061029
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=wan21 tasks=i2v
Submitted batch job 61061030
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=lvp tasks=t2v
Submitted batch job 61061034
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=physics_iq
targets=lvp tasks=i2v
Submitted batch job 61061036
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=lvp tasks=t2v
Submitted batch job 61061041
Submitted layout run_id=20260218_batch1
run_root=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1
datasets=wisa80k
targets=lvp tasks=i2v


Subsetting manifest for better selection given that it's going to take a while:
[bcupps@boslogin08 3dConsistency]$ conda activate /n/home12/bcupps/projects/3dConsistency/.mamba/wan2
(/n/home12/bcupps/projects/3dConsistency/.mamba/wan2) [bcupps@boslogin08 3dConsistency]$ python scripts/subset_layout_manifests.py \
>   --run-root "$RUN_ROOT" \
>   --datasets physics_iq,wisa80k \
>   --max-per-dataset 20 \
>   --perspective-preference center,left,right
[physics_iq] selected 20 sample_ids (max_per_dataset=20, preference=center,left,right)
[physics_iq] first 5 selected: ['0001_perspective-left_trimmed-ball-and-block-fall', '0002_perspective-center_trimmed-ball-and-block-fall', '0003_perspective-right_trimmed-ball-and-block-fall', '0004_perspective-left_trimmed-ball-behind-rotating-paper', '0005_perspective-center_trimmed-ball-behind-rotating-paper']
  [wan22/physics_iq/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan22/physics_iq/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan21/physics_iq/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan21/physics_iq/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [lvp/physics_iq/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [lvp/physics_iq/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
[wisa80k] selected 20 sample_ids (max_per_dataset=20, preference=center,left,right)
[wisa80k] first 5 selected: ['49fb8801b94439ce89ea3e8e866fdaebb3637afdb65f49fec24a53fd7e381367.mp4', '8810444677f17196492ef52c8b1cb786dca17782573243e54625ac50d00a51c5.mp4', 'f381a87f9deb85ef849b384dace6f17f0e08171f6f24d0bca2782ea8b6f25343.mp4', '6cd603d147a10a523e51216cdbd1a1da3954b8670840313325c29da7a49ba917.mp4', '1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4']
  [wan22/wisa80k/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan22/wisa80k/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan21/wisa80k/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [wan21/wisa80k/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [lvp/wisa80k/t2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
  [lvp/wisa80k/i2v] 100 -> 20 rows (backup: manifest.jsonl.pre_subset.bak)
Manifest subsetting complete.


## SERVING HTML:
python scripts/serve_private_gallery.py \
  --run-root "$RUN_ROOT" \
  --run-root /n/netscratch/ydu_lab/Lab/bcupps/results/<older_run_id> \
  --bind 127.0.0.1 \
  --port 8000

to access on local browser:
ssh -J bcupps@login.rc.fas.harvard.edu -N -L 18000:127.0.0.1:8000 bcupps@boslogin08
or 
ssh -N -L 8000:127.0.0.1:8000 bcupps@login.rc.fas.harvard.edu
then: open http://127.0.0.1:18000


## LORA Finetuning: