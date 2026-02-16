# 3dConsistency
~~Work here builds on persistent memory models used in diffusion based video generation to enforce scene consistency~~
Work here is tries to understand physics consistency in video generation models.

## Local vs Remote Workflow (Wan2.2)

Local setup (no inference, no flash_attn):
1. Create venv: `/opt/homebrew/bin/python3.12 -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Upgrade tools: `python -m pip install -U pip setuptools wheel`
4. Install local deps: `python -m pip install -r requirements/local.txt`
5. Init submodule: `git submodule update --init --recursive`

Sync to cluster:
1. `bash scripts/sync_to_remote.sh`

Remote one-time setup (on cluster):
1. `ssh user@login.rc.fas.harvard.edu`
2. `cd ~/projects/3dConsistency`
3. `python -m venv .venv`
4. `source .venv/bin/activate`
5. `python -m pip install -U pip setuptools wheel`
6. `python -m pip install -r requirements/remote.txt`
7. Install a CUDA-compatible torch build per your cluster guidance.
8. Store temp data at temporary scratch storage found at /n/netscratch. This is a VAST file system with 4 PB of storage and connected via Infiniband fabric. This temporary scratch space is available from all compute nodes.
9. Do not run inference on login nodes; submit jobs via Slurm.
10. Send inference data back to scratch space

Submit inference job:
1. Find partitions you can use: `spart`
2. `GPU_PARTITION=<your_gpu_partition> bash scripts/submit_slurm.sh`
3. Optional custom run id: `bash scripts/submit_slurm.sh my_run_id`

Submit all-model inference job (Wan2.2 + Wan2.1 + LVP):
1. Smoke test for all three:
   `GPU_PARTITION=<your_gpu_partition> TARGETS=wan22,wan21,lvp RUN_MODE=smoke bash scripts/submit_slurm_all.sh`
2. Full LVP path (uses `algorithm=wan_i2v`) plus Wan2.1/Wan2.2:
   `GPU_PARTITION=<your_gpu_partition> TARGETS=wan22,wan21,lvp RUN_MODE=full bash scripts/submit_slurm_all.sh`
3. Run subset only (example: Wan2.1 + LVP):
   `GPU_PARTITION=<your_gpu_partition> TARGETS=wan21,lvp RUN_MODE=smoke bash scripts/submit_slurm_all.sh my_run_id`

Notes:
- `scripts/submit_slurm_all.sh` submits `scripts/remote_infer_all.sh`.
- `RUN_MODE=smoke` runs LVP with `wan_toy` + `dummy` dataset for fast validation.
- `RUN_MODE=full` runs LVP with `wan_i2v` and expects dataset/checkpoints to be ready.

Prepare dataset/task layout (model -> dataset -> {t2v,i2v}):
1. Create a tuple manifest (`.jsonl` or `.csv`) with fields:
   - required: `sample_id` (or `id`), `prompt`, `ground_truth_video`
   - optional: `i2v_image` (if omitted, first frame is extracted from `ground_truth_video`)
2. Run:
   `python scripts/prepare_inference_layout.py --manifest /path/to/tuples.jsonl --run-root /n/netscratch/ydu_lab/Lab/bcupps/results/<run_id> --dataset-name <dataset_name>`
3. Output structure includes:
   - `.../datasets/<dataset>/samples/<sample_id>/` (shared prompt/GT/image assets)
   - `.../<model>/<dataset>/t2v/` and `.../<model>/<dataset>/i2v/`
   - task manifests: `.../<model>/<dataset>/<task>/inputs/manifest.jsonl`

Example JSONL row:
`{"sample_id":"clip_0001","prompt":"A robot arm stacks blocks.","ground_truth_video":"/n/netscratch/.../gt/clip_0001.mp4"}`

Fetch outputs:
1. `bash scripts/fetch_run.sh <run_id>`

General Job Submission tips:

Likely using sbatch ____ to run a script which then allows the compute node access to run what it needs and informatiion about what to write back. 


#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=100           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01

# run code
python -c 'print("Hi there.")'

In general, the script is composed of 4 parts.

the #!/bin/bash line allows the script to be run as a bash script
the #SBATCH lines are technically bash comments, but they set various parameters for the SLURM scheduler
loading any necessary modules and setting any variables, paths, etc.
the command line itself, in this case calling python and having it print a message.


i think though we are close in terms of having the right package there and havingth right python

Upon login:
module load Miniforge3/24.11.3-fasrc02
eval "$(/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda shell.bash hook)"
conda activate ~/projects/3dConsistency/.mamba/wan2

which python

Then, cuda checks for flash_attn:
module load cuda/12.4.1-fasrc01
which nvcc
nvcc --version

needed for flash_attn:
python -m pip install -U psutil

potential test run:
python generate.py \
  --task ti2v-5B \
  --size '1280*704' \
  --ckpt_dir /n/netscratch/ydu_lab/Lab/bcupps/models/Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file /n/netscratch/ydu_lab/Lab/bcupps/results/ti2v_5b_test.mp4


Try 9:09 2/10/26
salloc -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 02:00:00
Then hop into a shell on the allocated node (usually optional; many clusters drop you in already):

srun --pty bash
B. Inside the allocation: load modules + activate env
module purge
module load cuda/12.4.1-fasrc01
module load Miniforge3/24.11.3-fasrc02
eval "$(/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda shell.bash hook)"
conda activate "$HOME/projects/3dConsistency/.mamba/wan2"
Sanity checks:

which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
which nvcc
nvcc --version

sim'd env and close vs open

try the earlier parts of Wan

Another project from Ruojin intuitive fixes for video models

Working on video model for more 3d consistency is too popular, instead, physical plausible. 

some existing datasets to finetune video models not enforcing the physics.

take the ground truth, replicate the prompts and then after generating the video we crate html to corroborate between them. 

next time, after meeting have a todo and send to the slack and have in HTML and google slides
and link to visualization and problems want to discuss. And what worked and what did not work on what you tried in last week.

video model, what I've tried, what's the results.

Trying the LVP seeing the diffrences in generation. 

lvp to base model, they fine tuned on embodied ai (how to interact with world) dataset, robot actually manipulating the objects versus base is images and artists stuff. 

lvp also using diffusion forcing. usual diffusion model is only condition on first image as clean and rest of the latents are noisy. Versus when we have many frames that we know are clean we have idiff noise leveling per frame. not just first frame conditioning, but multi frame conditioning. 

don't have good sense of how they have failed which is why we line up the videos next to each other

TODO:

Text to video and image to video on WISA for both of LVP and WAN2.2 and remark upon the differences.

finetune on wisa and both dataset

50 videos from each.

5 min one video so we can wait 
single GPU should be 33 hours for 50 videos two data sets 2 models

think a little bit about diffusion forcing as well. 

There is a dir from which we can pull the videos for the physics sim one
https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/code/download_physics_iq_data.py

LIST (2/14):
- Test inference on both LVP and WAN2.1 (new env for each?)
  - LVP inf docs
  - WAN2.1 inf docs
- Generate the script that pulls out image from first frame, generates (or takes) prompt, and generates the video for each model and fills out the db which we will then copy from to HTML at the time required

Fixing flash attn because logins might have newere glibc than we expected or than we see on gpu nodes:

module purge
module load cuda/12.4.1-fasrc01
module load Miniforge3/24.11.3-fasrc02

conda activate /n/home12/bcupps/projects/3dConsistency/.mamba/wan21
python -m pip uninstall -y flash-attn flash_attn || true

# IMPORTANT: make sure you are using system gcc, not a devtoolset
which gcc
gcc --version
ldd --version | head -n 1

# Ensure torch libs are discoverable during build and runtime:
export TORCH_LIB_DIR=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export MAX_JOBS=4

python -m pip install --no-build-isolation --no-binary flash-attn flash-attn

# sanity check: does the .so still require GLIBC_2.32?
SO=/n/home12/bcupps/projects/3dConsistency/.mamba/wan21/lib/python3.10/site-packages/flash_attn_2_cuda*.so
ldd -v $SO | grep -n "GLIBC_2.32" -n || echo "No GLIBC_2.32 requirement 👍"

python -c "import flash_attn; print(flash_attn.__version__)"

need to figure out where the flash build is actually coming from.

WE ARE NOT USING FLASH ATTENTION IN ANY MODEL, SO UP THE GPU AND H200 count. 


PLAN:
great, this is good, now lets see if we can run that inference script because eventually I want to have runs in the scratch space on remote broken doesn into model which breaks in 2.2 2.1 lvp each of which breaks into a dataset which each break into img to video and text to video. But, that structure will have to be created by running these scripts that put the outputs in exactly the right places and get the inputs from exactly the right place to match. I am going to load in tuples of prompts, ground truth data into the scratch space somewhere and then want scripts that for each goes through and runs+ records the prompt to video and then takes out first frame and runs a img to video as well and then stores in the format that I wanted (adding the groudtruth to there as well). 

ls -ltr ~/projects/3dConsistency/runs/20260215_233048/slurm-*.out
