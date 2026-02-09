# 3dConsistency
Work here builds on persistent memory models used in diffusion based video generation to enforce scene consistency

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
