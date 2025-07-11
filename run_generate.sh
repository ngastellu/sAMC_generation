#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --mem-per-gpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=slrmgenout
#SBATCH --error=slrmgen.err


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


module load httpproxy

export MASTER_PORT=12349
# WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm #SBATCH --cpus-per-task=1
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python generate.py --model_name='graphene' --softmax_temp=1.0 --sample_outpaint_ratio=5 --n_samples=3



