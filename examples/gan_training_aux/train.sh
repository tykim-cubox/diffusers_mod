#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=exp3
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=16
#SBATCH --tasks-per-node=8
#SBATCH --mem=0
##SBATCH --mem=20gb
#SBATCH --time=0-24:00:00
#SBATCH -o slurm.%x.%j.out

echo "Run started at:- "
date

srun singularity exec --nv --contain -B /purestorage/:/purestorage /purestorage/project/tyk/slurm/imgs/i2i.sif \
bash -c "cd /purestorage/project/tyk/project6/predefined_iti_train && \
python train.py --gpus=8 --config_path=/purestorage/project/tyk/project6/predefined_iti_train/configs/v3_1.yaml --num_nodes=2 --max_steps=50000 --val_freq=55 --exp_name=v2-exp3-idclip"

echo ""
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "################################################################"

## wandb login --relogin --host http://211.168.94.174:8080 local-6007eb7118e2a6abf95ee393d559eae7c6e23e9f && \