#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/ecosystem-embedding/resnet_test_slurm-%j.out
#SBATCH -p long


module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda activate ebird-env
conda deactivate
conda activate ebird-env

echo "Starting job"
python ~/scratch/ecosystem_project/eco-repo/train2.py ++auto_lr_find=False args.config=/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/eco-repo/configs/classification.yaml
echo 'done'
