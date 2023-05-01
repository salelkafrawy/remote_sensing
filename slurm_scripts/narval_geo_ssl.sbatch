#!/bin/bash
#SBATCH --gres=gpu:a100:4  # 4 GPUs of type a100
#SBATCH --cpus-per-task=12  # Cores proportional to GPUs (i.e. #cpus per core/#requested GPUs): 6 on Cedar, 10 on Beluga, 16 on Graham.
#SBATCH --mem=80G # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=7-00:00:00  # d-hh:mm:ss
#SBATCH --mail-user=sara.ebrahim.elkafrawy@umontreal.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/sara246/scratch/exps/%x_output_%j.out
#SBATCH -e /home/sara246/scratch/exps/%x_error_%j.out

ExpName=$1
CONFIG_FILE="geo_ssl_narval.yaml"

# 1. Activate the local virtual environement locally
module load python/3.9 cuda cudnn nccl

source /home/sara246/glc_venv/bin/activate

echo "virtual env activated."

start=`date +%s.%N`

# 2. copy all files from scratch (dataset or starting checkpoints)
mkdir $SLURM_TMPDIR/data

# 2.b copy seco ckpt
# cp $SCRATCH/ckpts/multigpu_correct_augmentations_epoch_151.ckpt $SLURM_TMPDIR/data

# 2.c Copy your dataset to the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/data/small_geo_data.zip $SLURM_TMPDIR/data


# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data



end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "It took ${runtime} to copy and unzip the data"

# 4. create tmp logging dir
mkdir $SCRATCH/exps/${ExpName}

echo "Starting job"

python $SCRATCH/rs_narval/train_SSL.py \
		+data_dir=$SLURM_TMPDIR/data \
        +log_dir=$SCRATCH/exps/${ExpName} \
        +exp_name=$ExpName \
        +mocov2_ssl_ckpt_path="" \
        ++args.config_file=${CONFIG_FILE}  

echo 'done'
