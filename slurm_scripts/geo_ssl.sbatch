#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=40G
#SBATCH -p main
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_error-%j.out

ExpName=$1
CONFIG_FILE="geo_ssl.yaml"
SSL_CKPT_PATH="" # BE CAREFUL: not included in the python command 


echo "exp name set."
echo "${SCRATCH}/ecosystem_project/exps/"

# 1. Create your environement locally
module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda activate ffcv2
conda deactivate
conda activate ffcv2

echo "conda env activated."

start=`date +%s.%N`

# 2. copy all files from scratch (dataset or starting checkpoints)
mkdir $SLURM_TMPDIR/data

# 2.b copy ckpt
cp $SCRATCH/ecosystem_project/ckpts/${SSL_CKPT_PATH} $SLURM_TMPDIR/data

# 2.c Copy your dataset to the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "It took ${runtime} to copy and unzip the data"

# 4. create tmp logging dir
mkdir $SCRATCH/ecosystem_project/exps/${ExpName}

echo "Starting job"

python $SCRATCH/ecosystem_project/rs_mila/train_SSL.py \
		+data_dir=$SLURM_TMPDIR/data \
        +log_dir=$SCRATCH/ecosystem_project/exps/${ExpName} \
        +exp_name=$ExpName \
        +mocov2_ssl_ckpt_path="" \
        ++args.config_file=${CONFIG_FILE} 

echo 'done'

#$SLURM_TMPDIR/data/${SSL_CKPT_PATH}