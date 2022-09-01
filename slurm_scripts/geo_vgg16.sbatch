#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p long
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/cnn_vgg16_exp_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/cnn_vgg16_exp_error-%j.out

ExpName="cnn_vgg16_exp"

echo "exp name set."
echo "${SCRATCH}/ecosystem_project/exps/"

# 1. Create your environement locally
module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda activate glc
conda deactivate
conda activate glc

echo "conda env activated."

start=`date +%s.%N`

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
mkdir $SLURM_TMPDIR/data
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "It took ${runtime} to copy and unzip the data"

# 4. create tmp logging dir
mkdir $SCRATCH/ecosystem_project/exps/${ExpName}

echo "Starting job"
python /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/geolife_kaggle/train.py \
		+data_dir=$SLURM_TMPDIR/data \
        +log_dir=$SCRATCH/ecosystem_project/exps/${ExpName} \
        ++args.config_file="cnn_baseline_vgg.yaml"

echo 'done'
