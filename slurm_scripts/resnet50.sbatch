#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p main
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_error-%j.out

ExpName=$1
CONFIG_FILE="cnn_resnet50_MOSAIKS.yaml"
RESNET50_MODEL_NAME="resnet50_mosaiks_whiten.pt"
#================================================
# SEN12MS:
# config file: cnn_resnet50_from_ssl.yaml
# resnet50_ssl_SEN12MS.pth
#------------------------------
# SSL4EO:
# config file: cnn_SSL4EO.yaml
# resnet50_ssl_e99_SSL4EO.pth
#------------------------------
# SeCo:
# config file: cnn_seco.yaml
# seco_resnet50_1m.ckpt
#------------------------------
# MOSAIKS+Kmeans:
# config file: cnn_resnet50_MOSAIKS.yaml
# resnet50_mosaiks_whiten.pt
# resnet50_mosaiks_no_whiten_minibatch.pt
#------------------------------
# ImageNet:
# config file: cnn_resnet50.yaml
#================================================

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

# 2.b copy a custom resnet50 model
cp $SCRATCH/ecosystem_project/ckpts/resnet50/${RESNET50_MODEL_NAME} $SLURM_TMPDIR/data

# 2.d copy the dataset
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
# cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
cp $SCRATCH/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
# unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data


end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "It took ${runtime} to copy and unzip the data"

# 4. create tmp logging dir
mkdir $SCRATCH/ecosystem_project/exps/${ExpName}

echo "Starting job"
python /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/remote_sensing/train.py \
		+data_dir=$SLURM_TMPDIR/data \
        +log_dir=$SCRATCH/ecosystem_project/exps/${ExpName} \
        +exp_name=$ExpName \
        +mosaiks_weights_path="" \
        +cnn_ckpt_path=$SLURM_TMPDIR/data/${RESNET50_MODEL_NAME} \
        +mocov2_ssl_ckpt_path="" \
        +ffcv_write_path=$SLURM_TMPDIR/data \
        ++args.config_file=${CONFIG_FILE} \

echo 'done'
