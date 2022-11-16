#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p main
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_error-%j.out

ExpName=$1

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

# 2.a copy the mosaiks weights
cp $SCRATCH/ecosystem_project/ckpts/two_layer_mosaiks_kmeans_7_no_whiten_zcaBias_0.001.pt $SLURM_TMPDIR/data

# 2.d copy the dataset
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data

##############################################################################################

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
        +mosaiks_weights_path=$SLURM_TMPDIR/data/two_layer_mosaiks_kmeans_7_no_whiten_zcaBias_0.001.pt \
        ++args.config_file="mosaiks_two_layers.yaml" \
        +cnn_ckpt_path="" \
        +mocov2_ssl_ckpt_path="" \
        +ffcv_write_path=$SLURM_TMPDIR/data \
        

echo 'done'
