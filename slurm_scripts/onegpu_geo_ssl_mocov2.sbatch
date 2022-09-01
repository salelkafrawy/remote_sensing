#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p long
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_ssl_correct_norm_ckpt75_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_ssl_correct_norm_ckpt75_error-%j.out

ExpName="geolife_ssl_correct_norm_ckpt75"

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

# 2.a copy the mosaiks filters
cp $SCRATCH/ecosystem_project/ckpts/mosaik_geo_filters/mosaik_512_filters_geo.npy $SLURM_TMPDIR/data

# 2.b copy seco ckpt
cp $SCRATCH/ecosystem_project/ckpts/multigpu_correct_norms_epoch_74.ckpt $SLURM_TMPDIR/data

# 2.c copy the random initialization of resnet50
cp $SCRATCH/ecosystem_project/ckpts/resnet50_random_init $SLURM_TMPDIR/data

# 2.d copy the standard scaler to the compute node
cp $SCRATCH/env_vars_scaler.gz $SLURM_TMPDIR/data

# 2.c Copy your dataset to the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "It took ${runtime} to copy and unzip the data"

# 4. create tmp logging dir
mkdir $SCRATCH/ecosystem_project/exps/${ExpName}

echo "Starting job"

python $SCRATCH/ecosystem_project/remote_sensing/train_SSL.py \
		+data_dir=$SLURM_TMPDIR/data \
        +log_dir=$SCRATCH/ecosystem_project/exps/${ExpName} \
        +mosaiks_weights_path=$SLURM_TMPDIR/data/mosaik_512_filters_geo.npy \
        +random_init_path=$SLURM_TMPDIR/data/resnet50_random_init \
        +mocov2_ssl_ckpt_path=$SLURM_TMPDIR/data/multigpu_correct_norms_epoch_74.ckpt \
        +ffcv_write_path=$SLURM_TMPDIR/data \
        ++args.config_file="geo_ssl_enc.yaml" 

echo 'done'
