#!/bin/bash
#SBATCH --array=1-8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p long
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_error-%j.out

ExpName=$1
CONFIG_FILE="cnn_clip_resnet.yaml"
exp_no=$SLURM_ARRAY_TASK_ID-1
RESNET50_MODEL_NAME=""
#================================================
#------------------------------
# CLIP resnet50:
# config file: cnn_clip_resnet.yaml
#------------------------------
# SSL4EO:
# config file: cnn_SSL4EO.yaml   AND   cnn_SSL4EO_h2t_ft.yaml
# resnet50_ssl_e99_SSL4EO.pth
#------------------------------
# SeCo:
# config file: cnn_seco.yaml  AND   cnn_seco_h2t_ft.yaml
# seco_resnet50_1m.ckpt
#------------------------------
# MOSAIKS+Kmeans:
# config file: cnn_resnet50_MOSAIKS.yaml
# resnet50_mosaiks_whiten.pt
# resnet50_mosaiks_no_whiten_minibatch.pt
#------------------------------
# ImageNet:
# config file: cnn_resnet50.yaml   AND   cnn_resnet50_h2t_ft.yaml
#------------------------------
# ImageNet+SSL:
# config file: cnn_IN_SSL.yaml
# resnet50_ssl_200ep_ImageNet.pth
#------------------------------
# random init:
# config file: cnn_resnet50_random.yaml
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
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "It took ${runtime} to copy and unzip the data"
##############################################################################################

lr_vals=(0.01, 0.05)
bs_vals=(32 64)
opt_names=("SGD")
sched_names=("CosineRestarts" "ReduceLROnPlateau")
exp_names=()

lr_exps=()
bs_exps=()
opt_exps=()
sched_exps=()

for lr in ${lr_vals[@]}
do
    for bs in ${bs_vals[@]}
    do
        for sched in ${sched_names[@]}
        do  
            for opt in ${opt_names[@]}
            do
                curr_exp_name=("${ExpName}_lr${lr}_bs${bs}_${opt}_${sched}")
                lr_exps+=(${lr})
                bs_exps+=(${bs})
                opt_exps+=(${opt})
                sched_exps+=(${sched})
                exps_names+=($curr_exp_name)
            done
        done
    done
done

for exp in ${exps_names[@]}
do
    echo $exp
done
##############################################################################################

# create tmp logging dir
CURR_EXP_NAME="${exps_names[$exp_no]}"
mkdir "$SCRATCH/ecosystem_project/exps/$CURR_EXP_NAME"

echo "Starting job"
python /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/rs_mila/train.py \
                        +data_dir=$SLURM_TMPDIR/data \
                        +log_dir=$SCRATCH/ecosystem_project/exps/${CURR_EXP_NAME} \
                        +exp_name=$CURR_EXP_NAME \
                        +mosaiks_weights_path="" \
                        +cnn_ckpt_path=$SLURM_TMPDIR/data/${RESNET50_MODEL_NAME} \
                        +ffcv_write_path=$SLURM_TMPDIR/data \
                        ++args.config_file=${CONFIG_FILE} \
                        +learning_rate=${lr_exps[$exp_no]} \
                        +scheduler_name=${sched_exps[$exp_no]} \
                        +batch_size=${bs_exps[$exp_no]} \
                        +optimizer=${opt_exps[$exp_no]}

echo 'done'
