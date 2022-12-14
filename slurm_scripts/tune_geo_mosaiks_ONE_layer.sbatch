#!/bin/bash
#SBATCH --array=1-36
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p long
#SBATCH -o /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_output-%j.out
#SBATCH -e /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/%x_error-%j.out

ExpName=$1
CONFIG_FILE="mosaiks_one_layer.yaml"
exp_no=$SLURM_ARRAY_TASK_ID-1
MOSAIKS_MODEL_NAME="one_layer_mosaiks_kmeans_7_whiten_zcaBias_0.001_bias_False_filters_512.pt"

# one_layer_mosaiks_kmeans_7_no_whiten_zcaBias_0.001_bias_True_filters_100.pt
# one_layer_mosaiks_kmeans_7_whiten_zcaBias_0.001_bias_True_filters_100.pt
# one_layer_mosaiks_original_7_no_whiten_zcaBias_0.001_bias_True_filters_100.pt
# one_layer_mosaiks_original_7_whiten_zcaBias_0.001_bias_True_filters_100.pt
# one_layer_mosaiks_kmeans_7_whiten_zcaBias_0.001_bias_False_filters_512
# one_layer_mosaiks_original_7_whiten_zcaBias_0.001_bias_False_filters_512

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
cp $SCRATCH/ecosystem_project/ckpts/${MOSAIKS_MODEL_NAME} $SLURM_TMPDIR/data

# 2.d copy the dataset
cp /network/datasets/geolifeclef/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "It took ${runtime} to copy and unzip the data"
##############################################################################################

# CHANGE NUMBER OF ARRAY JOBS WHEN CHANGING HERE

lr_vals=(0.001 0.01 0.1 0.5)
bs_vals=(32 64 128)
opt_names=("SGD")
sched_names=("CosineRestarts" "ReduceLROnPlateau" "OneCycleLR")
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
                
echo "Starting job $CURR_EXP_NAME"
python /home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/remote_sensing/train.py \
                        +data_dir=$SLURM_TMPDIR/data \
                        +log_dir=$SCRATCH/ecosystem_project/exps/${CURR_EXP_NAME} \
                        +exp_name=$CURR_EXP_NAME \
                        +mosaiks_weights_path=$SLURM_TMPDIR/data/${MOSAIKS_MODEL_NAME} \
                        ++args.config_file=${CONFIG_FILE} \
                        +cnn_ckpt_path="" \
                        +mocov2_ssl_ckpt_path="" \
                        +ffcv_write_path=$SLURM_TMPDIR/data \
                        +learning_rate=${lr_exps[$exp_no]} \
                        +scheduler_name=${sched_exps[$exp_no]} \
                        +batch_size=${bs_exps[$exp_no]} \
                        +optimizer=${opt_exps[$exp_no]}