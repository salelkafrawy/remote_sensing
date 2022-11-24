#!/bin/bash
#SBATCH --array=1-48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -o /home/sara246/scratch/exps/%x_output-%j.out
#SBATCH -e /home/sara246/scratch/exps/%x_error-%j.out

# #SBATCH --cpus-per-task=12  # Cores proportional to GPUs (i.e. #cpus per core/#requested GPUs): 6 on Cedar, 10 on Beluga, 16 on Graham.
# #SBATCH --mem=160G # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
# #SBATCH --time=7-00:00:00  # d-hh:mm:ss

ExpName=$1
CONFIG_FILE="mosaiks_two_layers.yaml"
exp_no=$SLURM_ARRAY_TASK_ID-1
MOSAIKS_MODEL_NAME="two_layer_mosaiks_kmeans_7_whiten_allGeo_bs32_25eachimage_zcaBias_0.001.pt"

# two_layer_mosaiks_kmeans_7_whiten_allGeo_bs32_25eachimage_zcaBias_0.001.pt
# two_layer_mosaiks_kmeans_7_no_whiten_zcaBias_0.001.pt
# two_layer_mosaiks_original_7_no_whiten_zcaBias_0.001.pt
# two_layer_mosaiks_original_7_whiten_zcaBias_0.001.pt


# 1. Activate the local virtual environement locally
module load python/3.9 cuda cudnn nccl
source /home/sara246/scratch/glc_venv/bin/activate

echo "virtual env activated."

start=`date +%s.%N`

# 2. copy all files from scratch (dataset or starting checkpoints)
mkdir $SLURM_TMPDIR/data

# 2.a copy the mosaiks weights
cp $SCRATCH/ckpts/mosaiks/${MOSAIKS_MODEL_NAME} $SLURM_TMPDIR/data

# 2.d copy the dataset
cp $SCRATCH/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip $SLURM_TMPDIR/data
# cp $SCRATCH/data/small_geo_data.zip $SLURM_TMPDIR/data

# 3. Eventually unzip your dataset
unzip -qq $SLURM_TMPDIR/data/geolifeclef-2022-lifeclef-2022-fgvc9.zip -d $SLURM_TMPDIR/data
# unzip -qq $SLURM_TMPDIR/data/small_geo_data.zip -d $SLURM_TMPDIR/data

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "It took ${runtime} to copy and unzip the data"
##############################################################################################

lr_vals=(0.5 0.1 0.02 0.005)
bs_vals=(32 64 128)
opt_names=("SGD" "AdamW")
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
mkdir "$SCRATCH/exps/$CURR_EXP_NAME"
                
echo "Starting job $CURR_EXP_NAME"
python $SCRATCH/rs_beluga/train.py \
                        +data_dir=$SLURM_TMPDIR/data \
                        +log_dir=$SCRATCH/exps/${CURR_EXP_NAME} \
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