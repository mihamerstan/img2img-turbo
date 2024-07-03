#! /bin/bash -l
#SBATCH --job-name=img2img-train
#SBATCH --output=train.out
#SBATCH --partition=gpum
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=michael.hamer.stanley@gmail.com
#SBATCH --time=48:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --error=train.err
#SBATCH --cpus-per-task=12


# Load module and load environment
source ~/.bashrc
conda activate img2img-turbo
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/mhstan/img2img-turbo/src"
cd $WORKDIR

read -r -d '' standard_args << EOM
--pretrained_model_name_or_path="/group/jmearlesgrp/scratch/mhstan/model_checkpoints/img2img-turbo/sd-turbo" \
--output_dir="/group/jmearlesgrp/intermediate_data/mhstan/img2img-turbo/checkpoints" \
--dataset_folder "/group/jmearlesgrp/intermediate_data/mhstan/img2img-turbo/data/helios_to_borden_night" \
--train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
--learning_rate="1e-5" --max_train_steps=25000 \
--train_batch_size=1 --gradient_accumulation_steps=1 \
--report_to "wandb" --tracker_project_name "cyclegan-turbo-helios-to-borden-night" \
--enable_xformers_memory_efficient_attention --validation_steps 250 \
--lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 --enable_xformers_memory_efficient_attention
EOM

accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py $standard_args
