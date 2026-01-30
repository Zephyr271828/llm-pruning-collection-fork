#!/bin/bash

#SBATCH --chdir=.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate minitron

PROJ_DIR=$(pwd)

model_path=meta-llama/Llama-2-7b-hf
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs
calib_size=1024

python $PROJ_DIR/main.py bi \
    --model_path ${model_path} \
    --save_dir ${save_dir}/shortgpt \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 23 \
    --calib_size ${calib_size} \
    --seq_len 4096

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir}/minitron \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 23 \
    --calib_size ${calib_size} \
    --seq_len 4096

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir}/minitron \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 31 \
    --calib_size ${calib_size} \
    --seq_len 4096