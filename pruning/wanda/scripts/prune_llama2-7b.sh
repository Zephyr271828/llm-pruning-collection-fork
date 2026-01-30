#!/bin/bash


set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate wanda

methods=(
    wanda
    sparsegpt
    magnitude
)

sparsities=(
    unstructured
    2:4
    4:8
)

ratio=0.5

PROJ_DIR=$(pwd)
export PYTHONPATH=${PROJ_DIR}/src/lib:${PYTHONPATH:-''}
export PYTHONPATH=${PROJ_DIR}/lm-evaluation-harness:${PYTHONPATH}
model_path=meta-llama/Llama-2-7b-hf
model_name=$(basename ${model_path})
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs

cd $PROJ_DIR/src
for method in "${methods[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        echo "[INFO] Pruning with method: $method and sparsity: $sparsity and ratio: $ratio"
        python main.py \
            --model ${model_path} \
            --prune_method ${method} \
            --sparsity_ratio ${ratio} \
            --sparsity_type ${sparsity} \
            --save ${log_dir}/${method}/${sparsity}/ \
            --save_model ${save_dir}/${method}/${model_name}_${method}_${sparsity}_${ratio} \
            --eval_zero_shot
    done
done