#!/bin/bash

#SBATCH --job-name=eval_domain_loss_%j
#SBATCH --output=logs/eval_domain_loss_%j.out
#SBATCH --error=logs/eval_domain_loss_%j.err
#SBATCH --partition=sfscai
#SBATCH --nodes=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h20:1
#SBATCH --time=12:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue

hf_model_paths=(
    # "/scratch/yx3038/pruning/llm-pruning-collection-fork/checkpoints/minitron/Llama-3.1-8B_width_task_wikitext_hidden_size_3072_ffn_hidden_size_9216_calib_size_128_seqlen_8192"
    # "/scratch/yx3038/cache/Llama-3.1-Minitron-4B-Width-Base"
    "/scratch/yx3038/cache/Llama-3.2-1B"
    # "/scratch/yx3038/cache/Llama-3.2-3B"
    # "/scratch/yx3038/cache/Llama-3.1-8B"
    # "/scratch/yx3038/cache/Llama-3.1-8B-Instruct"
    # "/scratch/yx3038/cache/Llama-3.1-70B"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate minitron

for hf_model_path in "${hf_model_paths[@]}"; do
    python /scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/pruning/llmshearing/eval_domain_loss.py \
        --model_path "$hf_model_path" \
        --max_samples_per_domain 128 \
        --seq_len=8192 \
        --device_map auto \
        --data_root "/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/pruning/llmshearing/llmshearing/data/redpajama/for_prune_llama3.2"
done