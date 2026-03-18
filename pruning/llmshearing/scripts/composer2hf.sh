#!/bin/bash

MODEL_DIR=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_5_mlp_9984_sl8192_bs1_3200ba

convert_hf() {
    MODEL_DIR=$1
    MODEL_PATH=$MODEL_DIR/latest-rank0.pt
    OUTPUT_PATH=$MODEL_DIR/hf-pruned
    MODEL_CLASS=LlamaForCausalLM
    MODEL_NAME=Sheared-Llama3
    TOKENIZER_NAME=/scratch/yx3038/cache/Llama-3.1-8B

    # Auto-parse dimensions from the pruned checkpoint.
    # Assumes head_dim=128 (Llama3 standard) and n_heads = 4 * n_kv_heads.
    read HIDDEN_SIZE NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS NUM_HIDDEN_LAYERS INTERMEDIATE_SIZE VOCAB_SIZE <<< $(python3 -c "
import torch
w = torch.load('$MODEL_PATH', map_location='cpu', mmap=True)
if 'state' in w: w = w['state']['model']
HEAD_DIM = 128
hidden_size = w['model.transformer.wte.weight'].shape[1]
vocab_size = w['model.transformer.wte.weight'].shape[0]
intermediate_size = w['model.transformer.blocks.0.mlp.gate_proj.weight'].shape[0]
n_kv_heads = w['model.transformer.blocks.0.attn.wk.weight'].shape[0] // HEAD_DIM
n_heads = n_kv_heads * 4
num_layers = max(int(k.split('blocks.')[1].split('.')[0]) for k in w if 'blocks.' in k) + 1
print(hidden_size, n_heads, n_kv_heads, num_layers, intermediate_size, vocab_size)
")
    echo "Parsed dims: hidden_size=$HIDDEN_SIZE num_attention_heads=$NUM_ATTENTION_HEADS num_key_value_heads=$NUM_KEY_VALUE_HEADS num_hidden_layers=$NUM_HIDDEN_LAYERS intermediate_size=$INTERMEDIATE_SIZE vocab_size=$VOCAB_SIZE"

    python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model $MODEL_PATH

    python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
            model_class=${MODEL_CLASS} \
            hidden_size=${HIDDEN_SIZE} \
            num_attention_heads=${NUM_ATTENTION_HEADS} \
            num_hidden_layers=${NUM_HIDDEN_LAYERS} \
            intermediate_size=${INTERMEDIATE_SIZE} \
            num_key_value_heads=${NUM_KEY_VALUE_HEADS} \
            vocab_size=${VOCAB_SIZE} \
            tokenizer_name=${TOKENIZER_NAME} \
            rope_theta=500000 \
            _name_or_path=${MODEL_NAME}
}

for model_dir in /scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b*; do
    [ -d "$model_dir" ] || continue
    echo "Converting model in $model_dir"
    convert_hf "$model_dir"
done