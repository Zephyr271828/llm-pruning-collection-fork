#!/bin/bash
set -euo pipefail

# MODEL_DIR=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_5_mlp_9984_sl8192_bs1_3200ba
MODEL_DIR=/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing-2/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_4_mlp_10112_sl8192_bs1_3200ba

convert_hf() {
    local MODEL_DIR="$1"
    local MODEL_PATH="$MODEL_DIR/latest-rank0.pt"
    local OUTPUT_PATH="$MODEL_DIR/hf-pruned"
    local MODEL_CLASS="LlamaForCausalLM"
    local MODEL_NAME="Sheared-Llama3"
    local TOKENIZER_NAME="/scratch/yx3038/cache/Llama-3.1-8B"

    local DIR_BASENAME
    DIR_BASENAME="$(basename "$MODEL_DIR")"

    local HIDDEN_SIZE=""
    local NUM_KEY_VALUE_HEADS=""
    local INTERMEDIATE_SIZE=""

    if [[ "$DIR_BASENAME" =~ (^|_)h_([0-9]+)($|_) ]]; then
        HIDDEN_SIZE="${BASH_REMATCH[2]}"
    fi
    if [[ "$DIR_BASENAME" =~ (^|_)kv_([0-9]+)($|_) ]]; then
        NUM_KEY_VALUE_HEADS="${BASH_REMATCH[2]}"
    fi
    if [[ "$DIR_BASENAME" =~ (^|_)mlp_([0-9]+)($|_) ]]; then
        INTERMEDIATE_SIZE="${BASH_REMATCH[2]}"
    fi

    if [[ -z "$HIDDEN_SIZE" || -z "$NUM_KEY_VALUE_HEADS" || -z "$INTERMEDIATE_SIZE" ]]; then
        echo "Failed to parse one or more of hidden_size / kv_heads / mlp_size from dir name: $DIR_BASENAME" >&2
        exit 1
    fi

    local NUM_ATTENTION_HEADS
    NUM_ATTENTION_HEADS=$((NUM_KEY_VALUE_HEADS * 4))

    read NUM_HIDDEN_LAYERS VOCAB_SIZE <<< "$(python3 - <<PY
import torch

w = torch.load("$MODEL_PATH", map_location="cpu", mmap=True)
if "state" in w:
    w = w["state"]["model"]

num_layers = max(int(k.split("blocks.")[1].split(".")[0]) for k in w if "blocks." in k) + 1
vocab_size = w["model.transformer.wte.weight"].shape[0]

print(num_layers, vocab_size)
PY
)"

    echo "Parsed dims:"
    echo "  hidden_size=$HIDDEN_SIZE"
    echo "  num_attention_heads=$NUM_ATTENTION_HEADS"
    echo "  num_key_value_heads=$NUM_KEY_VALUE_HEADS"
    echo "  num_hidden_layers=$NUM_HIDDEN_LAYERS"
    echo "  intermediate_size=$INTERMEDIATE_SIZE"
    echo "  vocab_size=$VOCAB_SIZE"

    # exit 0

    python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model "$MODEL_PATH"
    local PRUNED_PATH="$MODEL_DIR/pruned-$(basename "$MODEL_PATH")"

    python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf "$PRUNED_PATH" "$OUTPUT_PATH" \
        model_class="${MODEL_CLASS}" \
        hidden_size="${HIDDEN_SIZE}" \
        head_dim=128 \
        num_attention_heads="${NUM_ATTENTION_HEADS}" \
        num_hidden_layers="${NUM_HIDDEN_LAYERS}" \
        intermediate_size="${INTERMEDIATE_SIZE}" \
        num_key_value_heads="${NUM_KEY_VALUE_HEADS}" \
        vocab_size="${VOCAB_SIZE}" \
        tokenizer_name="${TOKENIZER_NAME}" \
        _name_or_path="${MODEL_NAME}"
}

echo "Converting model in $MODEL_DIR"
convert_hf "$MODEL_DIR"