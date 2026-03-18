#!/bin/bash

hf_model_path=/mnt/weka/home/yucheng/yufeng/llm-pruning-collection-fork-llmshearing/checkpoints/llmshearing/llama3_8b_pruning_scaling_doremi_h_3072_kv_4_mlp_10112_sl8192_bs1_3200ba/hf_pruned_2
tasks=(
    c4
    wikitext
    wikitext2
    cnn_dailymail
    winogrande
    arc_easy
    arc_challenge
    hellaswag
    truthfulqa_mc1
    truthfulqa_mc2
    piqa
    sciq
    boolq
    anli_r1
    anli_r2
    anli_r3
    openbookqa
    rte
    mmlu
    record
)

PROJ_DIR=$(pwd)
export PYTHONPATH=$PROJ_DIR/lm-evaluation-harness:$PYTHONPATH

tasks_str=$(IFS=, ; echo "${tasks[*]}")
# echo "Evaluating tasks: $tasks_str"

python eval.py \
    --hf_path $hf_model_path \
    --tokenizer_path $hf_model_path \
    --tasks $tasks_str
