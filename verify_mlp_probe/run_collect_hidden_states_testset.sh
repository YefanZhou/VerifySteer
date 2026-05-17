#!/bin/bash

base_path=./
output_dir=${base_path}/state_for_predict_correctness
model_name=Qwen/Qwen3-1.7B

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python collect_hidden_states.py \
    --model_name ${model_name} \
    --output_dir ${output_dir} \
    --config gsm8k \
    --prompt-type processbench_careful_v1 \
    --base-path ${base_path}
