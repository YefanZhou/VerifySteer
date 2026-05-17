#!/bin/bash

# Baseline (no steering) — Hard2Verify (h2v) config with thinking mode

OUTPUT_DIR=./results

# ── Model config (swap MODEL_PATH to switch model) ────────────────────────────
MODEL_PATH=Qwen/Qwen3-8B          
# ─────────────────────────────────────────────────────────────────────────────

CUDA_VISIBLE_DEVICES=0 python run_eval_baseline.py \
    --model_path $MODEL_PATH \
    --configs h2v \
    --run_num 0 \
    --prompt-type h2v_error_id \
    --max_num_seqs 30 \
    --output_dir $OUTPUT_DIR \
    --max_output_length 16384 \
    --enable_thinking




MODEL_PATH=Salesforce/FARE-20B          
# ─────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python run_eval_baseline.py \
    --model_path $MODEL_PATH \
    --configs h2v \
    --run_num 0 \
    --prompt-type fare \
    --max_num_seqs 40 \
    --output_dir $OUTPUT_DIR \
    --max_output_length 16384 \
    --greedy \
    --max_model_len 40960 \
    --gpu_memory_utilization 0.85 \
    --enforce_eager True 

