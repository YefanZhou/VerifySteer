#!/bin/bash

# Baseline (no steering) — ProcessBench configs (gsm8k / math / olympiadbench / omnimath)

OUTPUT_DIR=./results

# ── Model config (swap MODEL_PATH to switch model) ────────────────────────────
MODEL_PATH=Qwen/Qwen3-8B          # Qwen/Qwen3-1.7B  Qwen/Qwen3-8B
# ─────────────────────────────────────────────────────────────────────────────
config=gsm8k  # math olympiadbench omnimath

CUDA_VISIBLE_DEVICES=0 python run_eval_baseline.py \
    --model_path $MODEL_PATH \
    --configs $config \
    --run_num 0 \
    --prompt-type processbench_careful_v1 \
    --max_num_seqs 40 \
    --output_dir $OUTPUT_DIR \
    --max_output_length 8192 \
    --greedy

