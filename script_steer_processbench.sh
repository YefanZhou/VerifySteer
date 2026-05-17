#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# 1. ProcessBench configs (gsm8k / math / olympiadbench / omnimath)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR=./results
VECTOR_BASE=/data/yefan/VerifySteer/steering_vector
vector_type=increasing_strictness   # increasing_strictness  decreasing_strictness 
config=gsm8k  # math olympiadbench omnimath

# ── Model config (swap MODEL_PATH to switch model) ────────────────────────────
MODEL_PATH=Qwen/Qwen3-8B          # Qwen/Qwen3-1.7B

case $MODEL_PATH in
    *Qwen3-8B*)
        MODEL_DIR=Qwen3-8B
        LAYER_RANGE="22 23"
        SCALE_INC_EASY=2.0   # gsm8k, math
        SCALE_INC_HARD=1.0   # olympiadbench, omnimath
        SCALE_DEC=1.5
        BETA_INC=0.0
        BETA_DEC=0.0
        ;;
    *Qwen3-1.7B*)
        MODEL_DIR=Qwen3-1.7B
        LAYER_RANGE="16 18"
        SCALE_INC_EASY=3.0
        SCALE_INC_HARD=3.0
        SCALE_DEC=0.5
        BETA_INC=0.6
        BETA_DEC=0.4
        ;;
esac
# ─────────────────────────────────────────────────────────────────────────────


case $config in
    gsm8k)
        vector=$VECTOR_BASE/$MODEL_DIR/gsm8k/${vector_type}.gguf
        scale=$( [ $vector_type = increasing_strictness ] && echo $SCALE_INC_EASY || echo $SCALE_DEC ) ;;
    math)
        vector=$VECTOR_BASE/$MODEL_DIR/math/${vector_type}.gguf
        scale=$( [ $vector_type = increasing_strictness ] && echo $SCALE_INC_EASY || echo $SCALE_DEC ) ;;
    olympiadbench|omnimath)
        vector=$VECTOR_BASE/$MODEL_DIR/olympiadbench_omnimath/${vector_type}.gguf
        scale=$( [ $vector_type = increasing_strictness ] && echo $SCALE_INC_HARD || echo $SCALE_DEC ) ;;
esac

beta=$( [ $vector_type = increasing_strictness ] && echo $BETA_INC || echo $BETA_DEC )

CUDA_VISIBLE_DEVICES=0 python steer_run_eval.py \
    --model_path $MODEL_PATH \
    --configs $config \
    --run_num 0 \
    --control_vector_path $vector \
    --vector_scale $scale \
    --layer_range $LAYER_RANGE \
    --beta $beta \
    --prompt-type processbench_careful_v1 \
    --steering_mode gated_norm_direct_cosine_mask \
    --max_num_seqs 40 \
    --output_dir $OUTPUT_DIR \
    --max_output_length 8192 \
    --greedy












