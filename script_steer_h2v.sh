#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# Hard2Verify (h2v) steering configs
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR=./results
VECTOR_BASE=/data/yefan/VerifySteer/steering_vector
vector_type=increasing_strictness   # increasing_strictness  decreasing_strictness

# ── Model config (swap MODEL_PATH to switch model) ────────────────────────────
MODEL_PATH=Qwen/Qwen3-8B            # Qwen/Qwen3-8B  Salesforce/FARE-20B

case $MODEL_PATH in
    *Qwen3-8B*)
        CUDA_DEVICES=0
        MODEL_DIR=Qwen3-8B-thinking
        LAYER_RANGE="22 23"
        SCALE_INC=1.5
        SCALE_DEC=1.5
        BETA_INC=0.0
        BETA_DEC=0.1
        PROMPT_TYPE=h2v_error_id
        MAX_NUM_SEQS=30
        EXTRA_ARGS="--enable_thinking"
        ;;
    *FARE-20B*)
        CUDA_DEVICES=0
        MODEL_DIR=FARE-20B
        LAYER_RANGE="17 21"
        SCALE_INC=1.0
        SCALE_DEC=1.0
        BETA_INC=0.4
        BETA_DEC=0.4
        PROMPT_TYPE=fare
        MAX_NUM_SEQS=40
        EXTRA_ARGS="--max_model_len 40960 --gpu_memory_utilization 0.85 --enforce_eager True --greedy"
        ;;
esac
# ─────────────────────────────────────────────────────────────────────────────

scale=$( [ $vector_type = increasing_strictness ] && echo $SCALE_INC || echo $SCALE_DEC )
beta=$( [ $vector_type = increasing_strictness ] && echo $BETA_INC || echo $BETA_DEC )
vector=$VECTOR_BASE/$MODEL_DIR/h2v/${vector_type}.gguf

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python steer_run_eval.py \
    --model_path $MODEL_PATH \
    --configs h2v \
    --run_num 0 \
    --control_vector_path $vector \
    --vector_scale $scale \
    --layer_range $LAYER_RANGE \
    --beta $beta \
    --prompt-type $PROMPT_TYPE \
    --steering_mode gated_norm_direct_cosine_mask \
    --max_num_seqs $MAX_NUM_SEQS \
    --output_dir $OUTPUT_DIR \
    --max_output_length 16384 \
    $EXTRA_ARGS
