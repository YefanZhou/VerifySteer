#!/bin/bash
# Evaluate MLP probes trained on ProcessBench.
#
# Before running, set HIDDEN_STATES_DIR to the directory that contains your
# extracted hidden states, structured as:
#   ${HIDDEN_STATES_DIR}/<model>/<subset>/verify_processbench_careful_v1/hidden_states/
#     <subset>_hidden_states_<hidden_setup>.pt
#
# The probe weights are loaded from verify_mlp_probe_weights/ in this repo.

HIDDEN_STATES_DIR=${HIDDEN_STATES_DIR:-"$(dirname "$0")/state_for_predict_correctness"}
REPO_ROOT="$(dirname "$0")/.."
WEIGHTS_BASE="${REPO_ROOT}/verify_mlp_probe_weights/processbench"

for model_name in 'Qwen3-1.7B' 'Qwen3-8B'; do

  if [ "${model_name}" = "Qwen3-1.7B" ]; then
    LAYER_IDX=17
    HIDDEN_SETUP='end_of_prompt'
  else
    LAYER_IDX=23
    HIDDEN_SETUP='all_generated_avg'
  fi

  HYPERPARAM_STR="layer${LAYER_IDX}_hid1024_3layer_bs32_lr1e-05_wd1e-02_epoch300"
  WEIGHTS_ROOT="${WEIGHTS_BASE}/${model_name}"

  for subset in 'math' 'gsm8k' 'omnimath' 'olympiadbench'; do

    CONFIG_DIR="${WEIGHTS_ROOT}/verify_processbench_careful_v1_${HIDDEN_SETUP}/${HYPERPARAM_STR}"
    VAL_PATH="${HIDDEN_STATES_DIR}/${model_name}/${subset}/verify_processbench_careful_v1/hidden_states/${subset}_hidden_states_${HIDDEN_SETUP}.pt"

    echo "Evaluating: model=${model_name}  subset=${subset}  hidden_setup=${HIDDEN_SETUP}"
    python eval_mlp_probe.py \
      --val_hidden_states_path_lst "${VAL_PATH}" \
      --model_path "${CONFIG_DIR}/layer_${LAYER_IDX}_model.pt" \
      --output_dir "${CONFIG_DIR}" \
      --layer_idx ${LAYER_IDX} \
      --num_layers 3 \
      --dataset_name "${subset}" \
      --dropout 0.1 \
      --hidden_dim 1024 \
      --batch_size 32

  done

done
