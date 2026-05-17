"""
"""

import argparse
import numpy as np
import os
import re
import sys
import json
import torch

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import decrypt_sample, parse_judgment_output
from prompts import (
    processbench_earlystep_onepass,
    processbench_earlystep_onepass_careful_v1,
    h2v_error_id_system_prompt,
    h2v_error_id_user_prompt,
    PROMPT_PROCESS_SYSTEM_ERROR_ID,
    PROMPT_SINGLE
)

def is_fare_model(model_path: str) -> bool:
    name = model_path.lower()
    return 'fare' in name


def save_args(args, output_dir):
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
            args_dict[key] = str(value)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)


def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def apply_chat_template(toker, messages, enable_thinking=False):
    return toker.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def prepare_input_boxed(template, input_d, prompt_type):
    has_question = 'question' in input_d
    has_problem = 'problem' in input_d
    if has_question and has_problem:
        raise ValueError("Both 'question' and 'problem' keys exist in input_d")
    if has_question:
        problem = input_d['question']
    elif has_problem:
        problem = input_d['problem']
    else:
        raise ValueError(f"Problem not found in input_d for prompt type '{prompt_type}'")

    has_mrbs = 'model_response_by_step' in input_d
    has_steps = 'steps' in input_d
    if has_mrbs and has_steps:
        raise ValueError("Both 'model_response_by_step' and 'steps' keys exist in input_d")
    if has_mrbs:
        steps = input_d['model_response_by_step']
    elif has_steps:
        steps = input_d['steps']
    else:
        raise ValueError(f"Steps not found in input_d for prompt type '{prompt_type}'")

    if prompt_type == 'fare':
        parts = [f"<step {sdx}> {step}</step {sdx}>" for sdx, step in enumerate(steps)]
        tagged_response = "\n".join(parts)
    else:
        tagged_response = ''
        for sdx, step in enumerate(steps):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()

    if isinstance(template, tuple):
        if prompt_type.startswith('fare'):
            prompt = template[1].format(instruction=problem, response=tagged_response)
        else:
            prompt = template[1].format(problem=problem, tagged_response=tagged_response)
        messages = [
            {'role': 'system', 'content': template[0]},
            {'role': 'user', 'content': prompt},
        ]
    else:
        prompt = template.format(problem=problem, tagged_response=tagged_response)
        messages = [{'role': 'user', 'content': prompt}]

    return messages


def get_label_key(config: str) -> str:
    if config == 'h2v':
        return 'human_labels_first_error_idx'
    return 'label'


def select_template(prompt_type: str):
    if prompt_type == 'processbench':
        return processbench_earlystep_onepass
    if prompt_type == 'processbench_careful_v1':
        return processbench_earlystep_onepass_careful_v1
    if prompt_type == 'h2v_error_id':
        return (h2v_error_id_system_prompt, h2v_error_id_user_prompt)
    if prompt_type == 'fare':
        return (PROMPT_PROCESS_SYSTEM_ERROR_ID, PROMPT_SINGLE)

    raise ValueError(f"Prompt type {prompt_type} not supported")


def build_matching_token_ids(toker):
    target_suffix = "ĊĊ"
    vocab = toker.get_vocab()
    return [
        token_id
        for token, token_id in vocab.items()
        if isinstance(token, str) and token.endswith(target_suffix)
    ]


def build_sv_request(args, matching_tokens_ids):

    return SteerVectorRequest(
        steer_vector_name="control",
        steer_vector_int_id=1,
        steer_vector_local_path=args.control_vector_path,
        generate_trigger_tokens=matching_tokens_ids,
        algorithm=args.steering_mode,
        scale=args.vector_scale,
        target_layers=args.layer_range,
        beta=args.beta,
    )
    

def build_vector_name(args) -> str:
    name_vector_path = args.control_vector_path 
    stem = os.path.basename(name_vector_path).replace(".gguf", "")
    dataset_dir = os.path.basename(os.path.dirname(name_vector_path))
    return f"{dataset_dir}/{stem}"


def build_layer_str(layer_range) -> str:
    if len(layer_range) == 1:
        return f"L{layer_range[0]}"
    if len(layer_range) == 2:
        return f"L{min(layer_range)}-{max(layer_range)}"
    return "L" + "-".join(str(x) for x in layer_range)


def build_run_str(args) -> str:
    """Compose the per-run subdirectory string.
    """
    if args.seed is None:
        run_str = f'run_{args.run_num}'
    else:
        run_str = f'seed_{args.seed}'
    if args.greedy:
        run_str += '_greedy'
    else:
        if args.temperature is not None:
            run_str += f'_temperature_{args.temperature}'
        if args.top_p != 1.0:
            run_str += f'_top_p_{args.top_p}'
        if args.top_k != -1:
            run_str += f'_top_k_{args.top_k}'

    run_str += f'_maxtokens_{args.max_output_length}'

    if is_fare_model(args.model_path):
        run_str += f'_enforce_eager_{args.enforce_eager}'

    return run_str


def build_output_dir(args, config: str) -> str:
    layer_str = build_layer_str(args.layer_range)
    steering_mode = args.steering_mode

    scale_str = (f'scale_{args.vector_scale}_{layer_str}_beta{args.beta}')
   

    run_str = build_run_str(args)

    if 'train' in config or 'numina' in config:
        hyperparam_str = f'{scale_str}/{args.prompt_type}/{run_str}/{config}'
    else:
        hyperparam_str = f'{scale_str}/{args.prompt_type}/{run_str}'


    vector_name = build_vector_name(args)
    nested_folder = os.path.join(steering_mode, vector_name)
    thinking_dir = (
        'enable_thinking_True' if args.enable_thinking else 'enable_thinking_False'
    )

    return os.path.join(
        args.output_dir, args.model_name, nested_folder, thinking_dir, hyperparam_str
    )


def build_sampling_params(args):
    common = dict(n=1, max_tokens=args.max_output_length)
    if args.seed is not None:
        common['seed'] = args.seed

    if args.greedy:
        print('temperature = 0.0, top_p = 1.0')
        return SamplingParams(temperature=0.0, top_p=1.0, **common)

    if args.enable_thinking:
        temp = args.temperature if args.temperature is not None else 0.6
        print(f'temperature = {temp}, top_p = 0.95, top_k = 20')
        return SamplingParams(temperature=temp, top_p=0.95, top_k=20, **common)

    temp = args.temperature if args.temperature is not None else 1.0
    print(f'temperature = {temp}, top_p = {args.top_p}, top_k = {args.top_k}')
    return SamplingParams(temperature=temp, top_p=args.top_p, top_k=args.top_k, **common)


def load_input_data(args, config: str):
    if config == 'h2v':
        ds = load_dataset('Salesforce/Hard2Verify', split='test')
        return ds.map(decrypt_sample)
    if 'train' in config or 'numina' in config:
        return load_dataset('json', split='train', data_files=args.train_set_path)
    return load_dataset('Qwen/ProcessBench', split=config)


def parse_prediction(generated_critique: str, prompt_type: str):
    if prompt_type.startswith('fare'):
        pred = parse_judgment_output(generated_critique, task='error_id')
        return None if pred == -2 else pred
    pred = extract_answer(generated_critique)
    try:
        return int(pred)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', type=str, nargs='+', default=None,
        choices=[
            'gsm8k', 'math', 'olympiadbench', 'omnimath', 'h2v',
            'train_v1_error_seed43', 'train_v1_correct_seed43',
            'train_v1_error', 'train_v1_correct',
            'numina_olympiad_mwp_10.0pct_seed42',
            'numina_math_train_metamath_mwp_40.0pct_seed42',
            'numina_imo_putnam_proof',
        ],
    )
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--run_num', type=int, default=None)
    parser.add_argument(
        '--layer_range', nargs='+', type=int,
        default=[],
    )
    parser.add_argument('--vector_scale', type=float, default=1.0)
    parser.add_argument('--control_vector_path', type=str, required=True)
    parser.add_argument('--enable_thinking', default=False, action='store_true')
    parser.add_argument('--max_num_seqs', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=32768)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument(
        '--prompt-type', type=str, default='processbench',
        choices=[
            'processbench',
            'processbench_careful_v1',
            'h2v_error_id',
            'fare',
        ],
    )
    parser.add_argument(
        '--steering_mode', type=str, default='gated_norm_direct_cosine_mask',
        choices=[
            'gated_norm_direct_cosine_mask',
        ],
    )
    parser.add_argument('--train_set_path', type=str, default='')
    parser.add_argument('--greedy', default=False, action='store_true')
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--enforce_eager', default='True', choices=['True', 'False'], type=str)

    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)
    
    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    tokenizer_src = args.tokenizer_path if args.tokenizer_path is not None else args.model_path
    toker = AutoTokenizer.from_pretrained(tokenizer_src)

    TEMPLATE = select_template(args.prompt_type)

    llm_kwargs = dict(
        model=args.model_path,
        tokenizer=tokenizer_src,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_steer_vector=True,
        enforce_eager=(args.enforce_eager == 'True'),
        enable_chunked_prefill=False,
        swap_space=16,
        max_num_seqs=args.max_num_seqs,
    )
    if args.max_model_len is not None:
        llm_kwargs['max_model_len'] = args.max_model_len

    llm = LLM(**llm_kwargs)

    sampling_params = build_sampling_params(args)
    matching_tokens_ids = build_matching_token_ids(toker)
    sv_request = build_sv_request(args, matching_tokens_ids)


    for config in args.configs:
        output_dir = build_output_dir(args, config)
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)

        save_args(args, output_dir)
       
        input_data = load_input_data(args, config)


        if args.debug:
            n = min(20, len(input_data))
            input_data = input_data.shuffle(seed=42).select(range(n))


        text_prompts = [
                apply_chat_template(
                    toker,
                    prepare_input_boxed(TEMPLATE, e, args.prompt_type),
                    args.enable_thinking,
                )
                for e in input_data
            ]

        if text_prompts:
            print(text_prompts[0])

        generations = llm.generate(
            prompts=text_prompts,
            sampling_params=sampling_params,
            steer_vector_request=sv_request,
        )

        label_key = get_label_key(config)
        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()
            generated_critique = generations[i].outputs[0].text
            pred = parse_prediction(generated_critique, args.prompt_type)
            d['generated_critique'] = generated_critique
            d['prediction'] = pred
            d['match'] = (pred == d[label_key])
            res_data.append(d)

        error_data = [e for e in res_data if e[label_key] != -1]
        correct_data = [e for e in res_data if e[label_key] == -1]

        suffix = '_debug' if args.debug else ''
        with open(os.path.join(output_dir, f'{config}_error{suffix}.jsonl'), 'w') as f:
            for e in error_data:
                f.write(json.dumps(e) + '\n')
        with open(os.path.join(output_dir, f'{config}_correct{suffix}.jsonl'), 'w') as f:
            for e in correct_data:
                f.write(json.dumps(e) + '\n')

        acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0.0
        acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0.0
        f1 = 2 * acc1 * acc2 / (acc1 + acc2) if (acc1 + acc2) > 0 else 0.0

        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
        print(f'{config} samples - error: {len(error_data)}, correct: {len(correct_data)}')

        metrics = {
            'config': config,
            'error_acc': acc1,
            'correct_acc': acc2,
            'f1': f1,
            'num_error_samples': len(error_data),
            'num_correct_samples': len(correct_data),
        }
        with open(os.path.join(output_dir, f'{config}_metrics{suffix}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
