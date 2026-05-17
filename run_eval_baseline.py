import argparse
import numpy as np
import os
import sys
import re
import json

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append('..')
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

def is_qwen3_model(model_path: str) -> bool:
    name = model_path.lower()
    return 'qwen3' in name


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
    """Format a single example into chat messages.

    Supports the ProcessBench schema (`problem`, `steps`) and the
    Hard2Verify schema (`question`, `model_response_by_step`), as well as
    the fare-style step formatting used by gpt-oss / FARE finetunes.
    """
    has_question = 'question' in input_d
    has_problem = 'problem' in input_d
    if has_question and has_problem:
        raise ValueError("Both 'question' and 'problem' keys exist in input_d")
    if has_question:
        problem = input_d['question']
    elif has_problem:
        problem = input_d['problem']
    else:
        raise ValueError(
            f"Problem not found in input_d for prompt type '{prompt_type}'"
        )

    has_mrbs = 'model_response_by_step' in input_d
    has_steps = 'steps' in input_d
    if has_mrbs and has_steps:
        raise ValueError(
            "Both 'model_response_by_step' and 'steps' keys exist in input_d"
        )
    if has_mrbs:
        steps = input_d['model_response_by_step']
    elif has_steps:
        steps = input_d['steps']
    else:
        raise ValueError(
            f"Steps not found in input_d for prompt type '{prompt_type}'"
        )
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
    """Return the field name that holds the ground-truth first-error index."""
    if config == 'h2v':
        return 'human_labels_first_error_idx'
    return 'label'


def build_sampling_params(args):
    """Build SamplingParams matching the source script's behavior.
    """
    common = dict(n=1, max_tokens=args.max_output_length)
    if args.seed is not None:
        common['seed'] = args.seed

    if is_fare_model(args.model_path):
        if args.greedy:
            print('temperature = 0.0, top_p = 1.0')
            return SamplingParams(temperature=0.0, top_p=1.0, **common)
  
        eff_temp = args.temperature if args.temperature is not None else 1.0
        print(f'temperature = {eff_temp}, top_p = {args.top_p}, top_k = {args.top_k}')
        return SamplingParams(
            temperature=eff_temp, top_p=args.top_p, top_k=args.top_k, **common
        )
        
    elif is_qwen3_model(args.model_path):
        if args.enable_thinking:
            eff_temp = args.temperature if args.temperature is not None else 0.6
            print(f'temperature = {eff_temp}, top_p = 0.95, top_k = 20')
            return SamplingParams(temperature=eff_temp, top_p=0.95, top_k=20, **common)

        if args.greedy:
            print('temperature = 0.0, top_p = 1.0')
            return SamplingParams(temperature=0.0, top_p=1.0, **common)

        print('temperature = 0.7, top_p = 0.8, top_k = 20')
        return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, **common)

    else:
        raise ValueError(f"Model {args.model_path} not supported")


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

def build_output_dir(args, config) -> str:
    """Compute the per-config output directory matching all source layouts."""
    base = f'baseline/{args.prompt_type}'
    run_str = build_run_str(args)

    if 'train' in config or 'numina' in config:
        hyperparam_str = f'{base}/{run_str}/{config}'
    else:
        hyperparam_str = f'{base}/{run_str}'

    thinking_dir = (
        'enable_thinking_True' if args.enable_thinking else 'enable_thinking_False'
    )
    return os.path.join(args.output_dir, args.model_name, thinking_dir, hyperparam_str)

def load_input_data(args, config: str):
    if config == 'h2v':
        ds = load_dataset('Salesforce/Hard2Verify', split='test')
        ds = ds.map(decrypt_sample)
        return ds
    if 'train' in config or 'numina' in config:
        return load_dataset('json', split='train', data_files=args.train_set_path)
    return load_dataset('Qwen/ProcessBench', split=config)


def parse_prediction(generated_critique, prompt_type: str):
    """Parse the prediction from a single generated string."""
    needs_judgment_parser = prompt_type.startswith('fare')

    if needs_judgment_parser:
        pred = parse_judgment_output(generated_critique, task='error_id')
        if pred == -2:
            pred = None
        return pred

    pred = extract_answer(generated_critique)
    try:
        return int(pred)
    except (TypeError, ValueError):
        return None

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', type=str, nargs='+', default=None,
        choices=[
            'h2v',
            'gsm8k', 'math', 'olympiadbench', 'omnimath',
            'train_v1_error', 
            'train_v1_correct',
            'train_v1_error_seed43', 
            'train_v1_correct_seed43',
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
    parser.add_argument('--enable_thinking', default=False, action='store_true')
    parser.add_argument('--max_num_seqs', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=32768)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=None,
                        help='')
    parser.add_argument(
        '--prompt-type', type=str, default='',
        choices=[
            'processbench',
            'processbench_careful_v1',
            'h2v_error_id',
            'fare',
        ],
    )
    parser.add_argument('--train_set_path', type=str, default='')
    parser.add_argument('--greedy', default=False, action='store_true')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling top_p')
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Top-k sampling')
    parser.add_argument('--enforce_eager', type=str, default='False',
                        choices=['True', 'False'],
                        help='Force eager execution in vLLM (string True/False).')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)

    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    tokenizer_src = args.tokenizer_path if args.tokenizer_path is not None else args.model_path
    toker = AutoTokenizer.from_pretrained(tokenizer_src)

    TEMPLATE = select_template(args.prompt_type)

    if args.gpu_memory_utilization is not None:
        gpu_mem = args.gpu_memory_utilization
    else:
        gpu_mem = 0.85

    llm_kwargs = dict(
        model=args.model_path,
        tokenizer=tokenizer_src,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_steer_vector=False,
        enforce_eager=(args.enforce_eager == 'True'),
        enable_chunked_prefill=False,
        swap_space=16,
        max_num_seqs=args.max_num_seqs,
    )
    if args.max_model_len is not None:
        print(f'max_model_len = {args.max_model_len}')
        llm_kwargs['max_model_len'] = args.max_model_len
        
    llm = LLM(**llm_kwargs)

    sampling_params = build_sampling_params(args)
    
    first_iteration = True
    for config in args.configs:
        output_dir = build_output_dir(args, config)
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)

        if first_iteration:
            save_args(args, output_dir)
            first_iteration = False

        input_data = load_input_data(args, config)


        if args.debug:
            n_debug = 20
            n_debug = min(n_debug, len(input_data))
            input_data = input_data.shuffle(seed=42).select(range(n_debug))


        text_prompts = [
            apply_chat_template(
                toker,
                prepare_input_boxed(TEMPLATE, e, args.prompt_type),
                args.enable_thinking,
            )
            for e in input_data
        ]

        if len(text_prompts) > 0:
            print(text_prompts[0])

        generations = llm.generate(prompts=text_prompts, sampling_params=sampling_params)

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

        acc1 = (
            np.mean([e['match'] for e in error_data]) * 100
            if len(error_data) > 0 else 0.0
        )
        acc2 = (
            np.mean([e['match'] for e in correct_data]) * 100
            if len(correct_data) > 0 else 0.0
        )
        f1 = 2 * acc1 * acc2 / (acc1 + acc2) if (acc1 + acc2) > 0 else 0.0

        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
        print(
            f'{config} samples - error: {len(error_data)}, correct: {len(correct_data)}'
        )

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
