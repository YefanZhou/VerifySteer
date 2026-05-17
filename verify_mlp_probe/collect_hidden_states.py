"""
Extract hidden states from a verifier model for probe training.

Supports ProcessBench, FARE, and Hard2Verify datasets with multiple prompt formats.
Mirrors the prompt setup used in steer_run_eval.py so extracted states correspond
to the same inputs seen during steering-vector evaluation.

Usage:
    python collect_hidden_states.py \\
        --model_name Qwen/Qwen3-8B \\
        --output_dir /path/to/output \\
        --config gsm8k \\
        --template-type verify \\
        --prompt-type processbench_careful_v1
"""

import argparse
import logging
import os
import random
import sys
import tqdm
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import decrypt_sample
from prompts import (
    processbench_earlystep_onepass,
    processbench_earlystep_onepass_careful_v1,
    h2v_error_id_system_prompt,
    h2v_error_id_user_prompt,
    PROMPT_PROCESS_SYSTEM_ERROR_ID,
    PROMPT_SINGLE
)



PROCESSBENCH_CONFIGS = frozenset({"gsm8k", "math", "olympiadbench", "omnimath"})

_HIDDEN_VARIANTS = ("end_of_prompt", "all_generated_avg")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from a verifier model for probe training."
    )
    parser.add_argument("--model_name", required=True,
                        help="HuggingFace model name or local path.")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for hidden-state outputs.")
    parser.add_argument(
        "--config", default="correct_seed42",
        help=(
            "Dataset config: a ProcessBench split (gsm8k / math / olympiadbench / omnimath), "
            "'h2v' for Hard2Verify, or a custom JSONL config name."
        ),
    )
    parser.add_argument(
        "--template-type", default="verify", choices=["verify"],
        help="Feed only the formatted prompt; hidden states are captured at end-of-prompt position.",
    )
    parser.add_argument(
        "--prompt-type", default="processbench",
        choices=[
            "processbench",
            "processbench_careful_v1",
            "h2v_error_id",
            "fare",
        ],
    )
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable chain-of-thought thinking tokens (Qwen3 only).")
    parser.add_argument("--base-path", default="/data/yefan",
                        help="Root path for checkpoint / dataset files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset subsampling (numina configs).")
    parser.add_argument(
        "--save-every", type=int, default=200,
        help="Flush hidden states to disk every N samples (chunked mode, used for large datasets).",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Run on the first 10 samples only.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path / logging
# ---------------------------------------------------------------------------

def build_base_path(args) -> str:
    model_tag = args.model_name.split("/")[-1]
    
    if args.enable_thinking:
        model_tag = f"{model_tag}_thinking"
        
    return (
        f"{args.output_dir}/{model_tag}/{args.config}"
        f"/{args.template_type}_{args.prompt_type}/hidden_states"
    )


def setup_logging(base_path: str, args) -> str:
    log_file = (
        f"{base_path}/construction_{args.config}"
        f"_{args.template_type}_{args.prompt_type}.log"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    root.addHandler(fh)

    # Use sys.__stdout__ to avoid circular writes after we redirect sys.stdout below.
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    class _LogWriter:
        def __init__(self, fn):
            self._fn = fn
        def write(self, msg):
            if msg.rstrip():
                self._fn(msg.rstrip())
        def flush(self):
            pass

    sys.stdout = _LogWriter(root.info)
    sys.stderr = _LogWriter(root.error)
    return log_file


# ---------------------------------------------------------------------------
# Template / prompt helpers  (mirrors steer_run_eval.py)
# ---------------------------------------------------------------------------

def select_template(prompt_type: str):
    mapping = {
        "processbench":           processbench_earlystep_onepass,
        "processbench_careful_v1": processbench_earlystep_onepass_careful_v1,
        "h2v_error_id":           (h2v_error_id_system_prompt, h2v_error_id_user_prompt),
        "fare":                (PROMPT_PROCESS_SYSTEM_ERROR_ID, PROMPT_SINGLE),
    }
    if prompt_type not in mapping:
        raise ValueError(f"Unsupported prompt_type: {prompt_type!r}")
    return mapping[prompt_type]


def get_label_key(config: str) -> str:
    return "human_labels_first_error_idx" if config == "h2v" else "label"


def get_unique_id(example: dict, config: str, idx: int) -> str:
    if config in PROCESSBENCH_CONFIGS:
        return example["id"]
    if config == "h2v":
        return example["unique_id"]
    return f"{config}_{idx}"


def _build_tagged_response(steps, prompt_type: str) -> str:
    """Format solution steps into a tagged string depending on prompt convention."""
    if prompt_type == "fare":
        return "\n".join(f"<step {i}> {step}</step {i}>" for i, step in enumerate(steps))
    # Default: paragraph tags used by ProcessBench and h2v_error_id prompts.
    return "".join(
        f"<paragraph_{i}>\n{step}\n</paragraph_{i}>\n\n" for i, step in enumerate(steps)
    ).strip()


def _build_messages(template, prompt_type: str, problem: str, tagged_response: str) -> list:
    if isinstance(template, tuple):
        if prompt_type.startswith("fare"):
            user = template[1].format(instruction=problem, response=tagged_response)
        else:
            user = template[1].format(problem=problem, tagged_response=tagged_response)
        return [{"role": "system", "content": template[0]}, {"role": "user", "content": user}]
    return [{"role": "user", "content": template.format(problem=problem, tagged_response=tagged_response)}]


def prepare_input_boxed(input_d, idx, template, prompt_type, tokenizer, enable_thinking):
    """Dataset.map-compatible function that adds a 'prompt' column."""
    if "question" in input_d and "problem" in input_d:
        raise ValueError("Both 'question' and 'problem' keys exist in input_d")
    if "model_response_by_step" in input_d and "steps" in input_d:
        raise ValueError("Both 'model_response_by_step' and 'steps' keys exist in input_d")

    problem = input_d.get("question") or input_d.get("problem")
    if problem is None:
        raise ValueError("Neither 'question' nor 'problem' found in input_d")
    steps = input_d.get("model_response_by_step") or input_d.get("steps")
    if steps is None:
        raise ValueError("Neither 'model_response_by_step' nor 'steps' found in input_d")

    messages = _build_messages(
        template, prompt_type, problem, _build_tagged_response(steps, prompt_type)
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    return {"prompt": prompt}


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(model, tokenizer, prompt: str, device: str = "cuda") -> dict:
    """
    Return per-layer hidden states at the end-of-prompt position and as an average
    over all prompt tokens.

    Returned keys (shapes are (num_layers, hidden_dim)):
        end_of_prompt_states     – hidden state at the last prompt token
        all_generated_avg_states – mean over all prompt tokens
        num_layers               – number of transformer layers
    """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    n_prompt = len(prompt_tokens)

    input_ids = torch.tensor([prompt_tokens]).to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    raw = outputs.hidden_states  # tuple: embedding + L layer tensors
    num_layers = len(raw) - 1
    all_layers = np.stack([raw[i + 1][0].float().cpu().numpy() for i in range(num_layers)])

    return {
        "end_of_prompt_states":     all_layers[:, n_prompt - 1, :],
        "all_generated_avg_states": all_layers.mean(axis=1) if n_prompt > 0 else None,
        "num_layers": num_layers,
    }


# ---------------------------------------------------------------------------
# Batch extraction helpers
# ---------------------------------------------------------------------------

def _new_store():
    return {v: {"hidden_states": [], "labels": [], "unique_ids": []} for v in _HIDDEN_VARIANTS}


def _accumulate(store, hidden_data, label, unique_id):
    """Append one sample's hidden states into the in-memory store."""
    eop = hidden_data["end_of_prompt_states"]
    aga = hidden_data["all_generated_avg_states"]

    if eop is not None:
        store["end_of_prompt"]["hidden_states"].append(eop)
        store["end_of_prompt"]["labels"].append(label)
        store["end_of_prompt"]["unique_ids"].append(unique_id)

    if aga is not None:
        store["all_generated_avg"]["hidden_states"].append(aga)
        store["all_generated_avg"]["labels"].append(label)
        store["all_generated_avg"]["unique_ids"].append(unique_id)


def _save_store(store, output_dir: str, config_name: str):
    """Stack and save each variant as a single .pt file."""
    for variant, data in store.items():
        if not data["hidden_states"]:
            print(f"Warning: no data collected for variant '{variant}', skipping.")
            continue
        hidden = np.stack(data["hidden_states"], axis=0)
        print(f"{variant}: shape={hidden.shape}, n={len(data['unique_ids'])}")
        path = os.path.join(output_dir, f"{config_name}_hidden_states_{variant}.pt")
        torch.save(
            {
                "hidden_states": torch.FloatTensor(hidden),
                "labels":        torch.LongTensor(data["labels"]),
                "unique_ids":    data["unique_ids"],
                "shape":         hidden.shape,
            },
            path,
        )
        print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Batch extraction: simple (in-memory) and chunked (for large datasets)
# ---------------------------------------------------------------------------

def batch_extract_and_save(model, tokenizer, dataset, output_dir: str, config_name: str, label_key: str):
    """Extract hidden states for all samples, accumulate in memory, then save."""
    print(f"Extracting hidden states for {len(dataset)} samples...")
    store = _new_store()

    for idx in tqdm.tqdm(range(len(dataset)), desc="Extracting"):
        example = dataset[idx]
        unique_id = get_unique_id(example, config_name, idx)
        try:
            hidden_data = extract_hidden_states(
                model, tokenizer,
                prompt=example["prompt"],
                device=model.device,
            )
            _accumulate(store, hidden_data, example[label_key], unique_id)
        except Exception as e:
            print(f"Error on sample {idx} ({unique_id}): {e}")

    _save_store(store, output_dir, config_name)


def batch_extract_and_save_chunked(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    config_name: str,
    label_key: str,
    save_every: int = 200,
):
    """
    Extract hidden states with periodic flushing to disk, then merge chunks.
    Preferred for large datasets (numina) where holding all states in RAM is infeasible.
    """
    print(f"Extracting hidden states for {len(dataset)} samples (flush every {save_every})...")
    store = _new_store()
    chunk_idx = 0

    def _flush(store, chunk_idx):
        for variant, data in store.items():
            if not data["hidden_states"]:
                continue
            hidden = np.stack(data["hidden_states"], axis=0)
            path = os.path.join(output_dir, f"{config_name}_{variant}_chunk{chunk_idx}.pt")
            torch.save(
                {
                    "hidden_states": hidden.astype(np.float16),
                    "labels":        list(data["labels"]),
                    "unique_ids":    list(data["unique_ids"]),
                },
                path,
            )
            data["hidden_states"].clear()
            data["labels"].clear()
            data["unique_ids"].clear()
        print(f"  Flushed chunk {chunk_idx}")

    for idx in tqdm.tqdm(range(len(dataset)), desc="Extracting"):
        example = dataset[idx]
        unique_id = get_unique_id(example, config_name, idx)
        try:
            hidden_data = extract_hidden_states(
                model, tokenizer,
                prompt=example["prompt"],
                device=model.device,
            )
            _accumulate(store, hidden_data, example[label_key], unique_id)
        except Exception as e:
            print(f"Error on sample {idx} ({unique_id}): {e}")

        if (idx + 1) % save_every == 0:
            _flush(store, chunk_idx)
            chunk_idx += 1

    _flush(store, chunk_idx)  # flush remainder

    print("Merging chunks...")
    for variant in _HIDDEN_VARIANTS:
        chunk_files = sorted(Path(output_dir).glob(f"{config_name}_{variant}_chunk*.pt"))
        if not chunk_files:
            continue
        all_hidden, all_labels, all_ids = [], [], []
        for cf in chunk_files:
            ck = torch.load(cf, weights_only=False)
            all_hidden.append(ck["hidden_states"])
            all_labels.extend(ck["labels"])
            all_ids.extend(ck["unique_ids"])
            os.remove(cf)
        hidden = np.concatenate(all_hidden, axis=0)
        path = os.path.join(output_dir, f"{config_name}_hidden_states_{variant}.pt")
        torch.save(
            {
                "hidden_states": torch.FloatTensor(hidden),
                "labels":        torch.LongTensor(all_labels),
                "unique_ids":    all_ids,
                "shape":         hidden.shape,
            },
            path,
        )
        print(f"  {variant}: {hidden.shape} -> {path}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(args):
    config = args.config
    checkpoint_dir = f"{args.base_path}/checkpoint/llm-verify-mech"

    if config in PROCESSBENCH_CONFIGS:
        return load_dataset("Qwen/ProcessBench", split=config)

    if config == "h2v":
        ds = load_dataset("Salesforce/Hard2Verify", split="test")
        return ds.map(decrypt_sample)

    if "numina" in config:
        ds = load_dataset(
            "json", split="train",
            data_files=f"{checkpoint_dir}/train/{config}.jsonl",
        )
        random.seed(args.seed)
        label_neg1 = ds.filter(lambda x: x["label"] == -1)
        label_pos  = ds.filter(lambda x: x["label"] > -1)
        print(f"label==-1: {len(label_neg1)}, label>-1: {len(label_pos)}")

        sample_size = min(len(label_neg1), len(label_pos))


        neg1_idx = random.sample(range(len(label_neg1)), min(sample_size, len(label_neg1)))
        pos_idx  = random.sample(range(len(label_pos)),  min(sample_size, len(label_pos)))
        ds = concatenate_datasets([label_neg1.select(neg1_idx), label_pos.select(pos_idx)])
        print(f"Dataset size after balancing: {len(ds)}")
        return ds

    # Fallback:
    else:
        raise ValueError(f"Unsupported config: {config}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    base_path = build_base_path(args)
    os.makedirs(base_path, exist_ok=True)
    log_file = setup_logging(base_path, args)

    print("=" * 80)
    print(f"Model:        {args.model_name}")
    print(f"Config:       {args.config}")
    print(f"Prompt type:  {args.prompt_type}  (template-type: {args.template_type})")
    print(f"Thinking:     {args.enable_thinking}")
    print(f"Output dir:   {base_path}")
    print(f"Log file:     {log_file}")
    print("=" * 80)

    template  = select_template(args.prompt_type)
    label_key = get_label_key(args.config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ds = load_eval_dataset(args)
    if args.debug:
        ds = ds.select(range(min(10, len(ds))))
        print(f"Debug mode: running on {len(ds)} samples")

    ds = ds.map(
        prepare_input_boxed,
        num_proc=1,
        with_indices=True,
        fn_kwargs={
            "template":        template,
            "prompt_type":     args.prompt_type,
            "tokenizer":       tokenizer,
            "enable_thinking": args.enable_thinking,
        },
    )

    output_dir = str(Path(base_path))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    if "numina" in args.config:
        batch_extract_and_save_chunked(
            model, tokenizer, ds, output_dir, args.config,
            label_key=label_key, save_every=args.save_every,
        )
    else:
        batch_extract_and_save(model, tokenizer, ds, output_dir, args.config, label_key=label_key)

    print(f"\nDone. Hidden states saved to: {output_dir}")


if __name__ == "__main__":
    main()
