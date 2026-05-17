import re,json
from sklearn.metrics import recall_score


# For dataset decryption purposes!
def _derive_keystream(canary: str, length: int) -> bytes:
    import hashlib
    from math import ceil
    digest = hashlib.sha256(canary.encode("utf-8")).digest()
    if length <= len(digest):
        return digest[:length]
    repeats = ceil(length / len(digest))
    return (digest * repeats)[:length]

def _xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    if len(data) != len(key_stream):
        raise ValueError("Data and keystream must be the same length for XOR.")
    return bytes([a ^ b for a, b in zip(data, key_stream)])

def _deserialize_field(text: str):
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return text

def decrypt_str(input_str, canary):
    import base64
    if input_str == "":
        return ""
    ct = base64.b64decode(input_str)
    ks = _derive_keystream(canary, len(ct))
    pt = _xor_bytes(ct, ks)
    text = pt.decode("utf-8")
    return _deserialize_field(text)


def decrypt_sample(example):
    if "canary" not in example:
        raise ValueError("Missing canary field `canary`.")
    canary = example["canary"]
    if not isinstance(canary, str):
        raise ValueError(f"Canary should be a string.")

    target_fields = ["question", "model_response_by_step", "human_labels", "human_labels_first_error_idx"]
    for k, v in example.items():
        if k in target_fields and isinstance(v, str):
            try:
                example[k] = decrypt_str(v, canary)
            except Exception:
                example[k] = v

    return example


OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3", "gpt-4.1"]
TOGETHER_MODELS = ["moonshotai/Kimi-K2-Instruct-0905", "zai-org/GLM-4.5-Air-FP8", "deepseek-ai/DeepSeek-R1"]

def get_sampling_params(args):
    is_thinking = args.enable_thinking
    if "qwen3" in args.model_name.lower() and is_thinking:
        return dict(temperature=0.6, top_p=0.95, top_k=20)
    elif "qwen3" in args.model_name.lower() and not is_thinking:
        return dict(temperature=0.7, top_p=0.8, top_k=20)
    elif (args.model_name in OPENAI_MODELS and args.model_name != "gpt-4.1") or 'gpt-oss' in args.model_name.lower():
        return dict(temperature=1.0, top_p=1.0, top_k=-1)
    elif args.model_name == "gpt-4.1":
        return dict(temperature=0.0, top_p=1.0, top_k=-1)
    elif 'kimi' in args.model_name.lower():
        return dict(temperature=0.6, top_p=1.0, top_k=-1)
    elif any(m in args.model_name.lower() for m in ["deepseek", "glm"]):
        return dict(temperature=0.6, top_p=0.95, top_k=-1)
    elif 'bytedance' in args.model_name.lower():
        return dict(temperature=1.1, top_p=0.95, top_k=-1)
    # default greedy
    else:
        return dict(temperature=0.0, top_p=1.0, top_k=-1)


step_level_system_prompt = (
    "You are a strict, reliable math grader."
    "Return the evaluation using the exact format requested by the user."
    "Provide brief justifications (1–2 sentences per step); do not reveal chain-of-thought or scratch work."
)

step_level_user_prompt = (
    "The following is a math problem and a solution (split into steps, enclosed with tags and indexed from 0):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Solution]\n\n"
    "{steps}\n\n"
    """ 
    Your task is to review and critique the solution step-by-step.
    For each step, determine if it is correct or incorrect.
    - A correct step is one where all of the content is correct, and is logically consistent with all previous steps and information given in the problem.
    - An incorrect step is one where the content is incorrect, or is not logically consistent with all previous steps and information given in the problem, or is based on an error in a previous step.

    Important: Any step that contains or is based on an error is considered incorrect. That is, if the error is carried forward from a previous step or is based on an error in the previous step, consider the step incorrect.

    Provide reasoning for your correctness determinations. Your final verdict should be a comma-separated list of yes and no's, where each yes or no corresponds to a step's correctness, with yes meaning correct and no meaning incorrect. 

    Please use the following format to return your answer:
    Reasoning: <your reasoning for each step>
    Verdict: <your comma-separated list of yes and no's>

    Do not use any other formatting, including markdown, bold text, code blocks, or any other formatting. If your formatting is incorrect, your evaluation will be affected.
    """
)

step_level_user_retry_prompt = (
    "The following is a math problem and a solution (split into steps, enclosed with tags and indexed from 0):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Solution]\n\n"
    "{steps}\n\n"
    """ 
    Your task is to review and critique the solution step-by-step.
    For each step, determine if it is correct or incorrect.
    - A correct step is one where all of the content is correct, and is logically consistent with all previous steps and information given in the problem.
    - An incorrect step is one where the content is incorrect, or is not logically consistent with all previous steps and information given in the problem, or is based on an error in a previous step.

    Important: Any step that contains or is based on an error is considered incorrect. That is, if the error is carried forward from a previous step or is based on an error in the previous step, consider the step incorrect.

    Provide reasoning for your correctness determinations. Your final verdict should be a comma-separated list of yes and no's, where each yes or no corresponds to a step's correctness, with yes meaning correct and no meaning incorrect. 
    The number of yes and no's should be equal to the number of steps in the solution, which is {num_steps} steps.

    Please use the following format to return your answer:
    Reasoning: <your reasoning for each step>
    Verdict: <your comma-separated list of yes and no's>

    Do not use any other formatting, including markdown, bold text, code blocks, or any other formatting. If your formatting is incorrect, your evaluation will be affected.
    """.strip()
)

error_id_system_prompt = """
You are a strict, reliable math grader performing an Error identification task.
Your task is to identify the first incorrect step in a mathematical solution.
Return your answer using the exact format requested by the user.
""".strip()

error_id_user_prompt = (
    "The following is a math problem and a solution (split into steps, enclosed with tags and indexed from 0):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Solution]\n\n"
    "{steps}\n\n"
    """ 
    Your task is to identify the first incorrect step in the solution.

    Instructions:
    - Review each step carefully for mathematical correctness and logical consistency
    - A step is incorrect if it contains mathematical errors, logical inconsistencies, or is based on errors from previous steps
    - Find the FIRST step that is incorrect (0-indexed)
    - If ALL steps are correct, return -1

    Provide your reasoning and then give your final answer as a single number in the specified format.

    Please use the following format to return your answer:
    Reasoning: <your detailed reasoning explaining which steps are correct/incorrect and why>
    Verdict: <the step number of the first incorrect step or -1 if all steps are correct>

    Examples:
    - If step 0 is the first incorrect step: 0
    - If step 3 is the first incorrect step: 3
    - If all steps are correct: -1

    Do not use any other formatting, including markdown, bold text, code blocks, or any other formatting. If your formatting is incorrect, your evaluation will be affected.
    """.strip()
)



def load_prompt(args):
    if args.task_type == "step_level":
        return step_level_system_prompt, step_level_user_prompt, step_level_user_retry_prompt
    elif args.task_type == "error_id":
        return error_id_system_prompt, error_id_user_prompt, ""
    else:
        raise ValueError(f"Task type {args.task_type} not supported")


def parse_judgment_output(judgment_output, task="step_level"):
    if judgment_output == "" or judgment_output is None:
        return [-2] if task == "step_level" else -2
    # Normalize newlines
    t = judgment_output.replace("\r\n", "\n").replace("\r", "\n")
    # Extract the Verdict line
    verdict_matches = re.findall(r"\bVerdict:\s*(.+)", t, flags=re.IGNORECASE)
    verdict = verdict_matches[-1].strip() if verdict_matches else ""

    if task == "step_level":
        verdict_split = verdict.split(',')
        preds = []
        for v in verdict_split:
            if "yes" in v.lower():
                preds.append(1)
            elif "no" in v.lower():
                preds.append(0)
            else:
                preds.append(-2)
        return preds
    elif task == "error_id":
        if verdict.strip() == "-1":
            return -1
        elif verdict.strip().isdigit():
            return int(verdict)
        else:
            return -2
    else:
        raise ValueError(f"Task type {task} not supported")

def calculate_metrics(predictions, labels):
    # tpr, tnr
    try:
        tpr = recall_score(labels, predictions)
        tnr = recall_score(labels, predictions, pos_label=0)
        balanced_acc = (tpr + tnr) / 2
        balanced_f1_score = 2 * (tpr * tnr) / (tpr + tnr)
    except Exception as e:
        print(labels)
        print(predictions)
        raise e
        
    return {
        "tpr": round(100*tpr, 2),
        "tnr": round(100*tnr, 2),
        "balanced_acc": round(100*balanced_acc, 2),
        "balanced_f1_score": round(100*balanced_f1_score, 2)
    }

def calculate_error_id_metrics(predictions, labels):
    correct_idx = [i for i in range(len(labels)) if labels[i] == -1]
    incorrect_idx = [i for i in range(len(labels)) if labels[i] != -1]
    tpr = 0
    for i in correct_idx:
        if predictions[i] == -1:
            tpr += 1
    tpr /= len(correct_idx)
    tnr = 0
    for i in incorrect_idx:
        if predictions[i] == labels[i]:
            tnr += 1
    tnr /= len(incorrect_idx)
    balanced_acc = (tpr + tnr) / 2
    balanced_f1_score = 2 * (tpr * tnr) / (tpr + tnr) if (tpr + tnr) != 0 else 0
    return {
        "tpr": round(100*tpr, 2),
        "tnr": round(100*tnr, 2),
        "balanced_acc": round(100*balanced_acc, 2),
        "balanced_f1_score": round(100*balanced_f1_score, 2)
    }


def grade_step_level_responses(dataset, args):
    all_step_predictions, all_step_labels = [], []
    all_solution_predictions, all_solution_labels = [], []
    parsed_preds_with_fixes_by_unique_id = {}
    for d in dataset:
        # parse judgment output
        parsed_eval_output = d['parsed_eval_output']
        labels = d["human_labels"]

        all_step_labels.extend(labels)
        if any(l == 0 for l in labels):
            all_solution_labels.append(0)
        else:
            all_solution_labels.append(1)

        # if any -2's appear in parsed_eval_output, we consider that step wrong
        idx_to_replace = []
        for i, lp in enumerate(parsed_eval_output):
            if i >= len(labels):
                break
            if lp == -2:
                replace_val = 0 if labels[i] == 1 else 1
                idx_to_replace.append((i, replace_val))
        if idx_to_replace != []:
            for i, lp in idx_to_replace:
                parsed_eval_output[i] = lp

        # We need to reconile mismatches in model predicted labels and number of steps when grading. We employ the following strategy:
        # - If the number of labels is equal to number of steps, grade as is
        # - If the number of labels is less than number of steps:
        #   - For step level, we add dummy labels which get counted as wrong. We add these to the end of the response, i.e., we assume that model has outputted the first N < N_true steps
        #   - For solution level, we use the existing labels to compute solution level correctness, without "filling in" missing labels
        # - If the number of labels is greater than number of steps:
        #   - For step level, we truncate the response to the number of steps
        #   - For solution level, we use the step-level labels after truncation to determine solution correctness
        solution_level_prediction = -2
        if len(parsed_eval_output) != len(d["model_response_by_step"]):
            if len(parsed_eval_output) < len(d["model_response_by_step"]):
                if any(lp == 0 for lp in parsed_eval_output):
                    solution_level_prediction = 0
                else:
                    solution_level_prediction = 1
                # Add dummy labels
                wrong_labels = [0 if l == 1 else 1 for l in labels][len(parsed_eval_output):]
                parsed_eval_output = parsed_eval_output + wrong_labels
            else:
                # truncate
                parsed_eval_output = parsed_eval_output[:len(d["model_response_by_step"])]
                if any(lp == 0 for lp in parsed_eval_output):
                    solution_level_prediction = 0
                else:
                    solution_level_prediction = 1
        else:
            if any(lp == 0 for lp in parsed_eval_output):
                solution_level_prediction = 0
            else:
                solution_level_prediction = 1

        parsed_preds_with_fixes_by_unique_id[d['unique_id']] = {
            "step_level_with_fixes": parsed_eval_output,
            "solution_level_with_fixes": solution_level_prediction
        }
        all_step_predictions.extend(parsed_eval_output)
        all_solution_predictions.append(solution_level_prediction)

    step_level_metrics = calculate_metrics(all_step_predictions, all_step_labels)
    solution_level_metrics = calculate_metrics(all_solution_predictions, all_solution_labels)

    def update_dataset_with_parsed_preds_with_fixes(example):
        example["parsed_preds"] = parsed_preds_with_fixes_by_unique_id[example['unique_id']]
        return example
    dataset = dataset.map(update_dataset_with_parsed_preds_with_fixes)

    # merge dicts:
    all_metrics = {
        "step_level": step_level_metrics,
        "solution_level": solution_level_metrics
    }
    
    return all_metrics, dataset

def grade_error_id_responses(dataset, args):
    all_error_id_predictions, all_error_id_labels = [], []
    for d in dataset:
        first_error_idx = d["human_labels_first_error_idx"]
        all_error_id_labels.append(first_error_idx)
        parsed_eval_output = d['parsed_eval_output']
        all_error_id_predictions.append(parsed_eval_output)

    error_id_metrics = calculate_error_id_metrics(all_error_id_predictions, all_error_id_labels)
    return error_id_metrics, dataset

def grade_responses(dataset, args):
    if args.task_type == "step_level":
        return grade_step_level_responses(dataset, args)
    elif args.task_type == "error_id":
        return grade_error_id_responses(dataset, args)
    else:
        raise ValueError(f"Task type {args.task_type} not supported")
        

# For dataset decryption purposes!
def _derive_keystream(canary: str, length: int) -> bytes:
    import hashlib
    from math import ceil
    digest = hashlib.sha256(canary.encode("utf-8")).digest()
    if length <= len(digest):
        return digest[:length]
    repeats = ceil(length / len(digest))
    return (digest * repeats)[:length]

def _xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    if len(data) != len(key_stream):
        raise ValueError("Data and keystream must be the same length for XOR.")
    return bytes([a ^ b for a, b in zip(data, key_stream)])

def _deserialize_field(text: str):
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return text

def decrypt_str(input_str, canary):
    import base64
    if input_str == "":
        return ""
    ct = base64.b64decode(input_str)
    ks = _derive_keystream(canary, len(ct))
    pt = _xor_bytes(ct, ks)
    text = pt.decode("utf-8")
    return _deserialize_field(text)

def decrypt_sample(example):
    if "canary" not in example:
        raise ValueError("Missing canary field `canary`.")
    canary = example["canary"]
    if not isinstance(canary, str):
        raise ValueError(f"Canary should be a string.")

    target_fields = ["question", "model_response_by_step", "human_labels", "human_labels_first_error_idx"]
    for k, v in example.items():
        if k in target_fields and isinstance(v, str):
            try:
                example[k] = decrypt_str(v, canary)
            except Exception:
                example[k] = v

    return example
       
       
       
       
def parse_judgment_output(judgment_output, task="step_level"):
    if judgment_output == "" or judgment_output is None:
        return [-2] if task == "step_level" else -2
    # Normalize newlines
    t = judgment_output.replace("\r\n", "\n").replace("\r", "\n")
    # Extract the Verdict line
    verdict_matches = re.findall(r"\bVerdict:\s*(.+)", t, flags=re.IGNORECASE)
    verdict = verdict_matches[-1].strip() if verdict_matches else ""

    if task == "step_level":
        verdict_split = verdict.split(',')
        preds = []
        for v in verdict_split:
            if "yes" in v.lower():
                preds.append(1)
            elif "no" in v.lower():
                preds.append(0)
            else:
                preds.append(-2)
        return preds
    elif task == "error_id":
        if verdict.strip() == "-1":
            return -1
        elif verdict.strip().isdigit():
            return int(verdict)
        else:
            return -2
    else:
        raise ValueError(f"Task type {task} not supported")