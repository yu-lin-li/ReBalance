import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from re import split as rsplit

# NOTE: keep your custom Qwen import path if you have a patched model
# e.g., your custom class that supports set_steering_flag/start_new_round
from modeling_utils.modeling_openpangu_dynamic import PanguEmbeddedForCausalLM  # ! change here for confidence only

# ---- silence all warnings/logging (place at very top) ----
import os, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"            # Inherit to child/subprocesses
os.environ["TRANSFORMERS_VERBOSITY"] = "error"     # Reduce HF transformers logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"     # Disable tokenizer parallelism warnings
warnings.filterwarnings("ignore")                  # Silence all Python warnings

# Mute common library logs/warnings
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

try:
    import datasets
    datasets.utils.logging.set_verbosity_error()
except Exception:
    pass

# Also silence NumPy numerical warnings (overflow/invalid operations)
try:
    import numpy as np
    np.seterr(all="ignore")
except Exception:
    pass

# If warnings still go through logging, disable logging completely (use with caution; this hides error logs).
# logging.disable(logging.CRITICAL)
# ---- end silence ----


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl_atomic(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(tmp, file_path)


def evaluate_and_save(args, combined_file):
    from utils.data_loader import load_data
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    from math import comb

    # --------- helpers ---------
    def _extract_first_text(gen):
        """Compatibly handle various structures; use the first text for length stats, while scoring all candidates for correctness."""
        if isinstance(gen, str):
            return gen
        if isinstance(gen, dict):
            for key in ("text", "content", "generated_response", "generated_text", "output", "message", "response"):
                v = gen.get(key)
                if isinstance(v, str):
                    return v
        if isinstance(gen, list) and gen:
            for item in gen:
                t = _extract_first_text(item)
                if t:
                    return t
        return ""

    def _extract_all_texts(gens):
        out = []
        for g in gens:
            if isinstance(g, str):
                out.append(g)
            elif isinstance(g, dict):
                for key in ("text", "content", "generated_response", "generated_text", "output", "message", "response"):
                    v = g.get(key)
                    if isinstance(v, str):
                        out.append(v); break
            elif isinstance(g, list) and g:
                # Use the first string from nested items
                t = _extract_first_text(g)
                if t:
                    out.append(t)
        return out
    
    # --------- load ---------
    outputs = read_jsonl(combined_file)
    outputs_by_idx = {o.get("idx", i): o for i, o in enumerate(outputs)}
    examples = load_data(args.dataset, args.split, args.dataset_dir)

    # --------- correctness & pass@k ---------
    total = len(examples)
    correct_cnt = 0
    pass_at_k_vals = []
    wrong_ids = []
    for i in tqdm(range(total), desc="Evaluating", leave=False):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.dataset)
        out = outputs_by_idx.get(i)
        if not out:
            wrong_ids.append(d.get("id", i))
            continue
        texts = _extract_all_texts(out.get("generated_responses", []))
        if not texts:
            wrong_ids.append(d.get("id", i))
            continue
        gen_answers = [extract_answer(t, args.dataset) for t in texts]
        is_correct_list = [check_is_correct(a, gt_ans) for a in gen_answers]
        if any(is_correct_list):
            correct_cnt += 1
        else:
            wrong_ids.append(d.get("id", i))
        if len(is_correct_list) > 1:
            c = sum(is_correct_list)
            n = len(is_correct_list)
            if c > 0:
                if n - c < args.k:
                    val = 1.0
                else:
                    val = 1.0 - (comb(n - c, args.k) / comb(n, args.k))
                pass_at_k_vals.append(val)
            else:
                pass_at_k_vals.append(0.0)

    acc = correct_cnt / total if total else 0.0
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "generated": len(outputs),
        "correct": correct_cnt,
        "accuracy": acc,
        "k": args.k
    }
    if pass_at_k_vals:
        metrics[f"pass@{args.k}"] = sum(pass_at_k_vals) / len(pass_at_k_vals)
    else:
        metrics[f"pass@{args.k}"] = acc  # Degenerates to accuracy when only one sample exists.

    # Add completeness check metadata.
    is_complete = len(outputs) == total
    metrics["is_complete"] = is_complete
    if not is_complete:
        metrics["missing_count"] = total - len(outputs)
    
    # --------- token length stats ---------
    # Compute stats based on outputs
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )

    test_num = len(outputs)
    resp_word_counts = []
    full_token_counts = []
    think_token_counts = []
    think_found = 0
    fallback_full = 0

    for data in outputs:
        gens = data.get("generated_responses", [])
        text = _extract_first_text(gens) if gens else ""
        # 1) Word count
        resp_word_counts.append(len(text.split()) if text else 0)
        # 2) Full token count
        if text:
            full_tokens_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        else:
            full_tokens_len = 0
        full_token_counts.append(full_tokens_len)
        # 3) Think-segment token count: truncate at [unused17], otherwise use full text
        lower = text.lower() if text else ""
        idx = lower.find("[unused17]")  # Pangu thought-segment boundary
        if idx != -1:
            think_text = text[:idx]
            think_found += 1
        else:
            think_text = text
            if text:
                fallback_full += 1
        if think_text:
            think_tokens_len = len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
        else:
            think_tokens_len = 0
        think_token_counts.append(think_tokens_len)

    avg_resp_words = (sum(resp_word_counts) / test_num) if test_num else 0.0
    avg_full_tokens = (sum(full_token_counts) / test_num) if test_num else 0.0
    avg_think_tokens = (sum(think_token_counts) / test_num) if test_num else 0.0

    metrics["token_stats"] = {
        "samples": test_num,
        "avg_word_count": avg_resp_words,
        "avg_full_token_count": avg_full_tokens,
        # "avg_think_token_count": avg_think_tokens,
        # "think_found": think_found,
        # "think_fallback_full": fallback_full
    }

    # --------- save ---------
    output_dir = os.path.dirname(combined_file)
    
    # Get file or directory base name (last path component).
    base_name = os.path.basename(combined_file).split('.')[0]
    # output_dir, base_name = build_output_paths(args)
    metrics_path = os.path.join(output_dir, f"{base_name}.metrics.json")
    wrong_ids_path = os.path.join(output_dir, f"{base_name}.wrong_ids.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(wrong_ids_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wrong_ids), "ids": wrong_ids}, f, ensure_ascii=False, indent=2)
    print(f" ✅ Metrics saved to: {metrics_path}")
    print(f" ✅ Wrong IDs saved to: {wrong_ids_path} (count={len(wrong_ids)})")
    return metrics_path, wrong_ids_path

def merge_json_files(a_path, b_path, c_path, k=3):
    try:
        # 1. Read JSONL file a (one JSON object per line)
        data_a = []
        with open(a_path, 'r', encoding='utf-8') as f_a:
            for line in f_a:
                line = line.strip()
                if line:
                    data_a.append(json.loads(line))
        
        # 2. Read JSONL file b (one JSON object per line)
        data_b = []
        with open(b_path, 'r', encoding='utf-8') as f_b:
            for line in f_b:
                line = line.strip()
                if line:
                    data_b.append(json.loads(line))
        
        # 3. Apply the replacement operation
        result = data_a.copy()
        result[:k] = data_b[:k]
        
        # 4. Save result to file c in JSONL format
        with open(c_path, 'w', encoding='utf-8') as f_c:
            for item in result:
                f_c.write(json.dumps(item, ensure_ascii=False) + '\n')
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--combined_file", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    if args.baseline == None:
        combined_file = args.combined_file
    else:
        combined_file = args.combined_file.replace('.jsonl', "_true.jsonl")
        print(combined_file)
        merge_json_files(args.combined_file, args.baseline, combined_file, args.num_samples)

    evaluate_and_save(args, combined_file)

if __name__ == "__main__":
    main()
