import glob
import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import random

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

def set_seed(seed=42):
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Base components
    components = []
    
    # Optional run_id prefix
    if args.run_id:
        components.append(args.run_id.strip())
    
    components.append("steer")
    
    # Always include max_generated_tokens
    components.append(f"maxlen{args.max_generated_tokens}")
    
    # Always include seed
    components.append(f"seed{args.seed}")

    # Always include steer_layer
    components.append(f"layer{args.steer_layer}")
    
    # Optional q25
    if args.q25 is not None:
        components.append(f"q25_{args.q25}")
    
    # Optional q75
    if args.q75 is not None:
        components.append(f"q75_{args.q75}")
    
    # Optional low_val
    if args.low_val is not None:
        components.append(f"low_{args.low_val}")
    
    # Optional tau
    if args.tau is not None:
        components.append(f"tau_{args.tau}")
    
    # Optional token_budget
    if args.token_budget is not None:
        components.append(f"tbudget{args.token_budget}")
    
    base_name = "_".join(components)
    return output_dir, base_name

def merge_all_shards(output_dir, base_name, remove_shards=True):
    """rank 0: merge *.shard*.jsonl (+ existing combined file), deduplicate by idx, write {base}.jsonl"""
    shard_files = sorted(glob.glob(os.path.join(output_dir, "*.shard*.jsonl")))
    combined_file = os.path.join(output_dir, f"{base_name}.jsonl")

    sources = []
    if os.path.exists(combined_file):
        sources.append(combined_file)
    sources += shard_files

    merged = {}
    for fp in sources:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                idx = obj.get("idx", None)
                if isinstance(idx, int) and idx not in merged:
                    merged[idx] = obj

    final = [merged[k] for k in sorted(merged.keys())]
    write_jsonl_atomic(final, combined_file)

    # Delete shard files.
    if remove_shards and shard_files:
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
                print(f"[rank 0] 🗑️  Removed shard: {shard_file}")
            except Exception as e:
                print(f"[WARN][rank 0] Failed to remove {shard_file}: {e}")
    
    return combined_file, len(final)

def evaluate_and_save(args, combined_file):
    from utils.data_loader import load_data
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    from math import comb

    # --------- helpers ---------
    def _extract_first_text(gen):
        """Support multiple nested structures as long as possible; use the first text for length stats while correctness checks still consider all candidates."""
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
                # Use the first string in nested structures.
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
        metrics[f"pass@{args.k}"] = acc  # For single-sample evaluation, this is equivalent to accuracy.

    # Add completeness metadata.
    is_complete = len(outputs) == total
    metrics["is_complete"] = is_complete
    if not is_complete:
        metrics["missing_count"] = total - len(outputs)
    
    # --------- token length stats ---------
    # Statistics are computed on outputs
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
        # 2) Full text token count
        if text:
            full_tokens_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        else:
            full_tokens_len = 0
        full_token_counts.append(full_tokens_len)
        # 3) Think segment token count: truncate at [unused17], otherwise use full text
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
        "avg_think_token_count": avg_think_tokens,
        "think_found": think_found,
        "think_fallback_full": fallback_full
    }

    # --------- save ---------
    output_dir, base_name = build_output_paths(args)
    metrics_path = os.path.join(output_dir, f"{base_name}.metrics.json")
    wrong_ids_path = os.path.join(output_dir, f"{base_name}.wrong_ids.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(wrong_ids_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wrong_ids), "ids": wrong_ids}, f, ensure_ascii=False, indent=2)
    print(f"[rank 0] ✅ Metrics saved to: {metrics_path}")
    print(f"[rank 0] ✅ Wrong IDs saved to: {wrong_ids_path} (count={len(wrong_ids)})")
    return metrics_path, wrong_ids_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--steer_layer', type=int, default=22)
    parser.add_argument('--run_id', type=str, default="", help="Optional tag to separate different runs in filenames")
    parser.add_argument('--max_generated_tokens', type=int, default=16000)
    parser.add_argument('--token_budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--q25', type=float, default=None)
    parser.add_argument('--q75', type=float, default=None)
    parser.add_argument('--low_val', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    # Print received arguments.

    # n = args.model_name_or_path.split('/')[-1]
    # path = os.path.join(args.output_path, n, args.dataset, 'base/a.txt')
    # with open(path, 'r') as f:
    #     args.token_budget = int(f.readline().strip())
    output_dir, base_name = build_output_paths(args)
    combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=True)
    print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")
    evaluate_and_save(args, combined_file)


if __name__ == "__main__":
    main()
