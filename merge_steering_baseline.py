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
    shard_files = sorted(glob.glob(os.path.join(output_dir, f"{base_name}.shard*.jsonl")))
    combined_file = os.path.join(output_dir, 'base', f"{base_name}.jsonl")

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


import argparse
import json
from typing import Any, List

from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )


def extract_all_texts(gens: Any) -> List[str]:
    out: List[str] = []
    if isinstance(gens, list):
        for g in gens:
            if isinstance(g, str):
                out.append(g)
            elif isinstance(g, dict):
                for key in (
                    "text",
                    "content",
                    "generated_response",
                    "generated_text",
                    "output",
                    "message",
                    "response",
                ):
                    v = g.get(key)
                    if isinstance(v, str):
                        out.append(v)
                        break
            elif isinstance(g, list):
                # Fall back to the first string-like item
                for item in g:
                    if isinstance(item, str):
                        out.append(item)
                        break
                    if isinstance(item, dict):
                        for key in (
                            "text",
                            "content",
                            "generated_response",
                            "generated_text",
                            "output",
                            "message",
                            "response",
                        ):
                            v = item.get(key)
                            if isinstance(v, str):
                                out.append(v)
                                break
                        if out:
                            break
    return out


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

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
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--q25', type=float, default=None)
    parser.add_argument('--q75', type=float, default=None)
    parser.add_argument('--low_val', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    # print received arguments

    output_dir, base_name = build_output_paths(args)
    combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=False)
    print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")

    tokenizer = load_tokenizer(args.model_name_or_path)

    total_tokens = 0
    total_responses = 0
    with open(combined_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            gens = obj.get("generated_responses", [])
            texts = extract_all_texts(gens)
            idx = obj.get("idx", i)
            print(f"sample {i} (idx={idx}) responses={len(texts)}")
            sample_tokens = 0
            for j, text in enumerate(texts):
                tok_len = count_tokens(tokenizer, text)
                sample_tokens += tok_len
                total_tokens += tok_len
                total_responses += 1
                print(f"  response {j}: tokens={tok_len}")
            if texts:
                avg_len = sample_tokens / len(texts)
                print(f"  avg_tokens={avg_len:.2f}")
            else:
                print("  avg_tokens=0.00")

    if total_responses:
        overall_avg = total_tokens / total_responses
        print(f"overall_avg_tokens={overall_avg:.2f} (responses={total_responses})")
    else:
        print("overall_avg_tokens=0.00 (responses=0)")
    path = os.path.join(output_dir, 'base/a.txt')
    with open(path, 'w') as f:
        f.write(str(int(overall_avg)))

if __name__ == "__main__":
    main()
