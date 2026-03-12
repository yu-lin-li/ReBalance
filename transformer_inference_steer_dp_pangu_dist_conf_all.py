import glob
import os
import json
import argparse
import gc
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from re import split as rsplit
import random

# NOTE: keep your custom Qwen import path if you have a patched model
# e.g., your custom class that supports set_steering_flag/start_new_round
from modeling_utils.modeling_openpangu_dynamic import PanguEmbeddedForCausalLM  # ! change here for confidence only

# ---- silence all warnings/logging (place at very top) ----
import os, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"            # inherited by child processes
os.environ["TRANSFORMERS_VERBOSITY"] = "error"     # quieter HF transformers logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"     # disable tokenizer parallelism hint
warnings.filterwarnings("ignore")                  # ignore all Python warnings

# silence common library logs/warnings
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

# also silence numpy numeric warnings (overflow/invalid ops)
try:
    import numpy as np
    np.seterr(all="ignore")
except Exception:
    pass

# if warnings still appear through logging, hard-disable logging with caution
# logging.disable(logging.CRITICAL)
# ---- end silence ----

# ------------------------
# Utils
# ------------------------
def parse_optional_float(value):
    """Parse None or float from command line."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return float(value)

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

def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()

def empty_device_cache():
    try:
        if is_npu():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

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

def scan_existing_outputs(output_dir, base_name):
    """
    Scan combined output and shard files.
    Returns:
      existing_idx(set), existing_q(set)
    for checkpoint resume and duplicate skipping.
    """
    combined = os.path.join(output_dir, f"{base_name}.jsonl")
    shard_glob = os.path.join(output_dir, f"{base_name}.shard*.jsonl")
    files = []
    if os.path.exists(combined):
        files.append(combined)
    files += sorted(glob.glob(shard_glob))

    idx_set, q_set = set(), set()
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                i = obj.get("idx", None)
                q = obj.get("question", None)
                if isinstance(i, int):
                    idx_set.add(i)
                if isinstance(q, str):
                    q_set.add(q)
    return idx_set, q_set

def merge_all_shards(output_dir, base_name, remove_shards=True):
    """rank 0: merge *.shard*.jsonl (+ existing combined file), dedupe by idx, write {base}.jsonl"""
    shard_files = sorted(glob.glob(os.path.join(output_dir, f"{base_name}.shard*.jsonl")))
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

    # remove shard files
    if remove_shards and shard_files:
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
                print(f"[rank 0] 🗑️  Removed shard: {shard_file}")
            except Exception as e:
                print(f"[WARN][rank 0] Failed to remove {shard_file}: {e}")
    
    return combined_file, len(final)

# ------------------------
# Distributed helpers
# ------------------------

def _auto_backend():
    if is_npu():
        return "hccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"

def init_distributed_if_needed():
    """
    Initialize with env:// when launched by torchrun; otherwise single-process mode.
    Returns: (is_dist, rank, world_size, local_rank, device)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_dist and not dist.is_initialized():
        dist.init_process_group(backend=_auto_backend(), init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # device selection
    device = torch.device("cpu")
    if is_npu():
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    return is_dist, rank, world_size, local_rank, device

# ------------------------
# Sampling
# ------------------------

# Top-p sampling (matching HF TopPLogitsWarper behavior)
@torch.no_grad()
def top_p_sampling_step(last_logits, temperature: float, top_p: float, output_logprobs: bool):
    """
    last_logits: [1, vocab_size] on device
    returns: next_token_id [1,1], next_token_logprob scalar (float)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    logits = last_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # sort by prob desc
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # keep tokens within cumulative prob <= top_p (at least 1 token)
    cutoff = (cumsum > top_p)
    cutoff[..., 0] = False  # keep at least one
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    # sample in sorted space, then map back
    next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_sorted_idx)

    if output_logprobs:
        chosen_prob = sorted_probs.gather(-1, next_sorted_idx)
        next_logprob = torch.log(chosen_prob + 1e-12).item()
    else:
        next_logprob = None

    return next_token, next_logprob

def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )

    dtype = torch.float16 if is_npu() else torch.float32

    model = PanguEmbeddedForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)

    model.eval()
    return model, tokenizer, dtype


# Custom step-wise sampling (matching HF generate do_sample+top_p+temperature behavior)
@torch.no_grad()
def sample_with_tracking(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    dtype: torch.dtype,
    device: torch.device,
    think_budget: int = None,
    answer_budget: int = None,
    think_end_ids: list = None,
    output_logprobs: bool = False
):
    """
    Returns:
      generated_ids: newly generated token ids ([T])
      token_logprobs: per-step logprobs aligned with generated_ids (list[float])
    """
    token_logprobs = []
    generated = []
    past_key_values = None

    # prefill full prompt to build KV cache (generation logic unchanged)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=None)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

    total_generated = 0
    answer_generated = 0
    in_answer_phase = False
    think_end_inserted = False

    def _append_forced_token(token_id: int):
        nonlocal past_key_values, last_logits, total_generated, answer_generated
        token_tensor = torch.tensor([[token_id]], device=device)
        if output_logprobs:
            logprob = torch.log_softmax(last_logits, dim=-1)[0, token_id].item()
        else:
            logprob = None
        token_logprobs.append(logprob)
        generated.append(token_id)
        total_generated += 1
        if in_answer_phase:
            answer_generated += 1
        with torch.inference_mode():
            out = model(
                input_ids=token_tensor,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :]

    while total_generated < max_new_tokens:
        if (not in_answer_phase) and (think_budget is not None) and total_generated >= think_budget:
            if think_end_ids and not think_end_inserted:
                for token_id in think_end_ids:
                    if total_generated >= max_new_tokens:
                        break
                    _append_forced_token(token_id)
                think_end_inserted = True
            in_answer_phase = True

        if in_answer_phase and answer_budget is not None and answer_generated >= answer_budget:
            break

        next_token, next_logprob = top_p_sampling_step(
            last_logits,
            temperature,
            top_p,
            output_logprobs=output_logprobs
        )
        token_logprobs.append(next_logprob)

        token_id = next_token.item()
        generated.append(token_id)
        total_generated += 1
        if in_answer_phase:
            answer_generated += 1

        if eos_token_id is not None and token_id == eos_token_id:
            break

        with torch.inference_mode():
            out = model(
                input_ids=next_token,  # [1,1]
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :]

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), []
    
    del past_key_values, last_logits, out, next_token
    empty_device_cache()

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs

# ------------------------
# Core worker
# ------------------------

def worker(args, rank, world_size, local_rank, device):
    # Dataset
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)
    N = len(questions)

    # Output paths
    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank:03d}.jsonl")
    # scan existing results (combined + all shards) for resume/skip
    existing_idx_global, existing_q_global = scan_existing_outputs(output_dir, base_name)

    # Model & tokenizer
    model, tokenizer, dtype = load_model_and_tokenizer(args, device)
    dist.barrier()
    if rank == 0:  
        print("=" * 80)
        print(f"[INFO] device = {model.device}")
        print("=" * 80)
        print(f"[INFO] model = {model}")
        print("=" * 80)
        print(f"[INFO] world_size = {world_size}")
        print(f"[INFO] output_dir = {output_dir}")
        print("=" * 80)
    dist.barrier()
    think_budget = None
    answer_budget = None
    think_end_ids = None
    if args.token_budget is not None and args.token_budget > 0:
        think_budget = int(args.token_budget)
        # answer_budget = max(1, think_budget // 4)  # ! remove answer budget here
    think_end_ids = tokenizer.encode("\n[unused17]\n\n", add_special_tokens=False)

    # Load steer vector
    steer_vector = torch.load(args.steer_vector_path, map_location="cpu").to(device, dtype=dtype)
    # Enable steering (custom API in your patched model)
    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
        q25=args.q25,
        q75=args.q75,
        low_val=args.low_val,
        tau=args.tau
    )

    # partition samples with DistributedSampler (no manual i % world_size)
    sampler = DistributedSampler(
        list(range(N-args.budget_num)),
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    sampler.set_epoch(0)
    raw_idx = list(iter(sampler))
    # deduplicate to avoid padding duplicates
    seen, my_indices = set(), []
    for i in raw_idx:
        if i < N and i not in seen:
            seen.add(i)
            my_indices.append(i)
    filtered_list = [x for x in my_indices if x >= my_indices[0]]
    my_indices = filtered_list

    print("=" * 80)
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx_global)}, question={len(existing_q_global)}")
    print(f"[rank {rank}] will process {len(my_indices)} items")
    # print(f"[rank {rank}] will process {my_indices} items")
    print("=" * 80)

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)

    for ii in my_indices:
        i = ii + args.budget_num
        q = questions[i]
        qtext = q.get("problem", "")

        # checkpoint resume: skip already processed samples
        if (i in existing_idx_global) or (qtext in existing_q_global):
            pbar.update(1)
            continue

        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": qtext}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                # custom step sampler avoids keeping all step scores in HF cache
                gen_ids, step_logprobs = sample_with_tracking(
                    model=model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
                    max_new_tokens=args.max_generated_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    dtype=dtype,
                    device=device,
                    think_budget=think_budget,
                    answer_budget=answer_budget,
                    think_end_ids=think_end_ids,
                    output_logprobs=False
                )
                # concatenate prompt and generated tokens for decoding
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids], dim=0)
                response_text = tokenizer.decode(full_ids[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i} : {qtext}...")
            empty_device_cache()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        result = {
            "idx": i,  # store idx for robust checkpointing
            "question": qtext,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", "")
        }

        # Append to shard file
        with open(shard_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # actively release memory fragmentation
        del inputs, gen_ids
        empty_device_cache()
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")

# ------------------------
# Evaluation
# ------------------------

def evaluate_and_save(args, combined_file):
    from utils.data_loader import load_data
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    from math import comb

    # --------- helpers ---------
    def _extract_first_text(gen):
        """Handle diverse structures, use first text for length stats; all candidates remain for grading."""
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
                # take first string from list items
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
        metrics[f"pass@{args.k}"] = acc  # pass degrades to accuracy for single-sample case

    # add integrity metadata
    is_complete = len(outputs) == total
    metrics["is_complete"] = is_complete
    if not is_complete:
        metrics["missing_count"] = total - len(outputs)
    
    # --------- token length stats ---------
    # stats based on outputs
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
        # 1) word count
        resp_word_counts.append(len(text.split()) if text else 0)
        # 2) full token count
        if text:
            full_tokens_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        else:
            full_tokens_len = 0
        full_token_counts.append(full_tokens_len)
        # 3) think segment token count: truncate at [unused17], fallback to full text
        lower = text.lower() if text else ""
        idx = lower.find("[unused17]")  # Pangu thinking boundary marker
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
    parser.add_argument('--steer_vector_path', type=str, required=True)
    parser.add_argument('--steer_layer', type=int, default=22)
    parser.add_argument('--steer_coef', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--run_id', type=str, default="", help="Optional tag to separate different runs in filenames")
    parser.add_argument('--max_generated_tokens', type=int, default=16000)
    parser.add_argument('--token_budget', type=int, default=None)
    parser.add_argument('--budget_num', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--q25', type=float, default=None)
    parser.add_argument('--q75', type=float, default=None)
    parser.add_argument('--low_val', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    set_seed(args.seed)
    
    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    if args.budget_num > 0:
        n = args.model_name_or_path.split('/')[-1]
        path = os.path.join(args.output_path, n, args.dataset, 'base/a.txt')
        with open(path, 'r') as f:
            args.token_budget = int(f.readline().strip())

    # print received arguments
    if rank == 0:
        print("=" * 80)
        print("Received Arguments:")
        print("=" * 80)
        for arg, value in vars(args).items():
            print(f"  {arg} = {value}")
        print("=" * 80)
    
    worker(args, rank, world_size, local_rank, device)


if __name__ == "__main__":
    main()
