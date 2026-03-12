# -*- coding: utf-8 -*-
import glob
import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import os
import json
import argparse
import sys
import gc
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from typing import List
from re import split as rsplit


# ------------------------
# Utils
# ------------------------

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(tmp, file_path)


def read_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        # The following two lines are not required for NPU/Ascend but are kept for compatibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _clear_npu():
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        try:
            torch.npu.ipc_collect()
        except Exception:
            pass
    gc.collect()


def _clear_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def _clear_device():
    _clear_npu()
    _clear_cuda()

# ------------------------
# Distributed helpers
# ------------------------

def _auto_backend():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "hccl"   # Ascend NPU
    if torch.cuda.is_available():
        return "nccl"   # NVIDIA GPU
    return "gloo"       # CPU fallback


def init_distributed_if_needed():
    """Initialize using env:// when launched by torchrun across nodes; otherwise fall back to single process."""
    # Automatically set by torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_dist and not dist.is_initialized():
        backend = _auto_backend()
        dist.init_process_group(backend=backend, init_method="env://")  # use environment variables
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    # device
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)  # default device
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return is_dist, rank, world_size, local_rank, device

# ------------------------
# Confidence helpers
# ------------------------

def _summarize_selected_logprobs(selected_logprobs: torch.Tensor, policy: str = 'avg2') -> float:
    if selected_logprobs is None or selected_logprobs.numel() == 0:
        return 0.0
    probs = torch.exp(selected_logprobs)
    if policy == 'min':
        return probs.min().item()
    elif policy == 'avg1':
        return probs.mean().item()
    return torch.exp(selected_logprobs.mean()).item()


def compute_token_logprobs_streaming(model, tokenizer, prompt_ids: torch.Tensor, gen_ids: torch.Tensor,
                                     device: torch.device, score_dtype: str = 'bf16') -> torch.Tensor:
    assert prompt_ids.ndim == 2 and prompt_ids.size(0) == 1
    assert gen_ids.ndim == 1

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = dtype_map.get(score_dtype, torch.bfloat16)

    logps = []
    with torch.inference_mode():
        out = model(prompt_ids.to(device), use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        logprob0 = F.log_softmax(logits, dim=-1)[0, gen_ids[0].to(device)].item()
        logps.append(logprob0)

        if gen_ids.numel() > 1:
            prev = gen_ids[0].view(1, 1).to(device)
            for t in range(1, gen_ids.numel()):
                out = model(prev, use_cache=True, past_key_values=past, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                lp = F.log_softmax(logits, dim=-1)[0, gen_ids[t].to(device)].item()
                logps.append(lp)
                prev = gen_ids[t].view(1, 1).to(device)

    return torch.tensor(logps, dtype=torch.float32)


# ------------------------
# Model / tokenizer
# ------------------------

def load_model_and_tokenizer_single_npu(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        local_files_only=True
    )

    model.to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Hidden-state saving
# ------------------------

def save_qwen2_think_split_tokens_only(model, tokenizer, input_ids, full_text, save_path,
                                       hs_device: str = 'auto'):
    import os as _os

    ids_flat = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    think_ids = tokenizer.encode("[unused17]", add_special_tokens=False)

    def _find_subseq(seq, subseq):
        if not subseq:
            return None
        L, M = len(seq), len(subseq)
        for s in range(0, L - M + 1):
            if seq[s:s + M] == subseq:
                return s
        return None

    pos = _find_subseq(ids_flat, think_ids)
    think_end_idx = pos if pos is not None else len(ids_flat)

    step_positions = []
    for i in range(think_end_idx - 1):
        tok = tokens[i]
        if tok == ".\n\n" or tok == "\n\n" or ("\n\n" in tok):
            step_positions.append(i + 1)

    hidden_dict = {}
    sample_id = 0

    def _forward_on(device):
        with torch.inference_mode():
            outputs = model(input_ids.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        for layer_id, layer_h in enumerate(hidden_states):
            h = layer_h.squeeze(0)
            if len(step_positions) > 0:
                idx = torch.tensor(step_positions, dtype=torch.long, device=h.device)
                step_h = h.index_select(dim=0, index=idx).to('cpu', non_blocking=True)
            else:
                step_h = torch.empty((0, h.shape[1]), dtype=h.dtype)
            hidden_dict[layer_id] = step_h

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ('auto', 'npu') and torch.npu.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != 'cpu':
                model.to('cpu')
            _forward_on(torch.device('cpu'))
    except torch.npu.OutOfMemoryError:
        print("[hs][gpu OOM] falling back to CPU for hidden-state dump...")
        _clear_device()
        if hs_device == 'npu':
            raise
        model.to('cpu')
        _forward_on(torch.device('cpu'))
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    tmp_path = save_path + ".tmp"
    torch.save(hidden_dict, tmp_path)
    _os.replace(tmp_path, save_path)


# ------------------------
# Shard helpers
# ------------------------

def load_existing_indices(shard_file):
    existing_idx = set()
    existing_q = set()
    num_lines = 0
    if os.path.exists(shard_file):
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                num_lines += 1
                try:
                    obj = json.loads(s)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    if "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    continue
    return existing_idx, existing_q, num_lines


def _read_jsonl_map_by_idx(file_path):
    m = {}
    if not os.path.exists(file_path):
        return m
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "idx" in obj:
                    m[int(obj["idx"])] = obj
            except Exception:
                continue
    return m


# ------------------------
# Shard helpers
# ------------------------

def load_existing_indices(file_path):
    existing_idx = set()
    existing_q = set()
    num_lines = 0
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                num_lines += 1
                try:
                    obj = json.loads(s)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    if "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    continue
    return existing_idx, existing_q, num_lines


def _read_jsonl_map_by_idx(file_path):
    m = {}
    if not os.path.exists(file_path):
        return m
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "idx" in obj:
                    m[int(obj["idx"])] = obj
            except Exception:
                continue
    return m


def scan_existing_outputs(output_dir, base_name):
    """
    Scan combined output and all shards, then build idx->obj / q->obj maps.
    Helps with hidden-state rescue on rerun, skipping duplicates, and final dedup merge.
    """
    combined = os.path.join(output_dir, f'{base_name}.jsonl')
    shard_glob = os.path.join(output_dir, f'{base_name}.shard*.jsonl')

    files = []
    if os.path.exists(combined):
        files.append(combined)
    files += sorted(glob.glob(shard_glob))

    idx_set, q_set = set(), set()
    idx_map, q_map = {}, {}
    total = 0

    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                total += 1
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                i = obj.get("idx", None)
                q = obj.get("question", None)
                if isinstance(i, int):
                    if i not in idx_map:
                        idx_map[i] = obj
                    idx_set.add(i)
                if isinstance(q, str):
                    if q not in q_map:
                        q_map[q] = obj
                    q_set.add(q)
    return idx_set, q_set, idx_map, q_map, total


def _reconstruct_full_text(tokenizer, sys_prompt, question_text, response_text):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + (response_text or "")


def merge_all_shards(output_dir, base_name, remove_shards=True):
    """rank 0: after all ranks finish, merge *.shard*.jsonl (+ existing combined file), dedupe by idx, and write {base}.jsonl"""
    shard_files = sorted(glob.glob(os.path.join(output_dir, f'{base_name}.shard*.jsonl')))
    combined_file = os.path.join(output_dir, f'{base_name}.jsonl')

    # also merge existing combined file to support checkpoint resume
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

    # sort by idx for output
    final = [merged[k] for k in sorted(merged.keys())]
    write_jsonl(final, combined_file)

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
# Worker
# ------------------------

def worker(args, rank, world_size, device):
    set_seeds(42 + rank)  # ! whether to vary by rank?
    dataset_file = 'train.jsonl' if args.dataset == 'Math_Math' else 'test.jsonl'
    dataset_path = os.path.join(args.dataset_dir, args.dataset, dataset_file)
    questions = read_jsonl(dataset_path)
    if args.dataset == 'Math_Math':
        questions = questions[:500]
    N = len(questions)  # TODO

    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    shard_file = os.path.join(output_dir, f'{base_name}.shard{rank:03d}.jsonl')

    existing_idx_global, existing_q_global, existing_map_by_idx, existing_map_by_q, total_lines = \
        scan_existing_outputs(output_dir, base_name)

    model, tokenizer = load_model_and_tokenizer_single_npu(args, device)
    
    # partition indices with DistributedSampler
    sampler = DistributedSampler(  # rank=i => indices[i:total_size:world_size]
        list(range(N)), num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False
    )

    # use sampler assignments and drop padding/duplicates
    raw_idx = list(iter(sampler))
    seen = set()
    my_indices = []
    for i in raw_idx:
        if i < N and i not in seen:
            seen.add(i)
            my_indices.append(i)
    dist.barrier()
    if rank == 0:  
        print("=" * 80)
        print(f"[INFO] device = {model.device}")
        print("=" * 80)
        print(f"[INFO] model = {model}")
        print("=" * 80)
        print(f"[INFO] world_size = {world_size}")
        print(f"[INFO] output_dir = {output_dir}")
        print(f"[INFO] combined existing lines = {total_lines}")
        print("=" * 80)
    dist.barrier()
    
    import time
    time.sleep(0.2*rank)
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] will process {len(my_indices)} items")
    print("=" * 80)
    dist.barrier()
    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in my_indices:
        q = questions[i]
        q_text = q.get("problem", "")
        hidden_save_path = os.path.join(output_dir, f"hidden_{i:3d}.pt")

        if (i in existing_idx_global) or (q_text in existing_q_global):
            if not os.path.exists(hidden_save_path):
                try:
                    entry = existing_map_by_idx.get(i) or existing_map_by_q.get(q_text)
                    if entry and entry.get("generated_responses"):
                        response_text = entry["generated_responses"][0] if entry["generated_responses"] else ""
                        full_text = _reconstruct_full_text(tokenizer, sys_prompt, q_text, response_text)
                        full_inputs = tokenizer(full_text, return_tensors="pt")
                        save_qwen2_think_split_tokens_only(
                            model, tokenizer, full_inputs["input_ids"].to(device), full_text, hidden_save_path,
                            hs_device=args.hs_device
                        )  # TODO
                        del full_inputs
                        print(f"[rank {rank}] rescued hidden for idx={i}")
                        _clear_device()
                except Exception as e_rescue:
                    print(f"[WARN][rank {rank}] rescue hidden failed at idx={i}: {e_rescue}")
                    _clear_device
            pbar.update(1)
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generated_tokens,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] idx={i} : {q_text[:80]}... skipping.")
            _clear_device()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        response_text = tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        full_text = prompt + response_text

        gen_token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:].detach().cpu()
        try:
            gen_logps = compute_token_logprobs_streaming(
                model, tokenizer,
                prompt_ids=inputs["input_ids"],
                gen_ids=gen_token_ids,
                device=device,
                score_dtype=args.score_dtype,
            )
        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] logprob pass at idx={i}; skipping confidences.")
            gen_logps = torch.empty(0)
        except Exception as e:
            print(f"[WARN][rank {rank}] logprob pass failed at idx={i}: {e}")
            gen_logps = torch.empty(0)

        text_before_think = response_text.split('[unused17]')[0]
        text_segments = rsplit(r'\n\n+', text_before_think)  # TODO: do not use texts to split tokens
        confidences = []
        start = 0
        for segment in text_segments:
            seg_ids = tokenizer(segment, add_special_tokens=False)['input_ids']
            end = start + len(seg_ids)
            if gen_logps.numel() > 0 and end > start and end <= gen_logps.numel():
                seg_logps = gen_logps[start:end]
                conf = _summarize_selected_logprobs(seg_logps, policy='avg2')
                confidences.append(conf)
            elif end > start:
                confidences.append(0.0)
            start = end

        try:
            if not os.path.exists(hidden_save_path):
                full_inputs = tokenizer(full_text, return_tensors="pt")
                save_qwen2_think_split_tokens_only(
                    model, tokenizer, full_inputs["input_ids"].to(device), full_text, hidden_save_path,
                    hs_device=args.hs_device
                )
                del full_inputs
            else:
                print(f"[rank {rank}] hidden exists, skip: {hidden_save_path}")
        except torch.npu.OutOfMemoryError:
            print(f"[WARN][rank {rank}] hidden save OOM at idx={i}; will rely on rescue next run.")
            _clear_device()
        except Exception as he:
            print(f"[WARN][rank {rank}] hidden save failed at idx={i}: {he} (rescueable next run)")

        result_entry = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences,
        }
        with open(shard_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

        del output, inputs, gen_token_ids, gen_logps
        _clear_device()
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--hs_device', type=str, default='auto', choices=['auto', 'npu', 'cpu'])
    parser.add_argument('--score_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    worker(args, rank, world_size, device)


if __name__ == "__main__":
    main()
