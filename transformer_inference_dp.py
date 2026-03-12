# -*- coding: utf-8 -*-
"""

What this script does
- Loads a HuggingFace causal LM + tokenizer from `--model_name_or_path`.
- Reads a JSONL dataset from: {dataset_dir}/{dataset}/test.jsonl
  Each line is expected to be a dict containing at least:
    - "problem": str (the user question / math problem)
  Optionally:
    - "answer": str (gold answer for reference; not used for generation)
- Runs deterministic (greedy) generation per example on multiple GPUs using Python multiprocessing:
    - Example i is assigned to rank = i % world_size
- Writes results into a per-rank shard file:
    {output_path}/{model_basename}/{dataset}/origin_temp{temperature}_maxlen{max_tokens}.shard{rank}.jsonl
- Optionally dumps hidden states to:
    {output_path}/{model_basename}/{dataset}/hidden_{idx}.pt

Key design choices / behaviors
- Deterministic generation: `do_sample=False` (greedy). `temperature` and `top_p` are still kept as CLI args
  to preserve a compatible interface if you switch sampling on in the future.
- Resume-friendly:
    - If an item already exists in the shard, it is skipped.
    - If the item exists but `hidden_{idx}.pt` is missing, a "rescue" path reconstructs the original
      prompt+response and re-dumps hidden states without regenerating tokens.
- Hidden-state dumping is "think-step" oriented:
    - It heuristically detects step boundaries before `</think>` based on newline-like tokens.
    - This is tokenizer/model dependent and should be treated as a best-effort heuristic, not a guarantee.

Security note on `--trust_remote_code`
- When enabled, HuggingFace may execute arbitrary Python code from the model repository.
  Only use it for repositories you trust.

Typical usage
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
python transformer_inference_dp.py \\
  --model_name_or_path /path/to/model \\
  --dataset_dir ./Data \\
  --dataset Math_Train \\
  --output_path ./outputs \\
  --max_generated_tokens 16000 \\
  --num_gpus 8 \\
  --trust_remote_code

License / attribution
- This script is meant to be included in your open-source repo as an inference utility.
- Please add your project license and any third-party notices at the repository level.
"""

# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
# By default we suppress warnings and HF "error-level" logs to keep multi-process
# output readable. For open-source projects, consider making this configurable,
# but we intentionally do NOT change runtime behavior here.
import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import os
import json
import argparse
import sys
import gc
import random
from typing import List
from re import split as rsplit

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


# ------------------------
# Utils
# ------------------------

def write_jsonl(data, file_path):
    """
    Write a list of Python dict objects to a JSONL file.

    Notes
    - Uses ensure_ascii=False to preserve non-ASCII characters.
    - Creates parent directory if needed.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(file_path):
    """
    Read a JSONL file into a list of dicts.

    If the file does not exist, return an empty list (and print a warning).
    """
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def set_seeds(seed=42):
    """
    Make the run as deterministic as possible across Python/Numpy/PyTorch.

    Notes
    - For GPU runs, enables deterministic cuDNN behavior and disables benchmark.
    - Sets matmul precision to "high" (affects speed/precision trade-off in PyTorch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")


def _clear_cuda():
    """
    Best-effort memory cleanup for long multi-sample inference loops.

    Why this exists
    - Generation + hidden-state dumping can retain GPU memory.
    - In multi-process settings, small leaks accumulate and increase OOM risk.

    What it does
    - Synchronize GPU (best-effort)
    - empty_cache + ipc_collect (best-effort)
    - gc.collect
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


# ------------------------
# Confidence calculator
# ------------------------

def _summarize_selected_logprobs(selected_logprobs: torch.Tensor, policy: str = 'avg2') -> float:
    """
    Convert a sequence of token log-probabilities into a single scalar "confidence" proxy.

    Inputs
    - selected_logprobs: Tensor[seq_len], log p(token_t | context)
    - policy:
        - 'min'  : min token probability in the segment
        - 'avg1' : mean token probability in the segment
        - 'avg2' : exp(mean logprob) == geometric mean of probabilities (default)

    Output
    - float confidence proxy in [0, 1] (for typical well-formed probabilities)

    Important
    - This is NOT calibrated model confidence.
    - It is a lightweight token-probability summary useful for relative comparisons only.
    """
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
    """
    Compute per-token log probabilities for the generated continuation, in a streaming manner.

    Motivation
    - After calling `model.generate`, we have the generated token ids.
    - We then want log p(gen_t | prompt + gen_<t) for each generated token.
    - Doing this in a streaming fashion reduces peak memory vs recomputing full logits.

    Inputs
    - prompt_ids: Tensor[1, prompt_len]
    - gen_ids:    Tensor[gen_len] (CPU tensor is OK; we'll move ids per-step)
    - device:     where to run the model forward passes
    - score_dtype:
        - 'bf16', 'fp16', 'fp32' controls autocast dtype for scoring pass

    Output
    - Tensor[gen_len] float32 log probabilities on CPU.

    Notes
    - We run the first forward pass on the full prompt to obtain KV cache.
    - Then we feed generated tokens one-by-one with `past_key_values`.
    """
    assert prompt_ids.ndim == 2 and prompt_ids.size(0) == 1
    assert gen_ids.ndim == 1

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = dtype_map.get(score_dtype, torch.bfloat16)

    logps = []
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=amp_dtype):
        # First pass: full prompt, build KV cache, and score the first generated token.
        out = model(prompt_ids.to(device), use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        logprob0 = F.log_softmax(logits, dim=-1)[0, gen_ids[0].to(device)].item()
        logps.append(logprob0)

        # Subsequent passes: one token at a time with KV cache.
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

def load_model_and_tokenizer_single_gpu(args, device):
    """
    Load tokenizer + model onto a single device (one process == one GPU).

    Tokenizer
    - Ensures pad_token_id is set (fallback to eos_token_id if missing).
    - Uses left-padding / left-truncation for chat prompts (common for generation).

    Model
    - Attempts to use FlashAttention2 if available; otherwise falls back to SDPA.
    - Uses bfloat16 on CUDA by default for better speed/memory trade-off.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and hasattr(tokenizer, 'eos_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        # FlashAttention2 may not be compiled/available; SDPA is a safe fallback.
        print(f"[warn] flash_attention_2 unavailable ({e}). Falling back to 'sdpa'.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

    model.to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Hidden-state saving
# ------------------------

def save_qwen2_think_split_tokens_only(model, tokenizer, input_ids, full_text, save_path,
                                       hs_device: str = 'auto'):
    """
    Dump hidden states for a single sample, focusing on "step" token positions inside the <think> region.

    What is saved
    - A dict: hidden_dict[layer_id][sample_id]["step"] = Tensor[num_steps, hidden_dim] on CPU
    - Where `num_steps` is the number of detected step boundary positions.

    How step boundaries are detected
    - We search for the token sequence of "</think>" in the *full input* token ids.
    - We only consider tokens *before* "</think>" as part of the "thinking" region.
    - Within that region, we treat tokens containing "ĊĊ" (newlines in some tokenizers) as boundaries.
      This is heuristic and tokenizer-dependent.

    Device strategy
    - If `hs_device` is 'auto' or 'cuda', we try running the forward pass on the model's current device.
    - If OOM occurs, we fall back to CPU unless `hs_device == 'cuda'` explicitly requests GPU-only.

    Atomic write
    - Saves to `save_path + ".tmp"` and then renames to `save_path` to avoid partial/corrupt files.
    """
    import os as _os

    # Flattened ids for subsequence search.
    ids_flat = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Token ids representing "</think>" (no special tokens).
    think_ids = tokenizer.encode("</think>", add_special_tokens=False)

    def _find_subseq(seq, subseq):
        """Return start index of subseq in seq, or None if not found."""
        if not subseq:
            return None
        L, M = len(seq), len(subseq)
        for s in range(0, L - M + 1):
            if seq[s:s + M] == subseq:
                return s
        return None

    # Locate "</think>" in the full input. If absent, we treat the whole sequence as "thinking region".
    pos = _find_subseq(ids_flat, think_ids)
    think_end_idx = pos if pos is not None else len(ids_flat)

    # Detect heuristic step boundary positions (token index + 1).
    # These positions are later used as index_select points in the sequence dimension.
    step_positions = []
    for i in range(think_end_idx - 1):
        tok = tokens[i]
        if tok == ".ĊĊ" or tok == "ĊĊ" or ("ĊĊ" in tok):
            step_positions.append(i + 1)

    hidden_dict = {}
    sample_id = 0

    def _forward_on(device):
        """
        Run a full forward pass with output_hidden_states=True and gather step vectors.

        Notes
        - We intentionally disable cache here (use_cache=False) because we want full hidden states.
        - We store step vectors on CPU to keep the saved artifact portable and to reduce GPU memory.
        """
        with torch.inference_mode():
            outputs = model(input_ids.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        for layer_id, layer_h in enumerate(hidden_states):
            # layer_h: [1, seq_len, hidden_dim]
            h = layer_h.squeeze(0)  # [seq_len, hidden_dim]
            if len(step_positions) > 0:
                idx = torch.tensor(step_positions, dtype=torch.long, device=h.device)
                step_h = h.index_select(dim=0, index=idx).to('cpu', non_blocking=True)
            else:
                step_h = torch.empty((0, h.shape[1]), dtype=h.dtype)
            hidden_dict[layer_id] = {sample_id: {"step": step_h}}

    # Remember original model device so we can restore it if we move to CPU temporarily.
    orig_device = next(model.parameters()).device
    try:
        if hs_device in ('auto', 'cuda') and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            # CPU mode: move model to CPU if needed.
            if orig_device.type != 'cpu':
                model.to('cpu')
            _forward_on(torch.device('cpu'))
    except torch.cuda.OutOfMemoryError:
        print("[hs][gpu OOM] falling back to CPU for hidden-state dump...")
        _clear_cuda()
        if hs_device == 'cuda':
            # User explicitly requested GPU-only behavior.
            raise
        model.to('cpu')
        _forward_on(torch.device('cpu'))
    finally:
        # Restore model back to its original device if we moved it.
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
    """
    Scan an existing shard JSONL file and collect:
    - existing_idx: set of `idx` already written
    - existing_q:   set of `question` strings already written
    - num_lines:    number of non-empty lines observed

    This supports "resume" behavior: skip already-processed examples.
    """
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
                    # If a malformed line exists, ignore it and continue scanning.
                    continue
    return existing_idx, existing_q, num_lines


def _read_jsonl_map_by_idx(file_path):
    """
    Read a JSONL shard file into a dict keyed by `idx`.

    Purpose
    - Needed for "hidden rescue": if we already have a generated response for idx,
      we can reconstruct full text and dump hidden states without regenerating.
    """
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


def _reconstruct_full_text(tokenizer, sys_prompt, question_text, response_text):
    """
    Reconstruct the exact "full_text" used for hidden-state dumping.

    full_text = chat_template(system + user + generation_prompt) + response_text

    Important
    - This assumes the tokenizer's chat template matches the one used when the
      response was originally generated.
    - If you change templates or system prompt, "rescued" hidden states may not
      align with the original generation context.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + (response_text or "")


# ------------------------
# Worker
# ------------------------

def worker(rank, world_size, args):
    """
    One worker process bound to one GPU (rank).

    Responsibilities
    - Load model/tokenizer on its device.
    - Select the subset of examples assigned to this rank: i % world_size == rank.
    - Resume from existing shard file.
    - For each assigned example:
        - If already processed: optionally rescue hidden states.
        - Else: generate response, compute token-level logprobs, summarize per "segment" confidence,
                dump hidden states, append to shard file.
    """
    set_seeds(42)

    # In a standard `CUDA_VISIBLE_DEVICES=...` setting, rank maps to the visible GPU index.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset convention: dataset/test.jsonl
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)

    # Output convention: outputs/{model_basename}/{dataset}/
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Shard filename encodes temperature and max_new_tokens for provenance.
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    shard_file = os.path.join(output_dir, f'{base_name}.shard{rank}.jsonl')

    # Resume bookkeeping.
    existing_idx, existing_q, num_lines = load_existing_indices(shard_file)
    existing_map = _read_jsonl_map_by_idx(shard_file)

    # Load model/tokenizer on the worker's device.
    model, tokenizer = load_model_and_tokenizer_single_gpu(args, device)

    # Data-parallel assignment: each rank handles indices congruent to rank mod world_size.
    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}, lines={num_lines}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    # Progress bar per rank. `position=rank` keeps multi-rank bars visually separated.
    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)

    # System prompt for math-style reasoning tasks.
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in my_indices:
        q = questions[i]

        # Hidden-state artifact path is index-based, independent of rank.
        hidden_save_path = os.path.join(output_dir, f"hidden_{i}.pt")

        # ---------------------------------------------------------------------
        # Resume / skip logic
        # ---------------------------------------------------------------------
        # If the item already exists in the shard (by idx OR question string),
        # we skip generation. However, if hidden states are missing, we try to
        # "rescue" them from the stored response.
        if (i in existing_idx) or (q.get("problem") in existing_q):
            if not os.path.exists(hidden_save_path):
                try:
                    entry = existing_map.get(i)
                    if entry and entry.get("generated_responses"):
                        response_text = entry["generated_responses"][0] if entry["generated_responses"] else ""
                        full_text = _reconstruct_full_text(tokenizer, sys_prompt, q.get("problem", ""), response_text)
                        full_inputs = tokenizer(full_text, return_tensors="pt")
                        save_qwen2_think_split_tokens_only(
                            model, tokenizer, full_inputs["input_ids"].to(device), full_text, hidden_save_path,
                            hs_device=args.hs_device
                        )
                        print(f"[rank {rank}] rescued hidden for idx={i}")
                        del full_inputs
                        _clear_cuda()
                except Exception as e_rescue:
                    print(f"[WARN][rank {rank}] rescue hidden failed at idx={i}: {e_rescue}")
                    _clear_cuda()
            pbar.update(1)
            continue

        # ---------------------------------------------------------------------
        # Build chat prompt with the tokenizer's chat template
        # ---------------------------------------------------------------------
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q['problem']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ---------------------------------------------------------------------
        # Generation (greedy)
        # ---------------------------------------------------------------------
        # Note: do_sample=False makes decoding deterministic.
        # temperature/top_p are passed for interface compatibility but unused by greedy decoding.
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generated_tokens,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            _clear_cuda()
            print(f"[OOM][rank {rank}] idx={i} : {q['problem'][:80]}... skipping.")
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        # Decode the generated continuation only (excluding prompt).
        response_text = tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        full_text = prompt + response_text

        # ---------------------------------------------------------------------
        # Token-level logprob scoring for the generated continuation
        # ---------------------------------------------------------------------
        gen_token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:].detach().cpu()
        try:
            gen_logps = compute_token_logprobs_streaming(
                model, tokenizer,
                prompt_ids=inputs["input_ids"],
                gen_ids=gen_token_ids,
                device=device,
                score_dtype=args.score_dtype,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM][rank {rank}] logprob pass at idx={i}; skipping confidences.")
            gen_logps = torch.empty(0)
        except Exception as e:
            print(f"[WARN][rank {rank}] logprob pass failed at idx={i}: {e}")
            gen_logps = torch.empty(0)

        # ---------------------------------------------------------------------
        # Segment-level "confidence" (heuristic)
        # ---------------------------------------------------------------------
        # We take the text before '</think>' (if any), split it into segments by blank lines,
        # and summarize token logprobs for each segment.
        #
        # Important limitations:
        # - Segment boundaries are defined in *text space* and mapped to *token space* by re-tokenization.
        # - This mapping may be imperfect if whitespace/tokenization differs between runs.
        text_before_think = response_text.split('</think>')[0]
        text_segments = rsplit(r'\n\n+', text_before_think)
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

        # ---------------------------------------------------------------------
        # Hidden-state dump (best-effort, rescueable)
        # ---------------------------------------------------------------------
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
        except torch.cuda.OutOfMemoryError:
            # We intentionally do not fail the whole run; next run can rescue from the shard response.
            print(f"[WARN][rank {rank}] hidden save OOM at idx={i}; will rely on rescue next run.")
            _clear_cuda()
        except Exception as he:
            print(f"[WARN][rank {rank}] hidden save failed at idx={i}: {he} (rescueable next run)")

        # ---------------------------------------------------------------------
        # Persist result entry (append-only)
        # ---------------------------------------------------------------------
        result_entry = {
            "idx": i,
            "question": q.get("problem", ""),
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences,
        }
        with open(shard_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

        # Cleanup large tensors/objects and clear caches.
        del output, inputs, gen_token_ids, gen_logps
        _clear_cuda()
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


# ------------------------
# Main
# ------------------------

def main():
    """
    Parse CLI args and launch either:
    - single-process inference (world_size=1)
    - multi-process inference using mp.spawn (world_size>1)

    Notes on `--num_gpus`
    - This should match the number of visible GPUs in CUDA_VISIBLE_DEVICES.
    - Rank mapping: rank 0..world_size-1 maps to cuda:0..cuda:world_size-1 in the visible set.
    """
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
    parser.add_argument('--hs_device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--score_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
