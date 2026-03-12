# -*- coding: utf-8 -*-
"""
transformer_inference_steer_dp.py

Purpose
-------
Distributed inference (data-parallel) for a patched Qwen2-style causal LM that supports
*steering* via a custom API:

  - model.set_steering_flag(...)
  - model.start_new_round()

For each example in a JSONL test split, the script:
  1) Builds a chat-style prompt.
  2) Performs token-by-token sampling (top-p + temperature) while tracking per-step logprobs.
  3) Decodes the generated text.
  4) Computes "sentence/segment confidences" by:
       - taking the text before </think>,
       - splitting by blank lines,
       - mapping each segment to its token span,
       - averaging step logprobs within the span, then exponentiating the mean logprob
         to obtain an average token probability proxy.
  5) Writes one JSON record per example into a rank-specific shard file and supports
     robust resumption via checkpointing.

Notes on reproducibility & performance
-------------------------------------
- Sampling is implemented explicitly (instead of `model.generate`) to avoid retaining
  per-step scores in memory and to make step-level logprob tracking straightforward.
- The script sets several environment variables to reduce logging noise across spawned
  processes.
- Attention backend selection prefers FlashAttention-2 when available, otherwise
  falls back to PyTorch SDPA.

This file intentionally keeps the original runtime behavior. Only comments/docstrings
have been rewritten for open-source readability.
"""

import os
import json
import argparse
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from re import split as rsplit
import torch.multiprocessing as mp
import random
import numpy as np

from modeling_utils.modeling_qwen2_dynamic_3D import Qwen2ForCausalLM

import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

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

try:
    np.seterr(all="ignore")
except Exception:
    pass


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    run_prefix = (args.run_id.strip() + "_") if args.run_id else ""
    base_name = f"{run_prefix}steer_temp{args.temperature}_maxlen{args.max_generated_tokens}"
    return output_dir, base_name


def load_existing_indices(shard_file):
    existing_idx = set()
    existing_q = set()
    if os.path.exists(shard_file):
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    elif "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    continue
    return existing_idx, existing_q


@torch.no_grad()
def top_p_sampling_step(last_logits, temperature: float, top_p: float):
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    logits = last_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumsum > top_p)
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_sorted_idx)

    chosen_prob = sorted_probs.gather(-1, next_sorted_idx)
    next_logprob = torch.log(chosen_prob + 1e-12).item()

    return next_token, next_logprob


def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    try:
        use_fa2 = False
        try:
            from transformers.utils import is_flash_attn_2_available
            use_fa2 = bool(is_flash_attn_2_available())
            if use_fa2:
                import flash_attn_2_cuda  # noqa: F401
        except Exception:
            use_fa2 = False

        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            if use_fa2:
                model.config.attn_implementation = "flash_attention_2"
                print("[INFO] Using flash_attention_2")
            else:
                model.config.attn_implementation = "sdpa"
                torch.backends.cuda.matmul.allow_tf32 = True
                if torch.cuda.is_available():
                    torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=True,
                    )
                print("[INFO] Using PyTorch SDPA (flash preferred, mem_efficient fallback)")
    except Exception as e:
        print(f"[WARN] attention backend setup failed: {e}")

    model.eval()
    return model, tokenizer, dtype


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
):
    device = input_ids.device
    token_logprobs = []
    generated = []

    past_key_values = None

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=(dtype in (torch.float16, torch.bfloat16)), dtype=dtype
    ):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=None)
        past_key_values = outputs.past_key_values

    last_token = None
    for _ in range(max_new_tokens):
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=(dtype in (torch.float16, torch.bfloat16)), dtype=dtype
        ):
            if last_token is None:
                logits = outputs.logits[:, -1, :]
            else:
                out = model(
                    input_ids=last_token,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

        next_token, next_logprob = top_p_sampling_step(logits, temperature, top_p)
        token_logprobs.append(next_logprob)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            generated.append(next_token.item())
            break

        generated.append(next_token.item())
        last_token = next_token

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), []

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs


def worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "test.jsonl")
    questions = read_jsonl(dataset_path)

    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank}.jsonl")

    existing_idx, existing_q = load_existing_indices(shard_file)

    model, tokenizer, dtype = load_model_and_tokenizer(args, device)

    steer_vector = torch.load(args.steer_vector_path, map_location="cpu").to(device, dtype=dtype)

    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
        dyn_hparams={
            "q25c": args.dyn_q25c,
            "q75c": args.dyn_q75c,
            "low_val_1": args.dyn_low_val_1,
            "q25v": args.dyn_q25v,
            "q75v": args.dyn_q75v,
            "low_val_2": args.dyn_low_val_2,
            "high_val_2": args.dyn_high_val_2,
        },
    )

    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)

    for i in my_indices:
        q = questions[i]

        if (i in existing_idx) or (q.get("problem") in existing_q):
            pbar.update(1)
            continue

        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q.get("problem", "")},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                gen_ids, step_logprobs = sample_with_tracking(
                    model=model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
                    max_new_tokens=args.max_generated_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    dtype=dtype,
                )
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids], dim=0)
                response_text = tokenizer.decode(
                    full_ids[inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i} : {q.get('problem','')[:80]}...")
            torch.cuda.empty_cache()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        text_before_think = response_text.split("</think>")[0]
        text_segments = rsplit(r"\n\n+", text_before_think)

        confidences = []
        start = 0
        gen_token_ids = gen_ids.tolist()

        for segment in text_segments:
            seg_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
            end = start + len(seg_ids)
            if end > len(gen_token_ids):
                break
            if end > start:
                seg_logprobs = step_logprobs[start:end]
                avg_logprob = sum(seg_logprobs) / max(1, len(seg_logprobs))
                confidences.append(float(torch.exp(torch.tensor(avg_logprob)).item()))
            start = end

        result = {
            "idx": i,
            "question": q.get("problem", ""),
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences,
        }

        with open(shard_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        del inputs, gen_ids
        torch.cuda.empty_cache()
        gc.collect()

        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


def main():
    MODEL_DYN_DEFAULTS = {
        "DeepSeek-R1-Distill-Qwen-1.5B": {
            "dyn_q25c": 0.662293,
            "dyn_q75c": 0.94805,
            "dyn_low_val_1": -1.02,
            "dyn_q25v": 0.000560,
            "dyn_q75v": 0.011597,
            "dyn_low_val_2": -1.91,
            "dyn_high_val_2": 0.1,
        },
        "DeepSeek-R1-Distill-Qwen-7B": {
            "dyn_q25c": 0.666017,
            "dyn_q75c": 0.927745,
            "dyn_low_val_1": -1.19,
            "dyn_q25v": 0.000488,
            "dyn_q75v": 0.009931,
            "dyn_low_val_2": -2.34,
            "dyn_high_val_2": 0.1,
        },
        "QwQ-32B": {
            "dyn_q25c": 0.700670,
            "dyn_q75c": 0.917506,
            "dyn_low_val_1": -1.31,
            "dyn_q25v": 0.000386,
            "dyn_q75v": 0.007279,
            "dyn_low_val_2": -2.73,
            "dyn_high_val_2": 0.1,
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--steer_vector_path", type=str, required=True)
    parser.add_argument("--steer_layer", type=int, default=22)
    parser.add_argument("--steer_coef", type=float, default=1.0)

    parser.add_argument("--dyn_q25c", type=float, default=None)
    parser.add_argument("--dyn_q75c", type=float, default=None)
    parser.add_argument("--dyn_low_val_1", type=float, default=None)
    parser.add_argument("--dyn_q25v", type=float, default=None)
    parser.add_argument("--dyn_q75v", type=float, default=None)
    parser.add_argument("--dyn_low_val_2", type=float, default=None)
    parser.add_argument("--dyn_high_val_2", type=float, default=None)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_generated_tokens", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--run_id", type=str, default="")
    args = parser.parse_args()

    selected = None
    for k in ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "QwQ-32B"):
        if k in args.model_name_or_path:
            selected = MODEL_DYN_DEFAULTS[k]
            break

    if selected is not None:
        for name in (
            "dyn_q25c",
            "dyn_q75c",
            "dyn_low_val_1",
            "dyn_q25v",
            "dyn_q75v",
            "dyn_low_val_2",
            "dyn_high_val_2",
        ):
            if getattr(args, name) is None:
                setattr(args, name, float(selected[name]))

    set_seed(42)

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()