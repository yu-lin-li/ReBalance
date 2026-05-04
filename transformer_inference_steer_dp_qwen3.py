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

from modeling_utils.modeling_qwen3_dynamic_3D import Qwen3ForCausalLM

import os, warnings, logging
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
    import numpy as np
    np.seterr(all="ignore")
except Exception:
    pass
REPETITION_PENALTY = 1.18  
MODEL_DYN_HPARAMS = {
    "Qwen3-14B": {
        "dyn_q25c": 0.816174,
        "dyn_q75c": 0.965398,
        "dyn_low_val_1": -1.03,
        "dyn_q25v": 0.000149,
        "dyn_q75v": 0.003593,
        "dyn_low_val_2": -1.13,
        "dyn_high_val_2": 0.1,
    }
}


def resolve_dyn_hparams(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))

    raw = {}
    if model_basename in MODEL_DYN_HPARAMS:
        raw.update(MODEL_DYN_HPARAMS[model_basename])
    elif "Qwen3-14B" in args.model_name_or_path:
        raw.update(MODEL_DYN_HPARAMS["Qwen3-14B"])

    cli_overrides = {
        "dyn_q25c": args.dyn_q25c,
        "dyn_q75c": args.dyn_q75c,
        "dyn_low_val_1": args.dyn_low_val_1,
        "dyn_q25v": args.dyn_q25v,
        "dyn_q75v": args.dyn_q75v,
        "dyn_low_val_2": args.dyn_low_val_2,
        "dyn_high_val_2": args.dyn_high_val_2,
    }
    raw.update({k: v for k, v in cli_overrides.items() if v is not None})

    if not raw:
        return None

    return {
        "q25c": raw.get("dyn_q25c"),
        "q75c": raw.get("dyn_q75c"),
        "low_val_1": raw.get("dyn_low_val_1"),
        "q25v": raw.get("dyn_q25v"),
        "q75v": raw.get("dyn_q75v"),
        "low_val_2": raw.get("dyn_low_val_2"),
        "high_val_2": raw.get("dyn_high_val_2"),
    }


def apply_repetition_penalty(logits: torch.Tensor, history_ids: torch.Tensor, penalty: float):
    if penalty is None or penalty <= 1.0:
        return logits
    if history_ids is None:
        return logits

    if history_ids.dim() == 2:

        history_ids = history_ids[0]
    if history_ids.numel() == 0:
        return logits

    logits = logits.clone()
    unique_ids = torch.unique(history_ids)

    selected = logits.index_select(dim=-1, index=unique_ids)  # [1, U]
    neg = selected < 0

    selected = torch.where(neg, selected * penalty, selected / penalty)

    logits.scatter_(dim=1, index=unique_ids.unsqueeze(0), src=selected)
    return logits


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
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
    """
    Robust checkpoint loader. Returns a set of completed sample indices.
    Falls back to 'question' for backward compatibility.
    """
    existing_idx = set()
    existing_q = set()
    if os.path.exists(shard_file):
        with open(shard_file, 'r', encoding='utf-8') as f:
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
                    # Skip malformed line
                    continue
    return existing_idx, existing_q
@torch.no_grad()
def top_p_sampling_step(last_logits, temperature: float, top_p: float):
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

    chosen_prob = sorted_probs.gather(-1, next_sorted_idx)  # [1,1]
    next_logprob = torch.log(chosen_prob + 1e-12).item()

    return next_token, next_logprob


def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    try:
        use_fa2 = False
        try:
            from transformers.utils import is_flash_attn_2_available
            use_fa2 = bool(is_flash_attn_2_available())
            if use_fa2:
                
                import flash_attn_2_cuda
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
                        enable_mem_efficient=True
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
    dtype: torch.dtype
):
    device = input_ids.device
    token_logprobs = []
    generated = []
    past_key_values = None
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(dtype in (torch.float16, torch.bfloat16)), dtype=dtype):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=None)
        past_key_values = outputs.past_key_values
    history = input_ids.clone()
    cur_len = 0
    last_token = None
    for _ in range(max_new_tokens):
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(dtype in (torch.float16, torch.bfloat16)), dtype=dtype):
            if last_token is None:

                logits = outputs.logits[:, -1, :]
            else:

                out = model(
                    input_ids=last_token,  # [1,1]
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]
        logits = apply_repetition_penalty(logits, history, REPETITION_PENALTY)

        next_token, next_logprob = top_p_sampling_step(logits, temperature, top_p)
        token_logprobs.append(next_logprob)


        if eos_token_id is not None and next_token.item() == eos_token_id:
            generated.append(next_token.item())

            history = torch.cat([history, next_token], dim=-1)
            break

        generated.append(next_token.item())
        last_token = next_token  # [1,1]

        history = torch.cat([history, next_token], dim=-1)
        cur_len += 1

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), []

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs


def worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)

    # Output paths
    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank}.jsonl")

    # Resume from checkpoint
    existing_idx, existing_q = load_existing_indices(shard_file)

    # Load model/tokenizer per rank
    model, tokenizer, dtype = load_model_and_tokenizer(args, device)

    # Load steer vector
    steer_vector = torch.load(args.steer_vector_path, map_location="cpu").to(device, dtype=dtype)

    # Enable steering (custom API in your patched model)
    dyn_hparams = resolve_dyn_hparams(args)
    if dyn_hparams is not None:
        print(f"[rank {rank}] auto dyn_hparams")

    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
        dyn_hparams=dyn_hparams,
    )

    # Partition: each rank handles indices congruent to its rank
    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    # Logging
    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(
        total=len(my_indices),
        desc=f"Rank {rank} DP Inference",
        position=rank,      # show both progress bars
        leave=True
    )

    for i in my_indices:
        q = questions[i]

        if (i in existing_idx) or (q.get('problem') in existing_q):
            pbar.update(1)
            continue

        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q.get("problem", "")}
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
                    dtype=dtype
                )
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids], dim=0)
                response_text = tokenizer.decode(full_ids[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i} : {q.get('problem','')[:80]}...")
            torch.cuda.empty_cache()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue
        text_before_think = response_text.split('</think>')[0]
        text_segments = rsplit(r'\n\n+', text_before_think)

        confidences = []
        start = 0

        gen_token_ids = gen_ids.tolist()

        for segment in text_segments:
            seg_ids = tokenizer(segment, add_special_tokens=False)['input_ids']
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
            "sentence_confidences": confidences
        }


        with open(shard_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        del inputs, gen_ids
        torch.cuda.empty_cache()
        gc.collect()

        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


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
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--run_id', type=str, default="", help="Optional tag to separate different runs in filenames")
    parser.add_argument('--dyn_q25c', type=float, default=None)
    parser.add_argument('--dyn_q75c', type=float, default=None)
    parser.add_argument('--dyn_low_val_1', type=float, default=None)
    parser.add_argument('--dyn_q25v', type=float, default=None)
    parser.add_argument('--dyn_q75v', type=float, default=None)
    parser.add_argument('--dyn_low_val_2', type=float, default=None)
    parser.add_argument('--dyn_high_val_2', type=float, default=None)
    args = parser.parse_args()
    set_seed(42)  
    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()