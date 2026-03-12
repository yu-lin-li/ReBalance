# -*- coding: utf-8 -*-
"""
check.py

Purpose
-------
Evaluate a JSONL generation file against a dataset split, producing:
  - Accuracy (any-of-n sampling correctness)
  - Pass@k (when multiple samples are present)
  - Batch-wise accuracy statistics across the first k samples
  - (Optional) token/length statistics (enabled in infer_2)

This script assumes you have the same project structure used during inference:
  - utils/ (grader, parser, data loader, etc.)
  - prompts/ (prompt templates; imported dynamically by prompt_type/data_name)

Inputs
------
1) Dataset:
   load_data(data_name, split, data_dir) must return an indexable list of examples.

2) Generation JSONL:
   Each line is a dict containing at least:
     - generated_responses: list[str] (or compatible nested structures; see infer_2 helpers)

Outputs
-------
Printed metrics to stdout.
Additionally, infer_2 writes a JSON file listing the IDs of incorrect examples:
  wrong_ids_{data_name}_{split}.json
stored next to generation_path.

Usage (example)
---------------
python check.py \
  --model_name_or_path /path/to/model \
  --data_dir ./Data \
  --data_name math \
  --split test \
  --generation_path ./outputs/xxx.jsonl \
  --k 8
"""

import json
from transformers import AutoTokenizer

import re
import importlib.util
import os
import argparse

import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
import pdb


def parse_list(arg: str):
    """Parse a comma-separated CLI value into a list of strings."""
    return arg.split(',')


def save_completions(completions, filepath: str):
    """Serialize completions to a pickle file (kept for compatibility with earlier workflows)."""
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./Data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--generation_path", default="test", type=str)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    args = parser.parse_args()
    return args


def get_conversation_prompt_by_messages(tokenizer, messages):
    """
    Render chat messages into a single prompt string using the tokenizer's chat template.
    This is a utility function for compatibility with chat-style models.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def get_three_prompt(prompt_type: str, data_name: str):
    """
    Load prompt components from ./prompts/{prompt_type}/{data_name}.py.

    Required symbols in the module:
      - system_prompt
      - few_shot_prompt
      - question_format
    """
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")

    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")

    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def read_jsonl(file_path: str):
    """Minimal JSONL reader (kept local to avoid changing upstream utils)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def infer(args):
    """
    Original evaluation path (expects generated_responses to be a list[str]).

    This function is preserved for compatibility, but the entrypoint uses infer_2()
    since it handles multiple output formats more robustly.
    """
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)

    print("llm generate done")
    print(len(file_outputs))

    pass_at_k_list = []
    k = args.k

    correct_cnt = 0
    for i in tqdm(range(len(file_outputs)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']

        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)

    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")

    response_length = []
    token_num = []
    wait_num = []
    alt_num = []

    test_num = len(file_outputs)
    correct_num = 0
    for data in file_outputs:
        response_length.append(len(data['generated_responses'][0].split()))
        tokens_response_len = len(tokenizer(data['generated_responses'][0])['input_ids'])
        token_num.append(tokens_response_len)

    avg_response_length = sum(response_length) / test_num
    avg_token_num = sum(token_num) / test_num

    print("length:", avg_response_length)
    print('token_num:', avg_token_num)


def infer_2(args):
    """
    Robust evaluation path.

    Enhancements vs infer():
      - Extracts text from multiple output structures (str / dict / nested lists).
      - Reports:
          * any-of-n accuracy
          * pass@k
          * batch-wise accuracy across the first k samples
          * per-question avg@k (mean/var/sd)
      - Writes wrong IDs to a JSON file beside generation_path.
      - Computes token/length statistics per batch and aggregates their variance across batches.

    The core correctness logic still relies on:
      - parse_ground_truth()
      - extract_answer()
      - check_is_correct()
    """
    import os, json, re
    from math import comb

    # -------- helpers --------
    def _extract_text_from_generated(g):
        """Best-effort extraction of raw text from common generation formats."""
        if isinstance(g, str):
            return g
        if isinstance(g, dict):
            for key in ("text", "content", "generated_response", "generated_text",
                        "output", "output_text", "message", "response"):
                v = g.get(key, None)
                if isinstance(v, str):
                    return v
        if isinstance(g, list) and g:
            for item in g:
                t = _extract_text_from_generated(item)
                if t:
                    return t
        return ""

    def _get_response_text(data):
        """
        Compatibility loader for earlier logging formats.
        - Preferred: data['generated_responses'][0]
        - Fallback: data['outputs'][0]['outputs'][0]['text'] (vLLM-style)
        """
        if isinstance(data, dict) and data.get('generated_responses'):
            first_gen = data['generated_responses'][0]
            return _extract_text_from_generated(first_gen)
        if isinstance(data, dict) and data.get('outputs'):
            o0 = data['outputs'][0] if data['outputs'] else None
            if isinstance(o0, dict) and o0.get('outputs'):
                oo0 = o0['outputs'][0]
                if isinstance(oo0, dict) and isinstance(oo0.get('text'), str):
                    return oo0['text']
        return ""

    # -------- load --------
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)

    print("llm generate done")
    print(len(file_outputs))

    # -------- correctness & pass@k --------
    pass_at_k_list = []
    avg_at_k_list = []   # per-question avg@k
    k = args.k

    correct_cnt = 0
    wrong_ids = []  # identifiers for incorrect examples

    # Batch-wise correctness (across the first k generated samples)
    batch_correct_counts = [0] * k
    batch_total_counts = [0] * k

    for i in tqdm(range(len(file_outputs)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)

        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(gr, args.data_name) for gr in generated_responses]
        is_correct_list = [check_is_correct(ga, gt_ans) for ga in generated_answers]
        is_correct = any(is_correct_list)

        if is_correct:
            correct_cnt += 1
        else:
            qid = d.get('id', i) if isinstance(d, dict) else i
            wrong_ids.append(qid)

        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

        # Per-question avg@k (prefix average over first k samples)
        n_samples = len(is_correct_list)
        k_eff = min(k, n_samples) if n_samples > 0 else 0
        if k_eff > 0:
            avg_k_i = sum(is_correct_list[:k_eff]) / k_eff
        else:
            avg_k_i = 0.0
        avg_at_k_list.append(avg_k_i)

        # Batch-wise aggregation (batch j = j-th sample among generated_responses)
        for j in range(k_eff):
            if is_correct_list[j]:
                batch_correct_counts[j] += 1
            batch_total_counts[j] += 1

        # pass@k (standard definition for multiple samples)
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)

    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")

    # -------- batch accuracy variance (across the first k samples) --------
    batch_acc = []
    for j in range(k):
        if batch_total_counts[j] > 0:
            batch_acc.append(batch_correct_counts[j] / batch_total_counts[j])

    if batch_acc:
        n_b = len(batch_acc)
        mean_batch_acc = sum(batch_acc) / n_b
        var_batch_acc = sum((x - mean_batch_acc) ** 2 for x in batch_acc) / n_b
        sd_batch_acc = var_batch_acc ** 0.5

        print(f"Batch acc list (k={n_b}): {batch_acc}")
        print(f"Avg accuracy over {n_b} batches: {mean_batch_acc:.4f}")
        print(f"Accuracy variance across batches: {var_batch_acc:.6f}")
        print(f"Accuracy SD across batches: {sd_batch_acc:.6f}")

    # Per-question avg@k variance
    if avg_at_k_list:
        n_q = len(avg_at_k_list)
        mean_avg_k_q = sum(avg_at_k_list) / n_q
        var_avg_k_q = sum((x - mean_avg_k_q) ** 2 for x in avg_at_k_list) / n_q
        sd_avg_k_q = var_avg_k_q ** 0.5
        print(f"[Per-question avg@{k}] mean={mean_avg_k_q:.4f}, var={var_avg_k_q:.6f}, sd={sd_avg_k_q:.6f}")

    # -------- save wrong IDs to JSON --------
    out_dir = os.path.dirname(os.path.abspath(getattr(args, "generation_path", "wrong_ids.json")))
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"wrong_ids_{getattr(args, 'data_name', 'dataset')}_{getattr(args, 'split', 'split')}.json"
    out_path = os.path.join(out_dir, out_name)
    wrong_payload = {
        "data_name": getattr(args, "data_name", None),
        "split": getattr(args, "split", None),
        "count": len(wrong_ids),
        "ids": wrong_ids,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrong_payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrong IDs saved to: {out_path} (count={len(wrong_ids)})")

    # -------- token/length stats (batch-wise) --------
    # For each sample, compute stats per batch j (0..k-1), then
    # aggregate over questions -> get mean for each batch -> compute variance across batches.
    batch_word_sum = [0.0] * k
    batch_token_sum = [0.0] * k
    batch_think_token_sum = [0.0] * k
    batch_count = [0] * k

    think_found = 0       # questions where at least one sample has a </think> boundary
    fallback_full = 0     # questions where no sample has </think> (full text used)

    test_num = len(file_outputs)

    for data in file_outputs:
        gens = None
        if isinstance(data, dict) and "generated_responses" in data:
            gens = data["generated_responses"]

        sample_has_text = False
        sample_has_think = False

        if isinstance(gens, list) and len(gens) > 0:
            n_samples = len(gens)
            k_eff = min(k, n_samples)

            for j in range(k_eff):
                text = _extract_text_from_generated(gens[j])
                if not text:
                    continue

                sample_has_text = True

                # word count
                wlen = len(text.split())
                batch_word_sum[j] += wlen

                # full token count
                tlen = len(tokenizer(text)["input_ids"])
                batch_token_sum[j] += tlen

                # think-boundary token count (prefix until </think>, otherwise full text)
                lower = text.lower()
                idx = lower.find("</think>")
                if idx == -1:
                    idx = lower.find("&lt;/think&gt;")

                if idx != -1:
                    think_text = text[:idx]
                    sample_has_think = True
                else:
                    think_text = text

                t_think = len(
                    tokenizer(think_text, add_special_tokens=False)["input_ids"]
                ) if think_text else 0
                batch_think_token_sum[j] += t_think

                batch_count[j] += 1

        else:
            # fallback: treat a single text as batch 0
            text = _get_response_text(data)
            if text:
                sample_has_text = True

                wlen = len(text.split())
                tlen = len(tokenizer(text)["input_ids"])

                lower = text.lower()
                idx = lower.find("</think>")
                if idx == -1:
                    idx = lower.find("&lt;/think&gt;")

                if idx != -1:
                    think_text = text[:idx]
                    sample_has_think = True
                else:
                    think_text = text

                t_think = len(
                    tokenizer(think_text, add_special_tokens=False)["input_ids"]
                ) if think_text else 0

                batch_word_sum[0] += wlen
                batch_token_sum[0] += tlen
                batch_think_token_sum[0] += t_think
                batch_count[0] += 1

        if sample_has_text:
            if sample_has_think:
                think_found += 1
            else:
                fallback_full += 1

    # per-batch means (averaged over questions)
    batch_mean_len = []
    batch_mean_token = []
    batch_mean_think_token = []

    for j in range(k):
        if batch_count[j] > 0:
            batch_mean_len.append(batch_word_sum[j] / batch_count[j])
            batch_mean_token.append(batch_token_sum[j] / batch_count[j])
            batch_mean_think_token.append(batch_think_token_sum[j] / batch_count[j])

    def _mean_var(xs):
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)  # population variance
        return m, v

    avg_response_length, var_response_length = _mean_var(batch_mean_len)
    avg_token_num, var_token_num = _mean_var(batch_mean_token)
    avg_think_token_num, var_think_token_num = _mean_var(batch_mean_think_token)

    print(batch_mean_token)
    print("length:", avg_response_length, "var:", var_response_length)
    print("token_num:", avg_token_num, "var:", var_token_num)
    print("sd", var_token_num**0.5)
    print("think_token_num:", avg_think_token_num, "var:", var_think_token_num)
    print(
        f"think blocks found by </think>: {think_found}/{test_num} (fallback_full={fallback_full})"
    )


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    infer_2(args)
