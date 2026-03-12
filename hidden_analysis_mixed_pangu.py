# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import Optional

import torch
import torch_npu
from torch.utils.data import TensorDataset, ConcatDataset

# =========================
# Token list (a hit means the token is considered a lexicon match)
# Preserved from the keyword script with original pattern/regex construction
# =========================
# ! v0 (raw)
LEXICON_BASE = [
    # Low-confidence terms for uncertain/reflection style and calculation/unknown responses
    "alternatively", "alternative", "another", "perhaps", "maybe", "wait", "but",
    "think again", "make sure", "just to ensure", "there any other", "some other",
    "should consider", "about whether", "if they have", "i was", "any errors",
    "or something", "let me check", "hold on", "double check", "however",
    "confusing", "differently", "careful", "sometimes", "alternate"
]

# ! v1 (expanded)
# LEXICON_BASE = [
#     "alternatively", "alternative", "another", "perhaps", "maybe", "wait", "but",
#     "think again", "make sure", "just to ensure", "there any other", "some other",
#     "should consider", "about whether", "if they have", "i was", "any errors",
#     "or something", "let me check", "hold on", "double check", "however",
#     "confusing", "differently", "careful", "sometimes", "alternate",
#     # ! newly added
#     "Hum", "Hummm", "check", "perhaps", "double-check", "recall", "also think", "remember",
#     "let me ensure", "be certain", "but what if", "I'm not sure", "could it be", "is that right"
# ]


def _normalize_text(s: str) -> str:
    # Normalize quotes and dash/hyphen variants
    return (s or "")\
        .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')\
        .replace("–", "-").replace("—", "-").replace("-", "-")  # en/em/nb-hyphen

def _token_pattern(tok: str) -> str:
    """
    Convert a word/phrase into a more robust regex fragment:
    - Collapse whitespace into \s+
    - Allow forms like 'double[-\s]check' with spaces or hyphen
    - Add optional variants for common endings
    """
    tok = _normalize_text(tok).lower().strip()
    parts = re.split(r"\s+", tok)

    def word2regex(w: str) -> str:
        if w == "verify":
            return r"verif(?:y|ies|ied|ying|ication(?:s)?)"
        if w == "alternative":
            return r"alternative(?:s|ly)?"
        if w == "confusing":
            return r"confus(?:e|es|ed|ing)"
        if w == "differently":
            return r"different(?:ly)?"
        if w == "careful":
            return r"careful(?:ly)?"
        if w == "sometimes":
            return r"sometimes?"
        if w == "alternate":
            return r"alternat(?:e|es|ed|ing)"
        
        # ! newly added (v1)
        if w == "check":
            return r"check(?:s|ed|ing)?"
        if w == "recall":
            return r"recall(?:s|ed|ing)?"
        if w == "remember":
            return r"remember(?:s|ed|ing)?"
    
        # Use strict token form by default
        return re.escape(w)

    if len(parts) >= 2:
        pieces = [word2regex(p) for p in parts]
        between = r"(?:[-\s]+)"  # whitespace or hyphen
        body = between.join(pieces)
    else:
        body = word2regex(parts[0])

    return rf"\b{body}\b"

def _compile_lexicon_regex(lexicon_base):
    expanded = []
    for term in lexicon_base:
        t = term.strip()
        if not t:
            continue
        expanded.append(_token_pattern(t))
    pattern = "|".join(expanded)
    return re.compile(pattern, flags=re.IGNORECASE)

LEXICON_RE = _compile_lexicon_regex(LEXICON_BASE)

def has_lexicon_hit(text: str) -> int:
    return 1 if LEXICON_RE.search(_normalize_text(text).lower()) else 0


# ----------------------
# Utilities
# ----------------------
def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def split_segments_for_conf(resp_text: str):
    """
    Keep only content before </think>, then split by consecutive blank lines.
    """
    text = resp_text or ""
    if "[unused17]" in text:
        text = text.split("[unused17]")[0]
    segs = re.split(r"\n\n+", text)
    segs = [s.strip() for s in segs if len(s.strip()) > 0]
    return segs


# ----------------------
# Dataset builder: (feature, label_mixed)
# label_mixed = 1 when (lexicon hit) OR (conf < threshold), otherwise 0
# Alignment strategy is aligned with both upstream scripts and uses minimal merging:
#   - V = hidden['step'].shape[0]
#   - C_raw = len(sentence_confidences)
#   - Expected V == C_raw - expected_offset
#   - Lexicon segments come from generated_responses before </think>
#   - Final aligned length m = min(V, len(lex_labels), len(confs_aligned))
# ----------------------
def build_dataset_from_layer_mixed(
    layer_id: int,
    json_item: dict,
    hidden_path: str,
    threshold: float = 0.7,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[TensorDataset]:
    confs_raw = json_item.get("sentence_confidences", [])
    if not confs_raw or len(confs_raw) <= expected_offset:
        if verbose:
            print("⛔ Skip: sentence_confidences is empty or too short")
        return None

    if not os.path.exists(hidden_path):
        if verbose:
            print(f"❌ Skip: file does not exist {hidden_path}")
        return None

    try:
        data = torch.load(hidden_path, map_location='cpu', weights_only=False)
    except Exception as e:
        if verbose:
            print(f"❌ Failed to load {hidden_path}: {e}")
        return None

    if layer_id not in data:
        if verbose:
            print(f"❌ Skip: layer {layer_id} does not exist in the file")
        return None

    tensor = data[layer_id]  # [step_num, hidden_dim]
    V = tensor.shape[0]
    C_raw = len(confs_raw)

    if V != C_raw - expected_offset:
        if verbose:
            print(f"❌ Vector count {V} does not match confidence count {C_raw} with offset {expected_offset}; skip")
        return None

    # 1) Lexicon labels
    responses_text = (json_item.get("generated_responses") or [""])[0]
    segs = split_segments_for_conf(responses_text)
    if expected_offset > 0 and len(segs) >= expected_offset:
        segs = segs[expected_offset:]
    lex_labels = [float(has_lexicon_hit(s)) for s in segs]  # 0/1

    # 2) Confidence labels
    confs = confs_raw[expected_offset:]
    # Truncate to avoid exceeding V
    confs = confs[:V]
    conf_labels = [1.0 if float(c) < threshold else 0.0 for c in confs]

    # 3) Align length m
    m = min(V, len(lex_labels), len(conf_labels))
    if m <= 0:
        if verbose:
            print("⛔ Aligned length is 0; skip")
        return None

    features = tensor[:m].to(dtype=torch.float32)
    lex_labels_t = torch.tensor(lex_labels[:m], dtype=torch.float32)
    conf_labels_t = torch.tensor(conf_labels[:m], dtype=torch.float32)

    # 4) Mixed rule: 1 when lexicon hit OR low confidence
    labels_mixed = torch.clamp(lex_labels_t + conf_labels_t, max=1.0)

    return TensorDataset(features, labels_mixed)


def batch_build_all_mixed(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    threshold: float = 0.7,
    max_files: int = 100,
    expected_offset: int = 1,
    verbose: bool = True
) -> ConcatDataset:
    data_json = read_jsonl(jsonl_path)
    datasets = []
    total = min(max_files, len(data_json))
    print(total)
    kept = 0
    for i in range(total):
        if i < 10:
            hidden_path = os.path.join(hidden_dir, f"hidden_  {i:d}.pt")
        elif i < 100:
            hidden_path = os.path.join(hidden_dir, f"hidden_ {i:d}.pt")
        else:
            hidden_path = os.path.join(hidden_dir, f"hidden_{i:d}.pt")
        print(hidden_path)
        ds = build_dataset_from_layer_mixed(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            threshold=threshold,
            expected_offset=expected_offset,
            verbose=verbose
        )
        if ds is not None:
            datasets.append(ds)
            kept += len(ds)
        else:
            if verbose:
                print(f"[skip] index={i}")
        if (i + 1) % 50 == 0 and verbose:
            print(f"[progress] processed {i+1}/{total}, collected samples: {kept}")

    if not datasets:
        raise RuntimeError("❌ No usable dataset was successfully built")
    merged = ConcatDataset(datasets)
    if verbose:
        print(f"\n🎉 Merge complete, total samples {len(merged)}, dataset files used {len(datasets)} / {total}")
    return merged


# ----------------------
# Unique steer: S = mean(hit) - mean(nonhit)
# ----------------------
def build_steer_vector_mean_only(merged_dataset: ConcatDataset) -> torch.Tensor:
    hits = []
    nonhits = []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]   # feat: [hidden_dim], y: {0.,1.}
        if int(y.item()) == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)

    if len(hits) == 0 or len(nonhits) == 0:
        raise RuntimeError("Need both hit (1) and miss (0) samples to compute the direction vector.")

    mu_hit = torch.stack(hits, dim=0).mean(dim=0)
    mu_non = torch.stack(nonhits, dim=0).mean(dim=0)
    S = mu_hit - mu_non  # No normalization
    return S, len(hits), len(nonhits)


# ----------------------
# Main (CLI)
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build steer vector (mixed): mean(H_hit) - mean(H_nonhit), hit = (lexicon OR conf<threshold)"
    )
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold: conf < threshold is treated as low confidence")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1,
                        help="Expected V = len(sentence_confidences) - expected_offset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    merged = batch_build_all_mixed(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        threshold=args.threshold,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=True
    )

    S, n_hit, n_non = build_steer_vector_mean_only(merged)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    l2 = float(torch.linalg.norm(S))
    print(f"✅ Saved steer vector to {args.save_path}")
    print(f"   dim={S.numel()} | ||S||_2 = {l2:.6f} | hit={n_hit} | nonhit={n_non}")

if __name__ == "__main__":
    main()
