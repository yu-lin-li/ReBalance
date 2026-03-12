# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import TensorDataset, ConcatDataset


# =========================
# Lexicon (hit if contains any token/phrase)
# =========================
LEXICON_BASE = [
    "alternatively", "alternative", "another", "perhaps", "maybe", "wait", "but",
    "think again", "make sure", "just to ensure", "there any other", "some other",
    "should consider", "about whether", "if they have", "i was", "any errors",
    "or something", "let me check", "hold on", "double check", "however",
    "confusing", "differently", "careful", "sometimes", "alternate"
]


def _normalize_text(s: str) -> str:
    return (s or "") \
        .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') \
        .replace("–", "-").replace("—", "-").replace("-", "-")


def _token_pattern(tok: str) -> str:
    """
    Turn a lexicon term (word/phrase) into a robust regex fragment:
      - collapse spaces into \s+
      - allow space or hyphen between phrase parts
      - add light morphological variants for a few common words
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
        return re.escape(w)

    if len(parts) >= 2:
        pieces = [word2regex(p) for p in parts]
        between = r"(?:[-\s]+)"
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


# =========================
# I/O helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def split_segments_for_conf(resp_text: str) -> List[str]:
    """
    Only consider content before </think>, then split by blank lines.
    """
    text = resp_text or ""
    if "</think>" in text:
        text = text.split("</think>")[0]
    segs = re.split(r"\n\n+", text)
    segs = [s.strip() for s in segs if len(s.strip()) > 0]
    return segs


# =========================
# Dataset builder (mixed label)
#   label_mixed:
#     - 1 if (lexicon hit) OR (conf < q25)
#     - 0 if (NOT lexicon hit) AND (conf > q75)
#     - otherwise: DROP (ignored)
#
# Alignment:
#   - features come from hidden_{i}.pt: data[layer_id][sample_id=0]['step'] -> [V, D]
#   - confidences come from json_item['sentence_confidences']
#   - requires V == len(confs_raw) - expected_offset
#   - segments from generated_responses (pre-</think>) are shifted by expected_offset
#   - we filter steps by the rule above, then return TensorDataset(features_kept, labels_kept)
# =========================
def build_dataset_from_layer_mixed(
    layer_id: int,
    json_item: Dict[str, Any],
    hidden_path: str,
    threshold: float = 0.7,
    expected_offset: int = 1,
    verbose: bool = False,
    high_threshold: Optional[float] = None,   # NEW: q75 (kept optional to avoid touching CLI)
) -> Optional[TensorDataset]:
    confs_raw = json_item.get("sentence_confidences", [])
    if not confs_raw or len(confs_raw) <= expected_offset:
        if verbose:
            print("⛔ Skip: sentence_confidences is empty or too short")
        return None

    if not os.path.exists(hidden_path):
        if verbose:
            print(f"❌ Skip: file not found {hidden_path}")
        return None

    try:
        data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if verbose:
            print(f"❌ Failed to load {hidden_path}: {e}")
        return None

    if layer_id not in data:
        if verbose:
            print(f"❌ Skip: layer {layer_id} does not exist in the file")
        return None

    layer_data = data[layer_id]
    sample_id = 0
    if sample_id not in layer_data or "step" not in layer_data[sample_id]:
        if verbose:
            print(f"❌ Skip: missing sample_id={sample_id} or step")
        return None

    tensor = layer_data[sample_id]["step"]  # [V, hidden_dim]
    if not torch.is_tensor(tensor) or tensor.dim() != 2:
        if verbose:
            print("❌ Skip: unexpected `step` tensor shape")
        return None

    V = tensor.shape[0]
    C_raw = len(confs_raw)
    if V != C_raw - expected_offset:
        if verbose:
            print(f"❌ Skip: number of vectors {V} and raw confidence count {C_raw} does not match expected offset {expected_offset} -> skip")
        return None

    # Fall back if q75 is missing: keep old behavior boundary safe.
    # (User asked to use q75; in normal runs q75 exists.)
    if high_threshold is None:
        high_threshold = 1.0

    # 1) Lexicon labels
    responses_text = (json_item.get("generated_responses") or [""])[0]
    segs = split_segments_for_conf(responses_text)
    if expected_offset > 0 and len(segs) >= expected_offset:
        segs = segs[expected_offset:]
    lex_labels = [float(has_lexicon_hit(s)) for s in segs]  # 0/1

    # 2) Confidences aligned to steps
    confs = confs_raw[expected_offset:][:V]

    # 3) Align to length m
    m = min(V, len(lex_labels), len(confs))
    if m <= 0:
        if verbose:
            print("❌ Aligned length is 0 -> skip")
        return None

    features_all = tensor[:m].to(dtype=torch.float32)

    # 4) NEW rule: keep only:
    #    - positives: lex==1 OR conf<threshold(q25)
    #    - negatives: lex==0 AND conf>high_threshold(q75)
    kept_feats = []
    kept_labels = []
    for i in range(m):
        lex = float(lex_labels[i])
        try:
            c = float(confs[i])
        except Exception:
            continue

        is_pos = (lex >= 0.5) or (c < float(threshold))
        is_neg = (lex < 0.5) and (c > float(high_threshold))

        if is_pos:
            kept_feats.append(features_all[i])
            kept_labels.append(1.0)
        elif is_neg:
            kept_feats.append(features_all[i])
            kept_labels.append(0.0)
        else:
            # middle region (q25 <= conf <= q75) and no lexicon hit -> drop
            continue

    if len(kept_labels) == 0:
        if verbose:
            print("❌ No kept steps after filtering -> skip")
        return None

    features = torch.stack(kept_feats, dim=0)
    labels_mixed = torch.tensor(kept_labels, dtype=torch.float32)
    return TensorDataset(features, labels_mixed)


def batch_build_all_mixed(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    threshold: float = 0.7,
    max_files: int = 100,
    expected_offset: int = 1,
    verbose: bool = False,
    high_threshold: Optional[float] = None,  # NEW: q75
) -> ConcatDataset:
    data_json = read_jsonl(jsonl_path)
    datasets: List[TensorDataset] = []
    total = min(max_files, len(data_json))

    for i in range(total):
        hidden_path = os.path.join(hidden_dir, f"hidden_{i}.pt")
        ds = build_dataset_from_layer_mixed(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            threshold=threshold,
            expected_offset=expected_offset,
            verbose=False,
            high_threshold=high_threshold,
        )
        if ds is not None:
            datasets.append(ds)
        elif verbose:
            print(f"[skip] index={i}")

    if not datasets:
        raise RuntimeError("❌ No usable datasets were successfully built")
    return ConcatDataset(datasets)


# =========================
# Steer vector: S = mean(hit) - mean(nonhit)
# =========================
@torch.no_grad()
def build_steer_vector_mean_only(merged_dataset: ConcatDataset) -> Tuple[torch.Tensor, int, int]:
    hits = []
    nonhits = []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]
        if int(y.item()) == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)

    if len(hits) == 0 or len(nonhits) == 0:
        raise RuntimeError("Both positive (hit=1) and negative (hit=0) samples are required to compute the difference vector.")

    mu_hit = torch.stack(hits, dim=0).mean(dim=0)
    mu_non = torch.stack(nonhits, dim=0).mean(dim=0)
    S = mu_hit - mu_non
    return S, len(hits), len(nonhits)


# =========================
# Tag-based diagonal LDA (hit vs non)
# =========================
@torch.no_grad()
def _collect_vectors_by_class(merged_dataset: ConcatDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    hits, nonhits = [], []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]
        feat = feat.view(-1).to(torch.float32)
        label = int(y.item()) if hasattr(y, "item") else int(y)
        if label == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)
    if len(hits) == 0 or len(nonhits) == 0:
        raise ValueError(f"Class imbalance: hit={len(hits)}, non={len(nonhits)}")
    return torch.stack(hits, dim=0), torch.stack(nonhits, dim=0)


@torch.no_grad()
def compute_lda_separator_and_steer(merged_dataset: ConcatDataset, ridge: float = 0.0) -> Dict[str, Any]:
    """
    Diagonal LDA approximation:
      w = (mu1 - mu0) / (var1 + var0)
      threshold t = -0.5*(w·mu1 + w·mu0),  b = -t
    Also reports pushing distance to move all hit samples across the boundary along -u_w or -S.
    """
    H_hit, H_non = _collect_vectors_by_class(merged_dataset)
    X1 = H_hit.to(torch.float64)
    X0 = H_non.to(torch.float64)

    mu1 = X1.mean(dim=0)
    mu0 = X0.mean(dim=0)
    S1 = X1.var(dim=0, unbiased=False)
    S0 = X0.var(dim=0, unbiased=False)
    if ridge > 0:
        S1 = S1 + ridge
        S0 = S0 + ridge

    w = (mu1 - mu0) / (S1 + S0 + 1e-12)
    w = w.to(torch.float64)

    m1 = (w * mu1).sum().item()
    m0 = (w * mu0).sum().item()
    t = -0.5 * (m1 + m0)
    b = -t

    S_vec = (mu1 - mu0)
    w_norm = float(torch.linalg.norm(w).item())
    S_norm = float(torch.linalg.norm(S_vec).item())
    if w_norm < 1e-12 or S_norm < 1e-12:
        return {"ok": False, "reason": "Direction norm is too small"}

    proj_hit = (X1 @ w).double()
    req = (proj_hit - t) / (w_norm + 1e-12)
    d_all_w = float(req.max().item())

    uS = S_vec / (S_norm + 1e-12)
    w_dot_uS = float((w * uS).sum().item())
    if w_dot_uS <= 0:
        d_all_S = float("inf")
        alpha_all_S = float("inf")
        d_mean_S = float("inf")
        alpha_mean_S = float("inf")
        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)
    else:
        alpha_u_all = float(((proj_hit - t) / (w_dot_uS + 1e-12)).max().item())
        alpha_all_S = alpha_u_all / (S_norm + 1e-12)
        d_all_S = alpha_u_all

        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)
        d_mean_S_u = max((m1 - t) / (w_dot_uS + 1e-12), 0.0)
        alpha_mean_S = d_mean_S_u / (S_norm + 1e-12)
        d_mean_S = d_mean_S_u

    return {
        "ok": True,
        "w_norm": w_norm,
        "S_norm": S_norm,
        "b": b,
        "threshold_t": t,
        "max_w_dot_h": float(proj_hit.max().item()),
        "d_all_w": d_all_w,
        "d_all_S": d_all_S,
        "alpha_all_S": alpha_all_S,
        "d_mean_w": float((m1 - t) / (w_norm + 1e-12) if w_norm > 0 else float("inf")),
        "d_mean_S": d_mean_S,
        "alpha_mean_S": alpha_mean_S,
        "w_dot_uS": w_dot_uS,
    }


# =========================
# Global confidence quartiles (coarse; scan JSONL)
# =========================
def compute_confidence_quartiles(jsonl_path: str) -> Tuple[Optional[float], Optional[float], int]:
    import math

    KEYS_DIRECT = ("confidence", "conf", "score")
    KEYS_LIST = ("sentence_confidences", "confidences", "scores", "sentence_scores", "step_confidences")

    def _collect_from_obj(o, sink: List[float]):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = str(k).lower()
                if lk in KEYS_DIRECT and isinstance(v, (int, float)) and math.isfinite(v):
                    sink.append(float(v))
                elif lk in KEYS_LIST and isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, (int, float)) and math.isfinite(x):
                            sink.append(float(x))
                        elif isinstance(x, dict):
                            for kk in KEYS_DIRECT:
                                if kk in x and isinstance(x[kk], (int, float)) and math.isfinite(x[kk]):
                                    sink.append(float(x[kk]))
                if isinstance(v, (dict, list, tuple)):
                    _collect_from_obj(v, sink)
        elif isinstance(o, (list, tuple)):
            for x in o:
                _collect_from_obj(x, sink)

    confidences: List[float] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _collect_from_obj(obj, confidences)
    except FileNotFoundError:
        return None, None, 0

    if not confidences:
        return None, None, 0

    t = torch.tensor(confidences, dtype=torch.float64)
    q25 = torch.quantile(t, 0.25).item()
    q75 = torch.quantile(t, 0.75).item()
    return q25, q75, len(confidences)


@torch.no_grad()
def compute_global_diff_quartiles(jsonl_path: str, expected_offset: int = 1) -> Tuple[Optional[float], Optional[float], int]:
    """
    v = ((a - b)^2) / 4.0, where a=confs[j], b=confs[j-1].
    Reports v25 and v75 (and N).
    """
    data_json = read_jsonl(jsonl_path)
    diffs: List[float] = []
    for item in data_json:
        confs = item.get("sentence_confidences") or []
        if not confs or len(confs) <= expected_offset:
            continue
        for j in range(expected_offset, len(confs)):
            try:
                a = float(confs[j])
                b = float(confs[j - 1])
            except Exception:
                continue
            d = ((a - b) ** 2) / 4.0
            if torch.isfinite(torch.tensor(d)):
                diffs.append(d)

    if not diffs:
        return None, None, 0

    t = torch.tensor(diffs, dtype=torch.float64)
    v25 = t.quantile(0.25).item()
    v75 = t.quantile(0.75).item()
    return v25, v75, int(t.numel())


def main():
    parser = argparse.ArgumentParser(
        description="Hidden-state analysis: build mixed-label dataset, save steer vector, run tag-LDA, and print global confidence stats."
    )
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="(deprecated) Ignored. Threshold is auto-set to global q25 of confidence.")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1, help="Expected: V == len(sentence_confidences) - expected_offset")
    parser.add_argument("--verbose", action="store_true", help="Print reasons for skipped files (default: off)")
    args = parser.parse_args()

    # Auto thresholds: use global q25/q75 of confidence (computed from jsonl).
    q25, q75, n_conf = compute_confidence_quartiles(args.jsonl_path)
    auto_threshold = q25 if q25 is not None else 0.7
    auto_high_threshold = q75 if q75 is not None else 1.0

    # 1) Build dataset & steer vector
    merged = batch_build_all_mixed(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        threshold=auto_threshold,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=args.verbose,
        high_threshold=auto_high_threshold,
    )

    S, n_hit, n_non = build_steer_vector_mean_only(merged)

    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    dim = S.numel()
    l2 = float(torch.linalg.norm(S).item())
    print(f"✅ Saved steer vector to {args.save_path}")
    print(f"   dim={dim} | ||S||_2 = {l2:.6f} | hit={n_hit} | nonhit={n_non}")

    # 2) Tag LDA block
    lda = compute_lda_separator_and_steer(merged)
    if lda.get("ok", False):
        print("\n—— LDA(diagonal) separator & steer")
        print(f"  • ||w||_2 = {lda['w_norm']:.6f} | ||S||_2 = {lda['S_norm']:.6f}")
        print(f"  • Decision threshold t = -b = {lda['threshold_t']:.6f} | max_i(w·h_i) = {lda['max_w_dot_h']:.6f}")
        print("  • Along -u_w:")
        print(f"      d_all_w  = {lda['d_all_w']:.6f}")
        print("  • Along -S:")
        if lda["d_all_S"] == float("inf"):
            print("      Angle between -S and w is too large; moving along -S cannot reduce the discriminant. Use -w instead or check the definition of S.")
        else:
            print(f"      d_all_S  = {lda['d_all_S']:.6f}   | alpha_all_S = {lda['alpha_all_S']:.6f}")
            print("  • Mean-only crossing:")
            print(f"      d_mean_w = {lda['d_mean_w']:.6f} | d_mean_S = {lda['d_mean_S']:.6f} | alpha_mean_S = {lda['alpha_mean_S']:.6f}")
    else:
        print(f"\n⚠️ Tag-based LDA separation failed: {lda.get('reason','unknown')}")

    # 3) Global quartiles
    v25, v75, n_v = compute_global_diff_quartiles(args.jsonl_path, expected_offset=args.expected_offset)
    if n_conf > 0 and q25 is not None and q75 is not None:
        print(f"   [global] confidence quantiles: q25 = {q25:.6f} | q75 = {q75:.6f} | N = {n_conf}")
    else:
        print("   [global] confidence quantiles: q25 = nan | q75 = nan | N = 0")

    if n_v > 0 and v25 is not None and v75 is not None:
        print(f"   [global] confidence quantiles: v25 = {v25:.6f} | v75 = {v75:.6f} | N = {n_v}")
    else:
        print("   [global] confidence quantiles: v25 = nan | v75 = nan | N = 0")


if __name__ == "__main__":
    main()
