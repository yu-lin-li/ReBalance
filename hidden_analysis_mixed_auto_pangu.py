# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import Tuple

import torch
import torch_npu
from torch.utils.data import ConcatDataset

try:
    from hidden_analysis_mixed_pangu import batch_build_all_mixed
except Exception as e:
    raise ImportError(
        "Failed to import batch_build_all_mixed from hidden_analysis_mixed. "
        f"Make sure the file is in the same directory or on PYTHONPATH. Original error: {repr(e)}"
    )


def compute_confidence_quartiles(jsonl_path: str):
    # Extract confidence-like values from JSONL as robustly as possible.
    # Supported keys:
    #   - direct keys: "confidence", "conf", "score"
    #   - list keys: "sentence_confidences", "confidences", "scores", "sentence_scores", "step_confidences"
    #   - nested dict/list values are traversed recursively.
    import json, math
    import torch

    def _collect_from_obj(o, sink):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = k.lower()
                # direct scalar
                if lk in ("confidence", "conf", "score") and isinstance(v, (int, float)) and math.isfinite(v):
                    sink.append(float(v))
                # list field
                elif lk in ("sentence_confidences", "confidences", "scores", "sentence_scores", "step_confidences"):
                    if isinstance(v, (list, tuple)):
                        for x in v:
                            if isinstance(x, (int, float)) and math.isfinite(x):
                                sink.append(float(x))
                            elif isinstance(x, dict):
                                for kk in ("confidence", "conf", "score"):
                                    if kk in x and isinstance(x[kk], (int, float)) and math.isfinite(x[kk]):
                                        sink.append(float(x[kk]))
                # recursively traverse nested structures
                if isinstance(v, (dict, list, tuple)):
                    _collect_from_obj(v, sink)
        elif isinstance(o, (list, tuple)):
            for x in o:
                _collect_from_obj(x, sink)

    confidences = []
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


# ====== Utilities: collect two vector classes ======
def _collect_vectors_by_class(merged_dataset: ConcatDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect two classes from the merged dataset and return H_hit, H_non as float32 [N, D] tensors."""
    hits, nonhits = [], []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]
        feat = feat.view(-1).to(torch.float32)  # ensure 1D
        label = int(y.item()) if hasattr(y, "item") else int(y)
        if label == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)
    if len(hits) == 0 or len(nonhits) == 0:
        raise ValueError(f"Imbalanced classes: hit={len(hits)}, non={len(nonhits)}")
    H_hit = torch.stack(hits, dim=0)   # [N1, D]
    H_non = torch.stack(nonhits, dim=0)  # [N0, D]
    return H_hit, H_non


# ====== Utilities: center-to-set min/max distance (chunked) ======
@torch.no_grad()
def center_to_set_minmax(center: torch.Tensor,
                         H: torch.Tensor,
                         block_size: int = 65536,
                         device: str = "cpu") -> Tuple[float, float]:
    """
    Compute the minimum/maximum Euclidean distance from center vector to set H.
    Process in blocks to reduce memory usage.
    center: [D], H: [N, D]
    return (min_dist, max_dist)
    """
    assert center.dim() == 1 and H.dim() == 2 and center.size(0) == H.size(1), \
        f"Dimension mismatch: center {center.shape}, H {H.shape}"
    center = center.to(torch.float32).to(device)
    H = H.to(torch.float32).to(device)

    n = H.size(0)
    min_d = float("inf")
    max_d = 0.0
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        block = H[s:e]  # [B, D]
        diff = block - center  # [B, D]
        d = torch.linalg.norm(diff, dim=1)  # [B]
        min_d = min(min_d, float(d.min().item()))
        max_d = max(max_d, float(d.max().item()))
    return min_d, max_d


# ====== Core: build centers and ranges ======
@torch.no_grad()
def build_centers_and_ranges(merged_dataset: ConcatDataset,
                             center_block_size: int = 65536,
                             device: str = "cpu"):
    """
    Build from merged dataset:
      - S = mean(H_hit) - mean(H_non)
      - class means μ_hit, μ_non for H_hit and H_non
      - min/max distance from μ_hit to H_non and μ_non to H_hit
    """
    H_hit, H_non = _collect_vectors_by_class(merged_dataset)  # [N1, D], [N0, D]

    # compute class means
    mu_hit = H_hit.mean(dim=0)  # [D]
    mu_non = H_non.mean(dim=0)  # [D]
    S = (mu_hit - mu_non).to(torch.float32)  # [D]

    # distance ranges (block-wise)
    hit_to_non_min, hit_to_non_max = center_to_set_minmax(mu_hit, H_non, center_block_size, device)
    non_to_hit_min, non_to_hit_max = center_to_set_minmax(mu_non, H_hit, center_block_size, device)

    return S, mu_hit, mu_non, H_hit, H_non, hit_to_non_min, hit_to_non_max, non_to_hit_min, non_to_hit_max


# ====== LDA (diagonal approximation) and steer ======
@torch.no_grad()
def compute_lda_separator_and_steer(merged_dataset: ConcatDataset, ridge: float = 0.0):
    """
    Compute a diagonal-covariance LDA separator, returning direction w and threshold b
    such that sign(w·x + b) can be used as a linear decision boundary.
    This is for analysis/steering direction only, not for training a classifier.
    """
    H_hit, H_non = _collect_vectors_by_class(merged_dataset)
    X1 = H_hit.to(torch.float64)
    X0 = H_non.to(torch.float64)

    mu1 = X1.mean(dim=0)
    mu0 = X0.mean(dim=0)
    S1 = X1.var(dim=0, unbiased=False)  # diagonal approximation
    S0 = X0.var(dim=0, unbiased=False)

    # add tiny ridge to avoid numerical instability from very small variance
    if ridge > 0:
        S1 = S1 + ridge
        S0 = S0 + ridge

    # diagonal LDA: w = (mu1 - mu0) / (S1 + S0)
    denom = S1 + S0 + 1e-12
    w = (mu1 - mu0) / denom
    w = w.to(torch.float64)

    # threshold from simplified Fisher midpoint
    # classify as class 1 when w·x + b > 0
    m1 = (w * mu1).sum().item()
    m0 = (w * mu0).sum().item()
    t = -0.5 * (m1 + m0)  # threshold t = -b
    b = -t

    # directional norms
    S_vec = (mu1 - mu0)
    w_norm = float(torch.linalg.norm(w).item())
    S_norm = float(torch.linalg.norm(S_vec).item())
    if w_norm < 1e-12 or S_norm < 1e-12:
        return {"ok": False, "reason": "Direction norm too small"}

    # shortest distance to move all hit samples across the boundary along -u_w
    # for each hit sample h, need d_i with t - w·h + d_i * ||w|| >= 0
    # equivalent to d_i >= (w·h - t) / ||w||
    proj_hit = (X1 @ w).double()  # [N1]
    req = (proj_hit - t) / (w_norm + 1e-12)  # [N1]
    d_all_w = float(req.max().item())

    # fixed-direction variant along -S with unit vector u_S
    uS = S_vec / (S_norm + 1e-12)
    w_dot_uS = float((w * uS).sum().item())  # w·uS
    if w_dot_uS <= 0:
        # if w and -S are too misaligned, movement along -S cannot reduce decision value
        d_all_S = float("inf")
        alpha_all_S = float("inf")
        d_mean_S = float("inf")
        alpha_mean_S = float("inf")
        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)
    else:
        # projection distance to push all hits across along -uS: max_i (w·h_i - t) / (w·uS)
        alpha_u_all = float(((proj_hit - t) / (w_dot_uS + 1e-12)).max().item())
        # convert distance in -uS direction to coefficient in -S direction: alpha_S = alpha_u / ||S||
        alpha_all_S = alpha_u_all / (S_norm + 1e-12)
        # minimal Euclidean displacement in -uS direction
        d_all_S = alpha_u_all

        # mean-only breach for mu1
        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)
        d_mean_S_u = max((m1 - t) / (w_dot_uS + 1e-12), 0.0)   # Euclidean distance along -uS
        alpha_mean_S = d_mean_S_u / (S_norm + 1e-12)            # convert to coefficient along -S
        d_mean_S = d_mean_S_u

    return {
        "ok": True,
        "w_norm": w_norm,
        "S_norm": S_norm,
        "w": w,
        "S": S_vec.to(torch.float64),
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute S, center-to-other distance ranges, and LDA-based steer thresholds (command unchanged)"
    )
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save mean-difference vector S (.pt)")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    # keep legacy args for compatibility (ignored)
    parser.add_argument("--pairwise_block_rows", type=int, default=2048, help="[compatibility] ignored")
    parser.add_argument("--pairwise_block_cols", type=int, default=4096, help="[compatibility] ignored")

    # active args
    parser.add_argument("--center_block_size", type=int, default=65536,
                        help="Block size for center-to-set distance computation")
    parser.add_argument("--device", type=str, choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--report_path", type=str, default="",
                        help="Optional JSON report output path")

    # keep these CLI args for compatibility; no longer used in margin calculations
    parser.add_argument("--gamma", type=float, default=1.5, help="[compatibility]")
    parser.add_argument("--epsilon", type=float, default=150, help="[compatibility]")
    parser.add_argument("--delta", type=float, default=35, help="[compatibility]")

    args = parser.parse_args()

    # build merged (feat, label) dataset
    merged = batch_build_all_mixed(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        threshold=args.threshold,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=args.verbose
    )

    (S, mu_hit, mu_non,
     H_hit, H_non,
     hit_to_non_min, hit_to_non_max,
     non_to_hit_min, non_to_hit_max) = build_centers_and_ranges(
        merged,
        center_block_size=args.center_block_size,
        device=args.device
    )

    # save S
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    # print center-to-set distances and S summary
    dim = S.numel()
    l2 = float(torch.linalg.norm(S).item())
    print(f"✅ Saved steer vector to {args.save_path}")
    print(f"   dim={dim} | ||S||_2 = {l2:.6f} | hit={H_hit.size(0)} | nonhit={H_non.size(0)}")
    print(f"   μ_hit → H_non  min distance: {hit_to_non_min:.6f} | max distance: {hit_to_non_max:.6f}")
    print(f"   μ_non → H_hit  min distance: {non_to_hit_min:.6f} | max distance: {non_to_hit_max:.6f}")

    # ===== LDA separator and steer metrics (ridge=0, no prior) =====
    lda = compute_lda_separator_and_steer(merged)
    if lda.get("ok", False):
        print("\n—— Diagonal LDA separator + steer ——")
        print(f"  • ||w||_2 = {lda['w_norm']:.6f} | ||S||_2 = {lda['S_norm']:.6f}")
        print(f"  • Threshold t = -b = {lda['threshold_t']:.6f} | max_i(w·h_i) = {lda['max_w_dot_h']:.6f}")

        print("  • Best direction (along -u_w):")
        print(f"      d_all_w  = {lda['d_all_w']:.6f}   # shortest Euclidean move to push all hits to non")

        print("  • Fixed direction (along -S):")
        if lda['d_all_S'] == float('inf'):
            print("      Angle with w is too large; along -S cannot reduce score. Use -w or check S definition.")
        else:
            print(f"      d_all_S  = {lda['d_all_S']:.6f}   | alpha_all_S = {lda['alpha_all_S']:.6f}")

        print("  • Mean-only breach:")
        if lda['d_mean_S'] == float('inf'):
            print(f"      d_mean_w = {lda['d_mean_w']:.6f} | d_mean_S undefined (w·S ≤ 0)")
        else:
            print(f"      d_mean_w = {lda['d_mean_w']:.6f} | d_mean_S = {lda['d_mean_S']:.6f} | alpha_mean_S = {lda['alpha_mean_S']:.6f}")
    else:
        print(f"⚠️ LDA separator failed: {lda.get('reason','unknown')}")

    # ===== Compute and print confidence quantiles =====
    q25, q75, n_conf = compute_confidence_quartiles(args.jsonl_path)
    if n_conf > 0:
        print(f"   confidence quantiles: q25 = {q25:.6f} | q75 = {q75:.6f} | N = {n_conf}")
    else:
        print("   No confidence values found in JSONL; skip quantile stats.")

    # ===== Optional report output =====
    if args.report_path:
        report = {
            "layer_id": args.layer_id,
            "dim": dim,
            "n_hit": H_hit.size(0),
            "n_non": H_non.size(0),
            "S_l2": l2,
            "hit_to_non_min": hit_to_non_min,
            "hit_to_non_max": hit_to_non_max,
            "non_to_hit_min": non_to_hit_min,
            "non_to_hit_max": non_to_hit_max,
            "jsonl_path": args.jsonl_path,
            "hidden_dir": args.hidden_dir,
            "threshold": args.threshold,
            "center_block_size": args.center_block_size,
            "device": args.device,
            "confidence_q25": q25,
            "confidence_q75": q75,
            "confidence_N": n_conf,
        }
        if lda.get("ok", False):
            report.update({
                "w_norm": lda["w_norm"],
                "S_norm": lda["S_norm"],
                "b": lda["b"],
                "threshold_t": lda["threshold_t"],
                "max_w_dot_h": lda["max_w_dot_h"],
                "margin_needed": lda["d_all_w"],
                "d_all_w": lda["d_all_w"],
                "d_all_S": lda["d_all_S"],
                "alpha_all_S": lda["alpha_all_S"],
                "d_mean_w": lda["d_mean_w"],
                "d_mean_S": lda["d_mean_S"],
                "alpha_mean_S": lda["alpha_mean_S"],
                "w_dot_uS": lda["w_dot_uS"],
            })
        if os.path.dirname(args.report_path):
            os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Report saved: {args.report_path}")


if __name__ == "__main__":
    main()
