# -*- coding: utf-8 -*-
"""
hidden_config_ridge.py


Core idea
- For each layer L:
  1) Load per-example hidden-state "step" vectors saved as `hidden_{idx}.pt`.
  2) Align each step vector with a segment-level confidence proxy from the corresponding JSONL record.
  3) Build a dataset X (hidden features) and y (confidence targets).
  4) Run PCA (optional dimensionality reduction) and train Ridge regression.
  5) Report test-set R^2 as the layer score.

- Finally, print the BEST LAYER (max R^2) and also print the best layer id alone
  (convenient for piping/capturing in shell scripts).

Data expectations
- JSONL file at `--jsonl_path`, each line is a dict including:
    - "sentence_confidences": list[float]
  (The confidence values are assumed to correspond to "think segments" or steps.)

- Hidden files under `--hidden_dir` in pattern `hidden_{idx}.pt`:
  Each file is expected to be a nested dict like:
      hidden_dict[layer_id][sample_id]["step"] = Tensor[num_steps, hidden_dim]
  This matches common "dump hidden states" pipelines.

Alignment rule (important)
- We align:
    V = number of step vectors in hidden state
    C_raw = len(sentence_confidences)
  and require:
    V == C_raw - expected_offset
  Then we take:
    y = sentence_confidences[expected_offset : expected_offset + V]
  This offset exists because pipelines sometimes include a placeholder segment/confidence at the front.

Default layer scanning
- `--layers all` (default) automatically infers the last layer id by loading the first existing
  hidden file, then scans layers 0..max_layer inclusive.

Reproducibility
- `--random_state` controls train/test split and PCA randomness (where applicable).

Typical usage
python hidden_config_ridge.py \
  --jsonl_path ./outputs/.../origin_temp0.7_maxlen16000.shard0.jsonl \
  --hidden_dir ./outputs/.../ \
  --layers all \
  --max_files 150 \
  --expected_offset 1 \
  --alpha 1.0 \
  --pca_components 64 \
  --test_size 0.2 \
  --random_state 42

Notes
- This script is intentionally simple and transparent, suitable for open-source release.
- It does not modify your generation logic; it only evaluates already-dumped artifacts.
"""

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# =============================================================================
# IO helpers
# =============================================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of Python dicts.

    Parameters
    ----------
    path : str
        Path to a JSONL file, one JSON object per line.

    Returns
    -------
    list[dict]
        Parsed objects from the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


# =============================================================================
# Per-sample dataset constructor
# =============================================================================

def build_dataset_from_layer_conf(
    layer_id: int,
    json_item: Dict[str, Any],
    hidden_path: str,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[TensorDataset]:
    """
    Build a TensorDataset for a single sample (one json record + one hidden_{idx}.pt).

    What this function does
    1) Load confidence targets from json_item["sentence_confidences"].
    2) Load step-level hidden vectors from hidden_path for the specified layer.
    3) Check alignment:
         V == C_raw - expected_offset
       where:
         V     = number of step vectors in hidden states
         C_raw = length of the raw confidence list
    4) Create:
         X = Tensor[V, D] float32
         y = Tensor[V]    float32
       and return as a TensorDataset.

    Parameters
    ----------
    layer_id : int
        Which layer's hidden states to use.

    json_item : dict
        One line from the JSONL file. Must contain "sentence_confidences" list.

    hidden_path : str
        Path to hidden_{idx}.pt.

    expected_offset : int
        Alignment offset for confidence list.

    verbose : bool
        If True, prints skip reasons.

    Returns
    -------
    TensorDataset or None
        Returns None if missing/invalid data or misalignment.
    """
    confs_raw = json_item.get("sentence_confidences", [])
    if not isinstance(confs_raw, list) or len(confs_raw) <= expected_offset:
        if verbose:
            print("[skip] sentence_confidences missing or too short.")
        return None

    if not os.path.exists(hidden_path):
        if verbose:
            print(f"[skip] hidden file not found: {hidden_path}")
        return None

    try:
        # weights_only=False for general torch.save dicts (not strictly model weights).
        data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if verbose:
            print(f"[skip] failed to load hidden file: {hidden_path} ({e})")
        return None

    if layer_id not in data:
        if verbose:
            print(f"[skip] layer {layer_id} not found in {os.path.basename(hidden_path)}")
        return None

    layer_data = data[layer_id]

    # By convention in the paired dumping script, sample_id is 0 for each file.
    sample_id = 0
    if sample_id not in layer_data or "step" not in layer_data[sample_id]:
        if verbose:
            print(f"[skip] missing sample_id=0 or 'step' in {os.path.basename(hidden_path)}")
        return None

    step_tensor = layer_data[sample_id]["step"]  # expected shape: [V, D]
    if not torch.is_tensor(step_tensor) or step_tensor.ndim != 2:
        if verbose:
            print("[skip] 'step' is not a 2D tensor.")
        return None

    V = int(step_tensor.shape[0])
    C_raw = int(len(confs_raw))

    # Alignment rule: V must match confidence count after removing offset.
    if V != C_raw - expected_offset:
        if verbose:
            print(f"[skip] alignment mismatch: V={V} vs C_raw={C_raw} with offset={expected_offset}")
        return None

    confs = confs_raw[expected_offset: expected_offset + V]
    if len(confs) != V:
        if verbose:
            print("[skip] confidence slicing produced incorrect length.")
        return None

    X = step_tensor.to(dtype=torch.float32)               # [V, D]
    y = torch.tensor(confs, dtype=torch.float32)          # [V]
    return TensorDataset(X, y)


# =============================================================================
# Batch dataset builder (across many hidden_{idx}.pt files)
# =============================================================================

def batch_build_all(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    max_files: int = 150,
    expected_offset: int = 1,
    file_pattern: str = "hidden_{idx}.pt",
    verbose: bool = True
) -> ConcatDataset:
    """
    Build a merged dataset for one layer by iterating over many samples.

    Parameters
    ----------
    layer_id : int
        Layer id to evaluate.

    jsonl_path : str
        Path to the JSONL file containing sentence_confidences.

    hidden_dir : str
        Directory containing hidden_{idx}.pt files.

    max_files : int
        Maximum number of JSONL lines (and hidden files) to consider.

    expected_offset : int
        Alignment offset for confidence list.

    file_pattern : str
        Filename pattern, default "hidden_{idx}.pt".

    verbose : bool
        Print progress and skip information.

    Returns
    -------
    ConcatDataset
        Concatenation of valid per-sample TensorDatasets.

    Raises
    ------
    RuntimeError
        If no samples can be collected.
    """
    data_json = read_jsonl(jsonl_path)
    total = min(max_files, len(data_json))

    datasets: List[TensorDataset] = []
    kept_files = 0
    kept_samples = 0

    for i in range(total):
        hidden_path = os.path.join(hidden_dir, file_pattern.format(idx=i))

        ds = build_dataset_from_layer_conf(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            expected_offset=expected_offset,
            verbose=False
        )

        if ds is not None:
            datasets.append(ds)
            kept_files += 1
            kept_samples += len(ds)
        else:
            if verbose:
                print(f"[skip] index={i}")

        if (i + 1) % 50 == 0 and verbose:
            print(f"[progress] processed {i+1}/{total} | kept_files={kept_files} | kept_samples={kept_samples}")

    if not datasets:
        raise RuntimeError("No valid samples were collected. Please check paths/layer_id/expected_offset.")

    merged = ConcatDataset(datasets)
    if verbose:
        print(f"\n[done] merged dataset: num_samples={len(merged)} | kept_files={kept_files}/{total}")
    return merged


# =============================================================================
# Evaluation: PCA + Ridge regression
# =============================================================================

def evaluate_pca_ridge(
    merged_dataset: ConcatDataset,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    pca_components: int = 64
) -> float:
    """
    Evaluate a dataset with PCA + Ridge regression and return test-set R^2.

    Pipeline
    - Collect all (feature, label) pairs:
        X: [N, D], y: [N]
    - Train/test split
    - PCA on train, transform test
    - Ridge regression on PCA features
    - Compute R^2 on test split

    Parameters
    ----------
    merged_dataset : ConcatDataset
        Dataset of (feature_vector, confidence_scalar) pairs.

    alpha : float
        Ridge regularization strength.

    test_size : float
        Fraction of samples used for test split.

    random_state : int
        Random seed for splitting and PCA randomness.

    pca_components : int
        Requested PCA components; will be capped by min(N_train, D).

    Returns
    -------
    float
        Test-set R^2.
    """
    features: List[np.ndarray] = []
    labels: List[float] = []

    for i in range(len(merged_dataset)):
        feat, lab = merged_dataset[i]  # feat: Tensor[D], lab: Tensor[()]
        features.append(feat.to(torch.float32).numpy())
        labels.append(float(lab.item()))

    X = np.asarray(features, dtype=np.float32)  # [N, D]
    y = np.asarray(labels, dtype=np.float32)    # [N]

    # Train/test split (reproducible via random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Cap PCA components to feasible range
    max_comp = min(int(pca_components), int(X_train.shape[0]), int(X_train.shape[1]))
    if max_comp <= 0:
        raise ValueError("Invalid PCA component count. Reduce --pca_components or ensure enough samples/features.")

    pca = PCA(n_components=max_comp, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    reg = Ridge(alpha=alpha)
    reg.fit(X_train_pca, y_train)
    y_pred = reg.predict(X_test_pca)

    r2 = r2_score(y_test, y_pred)
    print(f"[eval] PCA+Ridge: R^2={r2:.4f} | components={max_comp} | alpha={alpha}")
    return float(r2)


# =============================================================================
# Layer parsing / inference
# =============================================================================

def parse_layers(layer_str: str) -> List[int]:
    """
    Parse the `--layers` argument.

    Supported formats
    - "all" (case-insensitive): special value meaning "infer and scan all layers"
      (returns an empty list; caller is expected to infer from disk)
    - "20"            -> [20]
    - "0-28"          -> [0, 1, ..., 28]
    - "1,5,7-10"      -> [1, 5, 7, 8, 9, 10]
    - whitespace is allowed as separator in addition to commas

    Returns
    -------
    list[int]
        Sorted unique layer ids, or [] if layer_str == "all".
    """
    layer_str = (layer_str or "").strip()
    if layer_str.lower() == "all":
        return []

    layers: List[int] = []
    parts = re.split(r"[,\s]+", layer_str)
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                layers.extend(range(a, b + 1))
            else:
                layers.extend(range(a, b - 1, -1))
        else:
            layers.append(int(p))

    return sorted(set(layers))


def infer_all_layers_from_hidden(
    hidden_dir: str,
    file_pattern: str = "hidden_{idx}.pt",
    max_probe: int = 4096
) -> List[int]:
    """
    Infer the available layer range by loading the first existing hidden file.

    Strategy
    - Probe for the first existing hidden file among indices [0, max_probe).
    - Load it and read integer keys as layer ids.
    - Return a dense scan list: [0, 1, ..., max_layer].

    Rationale
    - Many pipelines dump all layers per file, so one file is sufficient to infer max_layer.

    Raises
    ------
    FileNotFoundError
        If no hidden files can be found within the probe range.

    RuntimeError
        If the found hidden file does not contain integer layer keys.
    """
    first_path = None
    for i in range(max_probe):
        p = os.path.join(hidden_dir, file_pattern.format(idx=i))
        if os.path.exists(p):
            first_path = p
            break

    if first_path is None:
        raise FileNotFoundError(
            f"Cannot infer layers: no hidden files found under '{hidden_dir}' "
            f"with pattern '{file_pattern}' within first {max_probe} indices."
        )

    data = torch.load(first_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"Hidden file is not a non-empty dict: {first_path}")

    layer_keys: List[int] = []
    for k in data.keys():
        try:
            layer_keys.append(int(k))
        except Exception:
            continue

    if not layer_keys:
        raise RuntimeError(f"No integer layer keys found in hidden file: {first_path}")

    max_layer = max(layer_keys)
    return list(range(0, max_layer + 1))


# =============================================================================
# Main entry
# =============================================================================

def main():
    """
    Command-line entry point.

    Outputs
    - Prints per-layer R^2 scores.
    - Prints the BEST LAYER (max R^2) with its score.
    - Prints the best layer id alone on the last line (shell-friendly).
    """
    ap = argparse.ArgumentParser(
        description="Layer selection via PCA+Ridge regression (test-set R^2) using hidden-state features."
    )
    ap.add_argument(
        "--jsonl_path", type=str,
        default="./outputs/Deepseek_7B_HiddenState/origin_temp0.7_maxlen16000.jsonl",
        help="Path to JSONL containing sentence_confidences per example."
    )
    ap.add_argument(
        "--hidden_dir", type=str, default="./outputs/Deepseek_7B_HiddenState/",
        help="Directory containing hidden_{idx}.pt files."
    )
    ap.add_argument(
        "--layers", type=str, default="all",
        help="Layer ids to scan. Use 'all' to scan 0..last inferred layer; "
             "or specify e.g. '0-28' or '0,3,7-10'."
    )
    ap.add_argument("--max_files", type=int, default=150, help="Maximum number of examples to load.")
    ap.add_argument("--expected_offset", type=int, default=1, help="Offset for aligning confidences to step vectors.")
    ap.add_argument("--file_pattern", type=str, default="hidden_{idx}.pt", help="Hidden file name pattern.")

    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    ap.add_argument("--pca_components", type=int, default=64, help="Requested PCA components (capped automatically).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")

    args = ap.parse_args()

    # Determine which layers to scan.
    layer_ids = parse_layers(args.layers)
    if args.layers.strip().lower() == "all":
        layer_ids = infer_all_layers_from_hidden(args.hidden_dir, args.file_pattern)
        print(f"[layers] Auto-scan inferred range: 0-{layer_ids[-1]} (total={len(layer_ids)})")

    # Track scores as (layer, r2 or None).
    r2_scores: List[Tuple[int, Optional[float]]] = []

    for layer in layer_ids:
        print(f"\n[layer] Evaluating layer {layer} ...")
        try:
            merged_dataset = batch_build_all(
                layer_id=layer,
                jsonl_path=args.jsonl_path,
                hidden_dir=args.hidden_dir,
                max_files=args.max_files,
                expected_offset=args.expected_offset,
                file_pattern=args.file_pattern,
                verbose=True
            )
            r2 = evaluate_pca_ridge(
                merged_dataset,
                alpha=args.alpha,
                pca_components=args.pca_components,
                test_size=args.test_size,
                random_state=args.random_state
            )
            r2_scores.append((layer, r2))
            print(f"[layer] Layer {layer} | test R^2 = {r2:.4f}")
        except Exception as e:
            print(f"[error] Layer {layer} failed: {e}")
            r2_scores.append((layer, None))

    # Summary table
    print("\n[summary] Per-layer PCA+Ridge test R^2:")
    for layer, r2 in r2_scores:
        if r2 is not None:
            print(f"  Layer {layer:2d}: R^2 = {r2:.4f}")
        else:
            print(f"  Layer {layer:2d}: (no result)")

    # Select and print best layer.
    valid = [(layer, r2) for layer, r2 in r2_scores if r2 is not None]
    if not valid:
        print("\n[best] No valid R^2 results; cannot select best layer.")
    else:
        best_layer, best_r2 = max(valid, key=lambda x: x[1])
        print(f"\n[BEST LAYER] Layer {best_layer} (max test R^2 = {best_r2:.4f})")
        print(best_layer)  # plain layer id (shell-friendly)


if __name__ == "__main__":
    main()
