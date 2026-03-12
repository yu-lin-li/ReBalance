
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge DP inference shard files into a single JSONL.
- Deduplicate by idx (preferred) or question (fallback)
- Keep ascending order by idx when available; else by original file order
- Prints a summary report
Usage:
  python merge_shards.py \
    --dir ./outputs_steer/DeepSeek-R1-Distill-Qwen-7B/Math_Math500 \
    --base 'steer_temp0.7_maxlen10000' \
    --out merged.jsonl
Optionally include run_id in base, e.g., 'expA_steer_temp0.7_maxlen10000'
"""
import os
import json
import argparse
from glob import glob

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Directory containing shard files')
    ap.add_argument('--base', required=True, help='Base filename prefix (without .shard*.jsonl)')
    ap.add_argument('--out', default='', help='Output filename (default: {base}.merged.jsonl in --dir)')
    args = ap.parse_args()

    shard_glob = os.path.join(args.dir, f"{args.base}.shard*.jsonl")
    paths = sorted(glob(shard_glob))
    if not paths:
        raise SystemExit(f"No shard files matched: {shard_glob}")

    print(f"Found {len(paths)} shard(s):")
    for p in paths:
        print("  -", os.path.basename(p))

    seen_idx = set()
    seen_q = set()
    items_with_idx = []
    items_wo_idx = []

    total_in = 0
    for p in paths:
        for obj in load_lines(p):
            total_in += 1
            if 'idx' in obj:
                idx = int(obj['idx'])
                if idx in seen_idx:
                    continue
                seen_idx.add(idx)
                items_with_idx.append((idx, obj))
            else:
                q = obj.get('question', '')
                if q in seen_q:
                    continue
                seen_q.add(q)
                items_wo_idx.append(obj)

    # sort by idx; keep wo_idx in append order
    items_with_idx.sort(key=lambda x: x[0])
    merged_objs = [o for _, o in items_with_idx] + items_wo_idx

    out_path = args.out or os.path.join(args.dir, f"{args.base}.merged.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for obj in merged_objs:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print("=== Merge Summary ===")
    print(f"Input lines (all shards): {total_in}")
    print(f"Unique with idx: {len(items_with_idx)}")
    print(f"Unique without idx: {len(items_wo_idx)}")
    print(f"Total written: {len(merged_objs)}")
    print(f"Output: {out_path}")

if __name__ == '__main__':
    main()
