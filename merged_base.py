# -*- coding: utf-8 -*-

import glob

import os
import json
import argparse

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(tmp, file_path)

def merge_all_shards(output_dir, base_name, remove_shards=True):
    shard_files = sorted(glob.glob(os.path.join(output_dir, f'{base_name}.shard*.jsonl')))
    combined_file = os.path.join(output_dir, f'{base_name}.jsonl')

    # Also merge existing combined file to support resume and deduplicate.
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

    # Sort by idx before output.
    final = [merged[k] for k in sorted(merged.keys())]
    write_jsonl(final, combined_file)

    # remove shard files
    if remove_shards and shard_files:
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
                print(f" 🗑️  Removed shard: {shard_file}")
            except Exception as e:
                print(f"[WARN] Failed to remove {shard_file}: {e}")

    return combined_file, len(final)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    args = parser.parse_args()
    
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=True)
    print(f"✅ Merged {count} entries to: {combined_file}")



if __name__ == "__main__":
    main()
