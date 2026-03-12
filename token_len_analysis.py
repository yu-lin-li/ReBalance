import argparse
import json
from typing import Any, List

from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )


def extract_all_texts(gens: Any) -> List[str]:
    out: List[str] = []
    if isinstance(gens, list):
        for g in gens:
            if isinstance(g, str):
                out.append(g)
            elif isinstance(g, dict):
                for key in (
                    "text",
                    "content",
                    "generated_response",
                    "generated_text",
                    "output",
                    "message",
                    "response",
                ):
                    v = g.get(key)
                    if isinstance(v, str):
                        out.append(v)
                        break
            elif isinstance(g, list):
                # Fall back to the first string-like item
                for item in g:
                    if isinstance(item, str):
                        out.append(item)
                        break
                    if isinstance(item, dict):
                        for key in (
                            "text",
                            "content",
                            "generated_response",
                            "generated_text",
                            "output",
                            "message",
                            "response",
                        ):
                            v = item.get(key)
                            if isinstance(v, str):
                                out.append(v)
                                break
                        if out:
                            break
    return out


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name_or_path)

    total_tokens = 0
    total_responses = 0
    with open(args.input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            gens = obj.get("generated_responses", [])
            texts = extract_all_texts(gens)
            idx = obj.get("idx", i)
            print(f"sample {i} (idx={idx}) responses={len(texts)}")
            sample_tokens = 0
            for j, text in enumerate(texts):
                tok_len = count_tokens(tokenizer, text)
                sample_tokens += tok_len
                total_tokens += tok_len
                total_responses += 1
                print(f"  response {j}: tokens={tok_len}")
            if texts:
                avg_len = sample_tokens / len(texts)
                print(f"  avg_tokens={avg_len:.2f}")
            else:
                print("  avg_tokens=0.00")

    if total_responses:
        overall_avg = total_tokens / total_responses
        print(f"overall_avg_tokens={overall_avg:.2f} (responses={total_responses})")
    else:
        print("overall_avg_tokens=0.00 (responses=0)")


if __name__ == "__main__":
    main()
