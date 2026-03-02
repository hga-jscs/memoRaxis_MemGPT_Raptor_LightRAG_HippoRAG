# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path

# 这个脚本用于随机抽样查看 R3 的一些 case（方便人工 spot check）
# 支持你传入任何结果 json：acc_ret_results_*.json / acc_ret_results_*_hipporag.json 等


def sample_r3_cases(results_path: str, k: int = 5):
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    r3_results = data.get("results", {}).get("R3", [])
    if not r3_results:
        print("No R3 results found.")
        return

    k = min(k, len(r3_results))
    indices = random.sample(range(len(r3_results)), k)

    for i, idx in enumerate(indices):
        item = r3_results[idx]
        print(f"=== Case {i+1} (Index {idx}) ===")
        print(f"Question: {item.get('question')}")
        print(f"Answer: {item.get('answer')}")
        print(f"Steps: {item.get('steps')}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to acc_ret_results_*.json")
    parser.add_argument("--k", type=int, default=5, help="How many random cases")
    args = parser.parse_args()

    sample_r3_cases(args.results, args.k)


if __name__ == "__main__":
    main()