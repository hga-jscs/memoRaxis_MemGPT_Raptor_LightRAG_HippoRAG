import json
import glob
import numpy as np
import re
from pathlib import Path

def load_id_map():
    with open("MemoryAgentBench/entity2id.json", 'r') as f:
        data = json.load(f)
    id_to_title = {}
    for uri, idx in data.items():
        raw = uri.replace("<http://dbpedia.org/resource/", "").replace(">", "")
        title = raw.replace("_", " ")
        id_to_title[str(idx)] = title
    return id_to_title

def normalize(text):
    return re.sub(r'[^\w\s]', '', text).lower().strip()

def analyze_ttl():
    eval_files = glob.glob("out/ttl_results_*.json")
    eval_files.sort(key=lambda x: int(re.search(r'ttl_results_(\d+)(?:_[^./\\]+)*\.json', x).group(1)))
    
    if not eval_files:
        print("No evaluation files found.")
        return
    
    id_map = load_id_map()
    print(f"Loaded {len(id_map)} entity mappings.")

    print(f"\n{'Inst':<5} | {'Adaptor':<8} | {'Accuracy':<10}")
    print("-" * 30)

    all_scores = {"R1": [], "R2": [], "R3": []}

    for fpath in eval_files:
        match = re.search(r'ttl_results_(\d+)(?:_[^./\\]+)*\.json', fpath)
        idx = int(match.group(1))
        
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        results = data["results"]
        
        for adaptor, items in results.items():
            correct = 0
            total = 0
            
            for item in items:
                pred = item.get("answer", "").strip()
                gt_ids = item.get("ground_truth", [])
                
                # --- 评测逻辑分流 ---
                if idx == 0:
                    # 电影推荐：需映射 ID -> Title
                    is_hit = False
                    for gid in gt_ids:
                        title = id_map.get(str(gid))
                        if title:
                            if title.lower() in pred.lower():
                                is_hit = True
                                break
                    if is_hit:
                        correct += 1
                else:
                    # 普通 TTL：ID 出现在输出里就算对
                    for gid in gt_ids:
                        if str(gid) in pred:
                            correct += 1
                            break
                
                total += 1
            
            acc = correct / total if total else 0
            all_scores[adaptor].append(acc)
            print(f"{idx:<5} | {adaptor:<8} | {acc:<10.2%}")

    print("\n=== Global Avg ===")
    for adaptor, scores in all_scores.items():
        if scores:
            print(f"{adaptor}: {np.mean(scores):.2%}")
        else:
            print(f"{adaptor}: N/A")

if __name__ == "__main__":
    analyze_ttl()