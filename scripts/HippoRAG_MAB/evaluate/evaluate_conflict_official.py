import json
import re
import string
import argparse
import glob
import numpy as np
from pathlib import Path

# --- 官方风格的文本处理函数 ---

def normalize_answer(s):
    """ 去小写、去标点、去冠词 """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for idx, ch in enumerate(text) if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

from collections import Counter

# --- 评测主逻辑 ---

def evaluate_conflict_results():
    results_files = glob.glob("out/conflict_res_results_*.json")
    results_files.sort(key=lambda x: int(re.search(r'conflict_res_results_(\d+)(?:_[^./\\]+)*\.json', x).group(1)))
    
    if not results_files:
        print("No conflict result files found.")
        return

    print(f"=== Conflict Resolution Official Evaluation Report (N={len(results_files)}) ===\n")
    print(f"{ 'Inst':<5} | { 'Adaptor':<8} | { 'ExactMatch':<10} | { 'SubMatch':<10} | { 'F1 Score':<10}")
    print("-" * 55)

    global_stats = {}

    for fpath in results_files:
        match = re.search(r'conflict_res_results_(\d+)(?:_[^./\\]+)*\.json', fpath)
        idx = int(match.group(1))
        
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        instance_gt_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{idx}.json"
        with open(instance_gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        gt_answers = gt_data["answers"]

        for adaptor, items in data["results"].items():
            exact_match = 0
            sub_match = 0
            f1_scores = []
            
            for item, gt in zip(items, gt_answers):
                pred = item.get("answer", "")
                gt = str(gt)
                
                # Exact match
                if normalize_answer(pred) == normalize_answer(gt):
                    exact_match += 1
                
                # Substring match
                if normalize_answer(gt) in normalize_answer(pred) or normalize_answer(pred) in normalize_answer(gt):
                    sub_match += 1
                
                # F1 score
                f1_scores.append(f1_score(pred, gt))
            
            exact_match_rate = exact_match / len(gt_answers)
            sub_match_rate = sub_match / len(gt_answers)
            avg_f1 = np.mean(f1_scores)

            # Print per instance stats
            print(f"{idx:<5} | {adaptor:<8} | {exact_match_rate:<10.2%} | {sub_match_rate:<10.2%} | {avg_f1:<10.2%}")

            # Collect global stats
            if adaptor not in global_stats:
                global_stats[adaptor] = {"exact": [], "sub": [], "f1": []}
            
            global_stats[adaptor]["exact"].append(exact_match_rate)
            global_stats[adaptor]["sub"].append(sub_match_rate)
            global_stats[adaptor]["f1"].append(avg_f1)

    print("\n=== Global Average ===")
    for adaptor, stats in global_stats.items():
        print(f"{adaptor:<8} | ExactMatch: {np.mean(stats['exact']):.2%} | SubMatch: {np.mean(stats['sub']):.2%} | F1: {np.mean(stats['f1']):.2%}")

if __name__ == "__main__":
    evaluate_conflict_results()