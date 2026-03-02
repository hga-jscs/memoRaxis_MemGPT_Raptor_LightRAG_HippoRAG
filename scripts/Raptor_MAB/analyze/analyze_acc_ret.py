import re
import numpy as np
#这段脚本的作用可以一句话概括：
#从一个“纯文本汇总报告”里把每个测试实例（instance）的 R1/R2/R3 准确率解析出来，
#然后做整体均值统计、按难度分区统计（用 R1 当基线）
#最后打印每个实例的详细对比表
def parse_raw_report(filename):# 打开并读取原始文本报告文件（整个文件一次性读入为字符串）
    with open(filename, 'r') as f:
        content = f.read()
    # 按照每个实例块的分隔符切分文本
    # 原始报告里每个实例通常以类似：
    # --- [Mechanical Evaluation Result: ... 的形式开始
    # Split by instance blocks
    blocks = content.split('--- [Mechanical Evaluation Result:')
    # 用来存放最终解析结果：
    # data[instance_idx] = {"R1":xx, "R2":xx, "R3":xx}
    data = {}
    # 遍历每个切分出来的 block（每个 block 理论上对应一个实例的评测结果）
    for block in blocks:
        if not block.strip():#如果 block 只是空白（比如 split 后的第一个块可能为空），跳过
            continue
            
        # Extract filename (instance id)
        # 从 block 里提取实例编号（instance id）
        # 这里假设 block 内含类似：acc_ret_results_123.json
        title_match = re.search(r'acc_ret_results_(\d+)\.json', block)
        if not title_match:
            continue
        # 如果找不到这个文件名模式，就认为不是有效实例块，跳过
        instance_idx = int(title_match.group(1))
        # 把捕获到的数字转换成整数作为 instance_idx
        # Extract scores
        # Pattern: Adaptor R1 : Accuracy = 95.00% (95/100)
        # 存放该实例下不同 adaptor（R1/R2/R3）的准确率
        # 目标模式类似：
        # Adaptor R1 : Accuracy = 95.00% (95/100)

        # 依次解析 R1 / R2 / R3 的 Accuracy 百分比
        scores = {}
        for adaptor in ['R1', 'R2', 'R3']:
            # 组装正则模式：
            # - "Adaptor R1 : Accuracy = 95.00%"
            # - \s+ 匹配若干空白（空格/换行/制表）
            # - ([\d\.]+) 捕获数字和小数点（例如 95 或 95.00）
            pattern = f"Adaptor {adaptor}\s+:\s+Accuracy\s+=\s+([\d\.]+)\%"
            # 在当前 block 中搜索该模式
            match = re.search(pattern, block)
            if match:
                scores[adaptor] = float(match.group(1))
            else:# 如果该 adaptor 的分数缺失，则默认记为 0.0
                scores[adaptor] = 0.0 # Default if missing
        # 把该实例的 scores 写入总字典
        data[instance_idx] = scores
        # 返回：所有实例的解析结果
    return data

def analyze(data):
    # 这里把“难度分区”定义为：用 R1（基线）来判定该实例有多难
    # - High: R1 >= 80
    # - Mid : 60 <= R1 < 80
    # - Low : R1 < 60
    # Calculate average scores per instance (to define zones based on average performance or baseline R1)
    # Strategy: Use R1's score to define difficulty zones.
    # High: R1 > 80, Mid: 60-80, Low: < 60
    
    zones = {# zones 用来按难度区间收集实例（注意这里收集的是 scores 字典）
        "High": [],
        "Mid": [],
        "Low": []
    }
    
    all_scores = {"R1": [], "R2": [], "R3": []}# all_scores 用于做全局平均：分别收集所有实例的 R1/R2/R3 分数
    # 遍历每个实例
    for idx, scores in data.items():
        # Collect for global
        # 把该实例的每个 adaptor 分数加入全局列表，供后面 np.mean 计算
        for k in scores:
            all_scores[k].append(scores[k])
            
        # Determine zone by R1 performance (Baseline)
        base_score = scores['R1']
        if base_score >= 80:
            zones["High"].append(scores)
        elif base_score >= 60:
            zones["Mid"].append(scores)
        else:
            zones["Low"].append(scores)
    # 打印报告标题：N=实例数量
    print(f"=== Accurate Retrieval Evaluation Report (N={len(data)}) ===\n")
    
    # 1. Global
    print("## 1. Global Performance (Avg Accuracy)")
    print(f"- R1 (Single-turn): {np.mean(all_scores['R1']):.2f}%")
    print(f"- R2 (Iterative)  : {np.mean(all_scores['R2']):.2f}%")
    print(f"- R3 (Plan-Act)   : {np.mean(all_scores['R3']):.2f}%")
    print("")

    # 2. Zones Analysis
    print("## 2. Zone Analysis (Based on R1 Baseline Difficulty)")
    
    for zone_name, items in [("High (>80%)", zones["High"]), ("Mid (60-80%)", zones["Mid"]), ("Low (<60%)", zones["Low"])]:
        if not items:
            continue
            
        n = len(items)
        r1_avg = np.mean([x['R1'] for x in items])
        r2_avg = np.mean([x['R2'] for x in items])
        r3_avg = np.mean([x['R3'] for x in items])
        
        print(f"### {zone_name} Zone (N={n})")
        print(f"- R1: {r1_avg:.2f}%")
        print(f"- R2: {r2_avg:.2f}% ({r2_avg - r1_avg:+.2f}% vs R1)")
        print(f"- R3: {r3_avg:.2f}% ({r3_avg - r1_avg:+.2f}% vs R1)")
        
        best_adaptor = "R1"
        if r2_avg > max(r1_avg, r3_avg): best_adaptor = "R2"
        if r3_avg > max(r1_avg, r2_avg): best_adaptor = "R3"
        
        print(f"  -> Winner: {best_adaptor}")
        print("")

    # 3. Detailed Table
    print("## 3. Detailed Instance Breakdown")
    print(f"{'Inst':<5} | {'R1':<6} | {'R2':<6} | {'R3':<6} | {'Winner':<6}")
    print("-" * 35)
    
    sorted_indices = sorted(data.keys())
    for idx in sorted_indices:
        s = data[idx]
        winner = max(s, key=s.get)
        print(f"{idx:<5} | {s['R1']:<6} | {s['R2']:<6} | {s['R3']:<6} | {winner:<6}")

if __name__ == "__main__":
    data = parse_raw_report("acc_ret_summary_raw.txt")
    analyze(data)
# 你可以把它理解成一个报告解析器+小型分析器：
# 前半段负责从原始 txt 里提取每个实例的准确率数字
# 后半段负责把这些数字按规则做均值、分组、排名
# 然后打印成一份更清晰的对比报告