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
    # data[instance_idx] = { 'R1': acc, 'R2': acc, 'R3': acc }
    data = {}

    for block in blocks:
        # Each block starts with " Instance X..."
        # We need to find instance index
        # Extract instance id from the title line
        # Extract filename (instance id)
        # 从 block 里提取实例编号（instance id）
        # 这里假设 block 内含类似：acc_ret_results_123.json
        title_match = re.search(r'acc_ret_results_(\d+)(?:_[^./\\]+)*\.json', block)
        if not title_match:
            continue
        # 如果找不到这个文件名模式，就认为不是有效实例块，跳过
        instance_idx = int(title_match.group(1))

        # Extract accuracy for each adaptor
        # Find lines like "R1 Accuracy: 0.8000"
        adaptor_acc = {}

        for adaptor in ['R1', 'R2', 'R3']:
            # Use regex to find "R1 Accuracy: X"
            acc_match = re.search(rf'{adaptor} Accuracy:\s*([0-9.]+)', block)
            if acc_match:
                adaptor_acc[adaptor] = float(acc_match.group(1))
            else:
                adaptor_acc[adaptor] = np.nan # If not found, set NaN

        data[instance_idx] = adaptor_acc

    return data


def analyze_accuracy(data):
    # Overall mean
    adaptors = ['R1', 'R2', 'R3']
    overall_means = {}
    for adaptor in adaptors:
        vals = [data[idx][adaptor] for idx in data if not np.isnan(data[idx][adaptor])]
        overall_means[adaptor] = np.mean(vals) if len(vals) > 0 else np.nan

    # Difficulty partitions based on R1 accuracy
    easy_idxs = [idx for idx in data if data[idx]['R1'] > 0.5]
    hard_idxs = [idx for idx in data if data[idx]['R1'] <= 0.5]

    def mean_for_partition(idxs):
        part_means = {}
        for adaptor in adaptors:
            vals = [data[idx][adaptor] for idx in idxs if not np.isnan(data[idx][adaptor])]
            part_means[adaptor] = np.mean(vals) if len(vals) > 0 else np.nan
        return part_means

    easy_means = mean_for_partition(easy_idxs)
    hard_means = mean_for_partition(hard_idxs)

    return overall_means, easy_means, hard_means, easy_idxs, hard_idxs


def print_report(data, overall_means, easy_means, hard_means, easy_idxs, hard_idxs):
    adaptors = ['R1', 'R2', 'R3']

    print("=== Overall Mean Accuracy ===")
    for adaptor in adaptors:
        print(f"{adaptor}: {overall_means[adaptor]:.4f}")

    print("\n=== Easy Instances (R1 > 0.5) ===")
    print(f"Count: {len(easy_idxs)}")
    for adaptor in adaptors:
        print(f"{adaptor}: {easy_means[adaptor]:.4f}")

    print("\n=== Hard Instances (R1 <= 0.5) ===")
    print(f"Count: {len(hard_idxs)}")
    for adaptor in adaptors:
        print(f"{adaptor}: {hard_means[adaptor]:.4f}")

    print("\n=== Detailed Instance Results ===")
    header = f"{'Inst':<6} | " + " | ".join([f"{ad:<6}" for ad in adaptors])
    print(header)
    print("-" * len(header))
    for idx in sorted(data.keys()):
        row = f"{idx:<6} | " + " | ".join([f"{data[idx][ad]:.4f}" if not np.isnan(data[idx][ad]) else "  NaN " for ad in adaptors])
        print(row)


if __name__ == "__main__":
    # Example usage:
    # python analyze_acc_ret.py raw_report.txt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_report", type=str, required=True, help="Path to the raw text report file")
    args = parser.parse_args()

    data = parse_raw_report(args.raw_report)
    overall_means, easy_means, hard_means, easy_idxs, hard_idxs = analyze_accuracy(data)
    print_report(data, overall_means, easy_means, hard_means, easy_idxs, hard_idxs)