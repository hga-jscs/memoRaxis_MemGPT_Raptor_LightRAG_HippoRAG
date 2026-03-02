import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_correct_mechanical(prediction: str, references: List[str]) -> bool:
    negative_patterns = [
        "does not contain any information",
        "insufficient information",
        "not mentioned in the context",
        "no information related to",
        "上下文没有提到",
        "没有找到相关信息",
        "信息不足",
    ]
    pred_norm = (prediction or "").lower()
    for pattern in negative_patterns:
        if pattern in pred_norm:
            return False

    pred_n = normalize_text(prediction)
    for ref in references or []:
        ref_n = normalize_text(ref)
        if ref_n and ref_n in pred_n:
            return True
    return False


def load_ground_truth_acc_ret(instance_idx: int) -> Dict[str, List[str]]:
    """
    返回 qa_map: question -> [answers...]
    """
    gt_path = Path(f"MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_{instance_idx}.json")
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth not found: {gt_path}. "
            f"Run: python scripts/LightRAG_MAB/data/convert_parquet_to_json.py"
        )

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    qa_map: Dict[str, List[str]] = {}
    for q, a in zip(gt.get("questions", []), gt.get("answers", [])):
        qa_map[str(q)] = a if isinstance(a, list) else [str(a)]
    return qa_map


def score_and_annotate(results: Dict[str, List[dict]], qa_map: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    - 给每条 prediction item 加 is_correct: bool
    - 返回 summary: adaptor -> {accuracy, correct_count, total_questions}
    """
    summary: Dict[str, Any] = {}

    for adaptor_name, preds in results.items():
        correct = 0
        total = len(preds)

        for item in preds:
            q = item.get("question", "")
            pred = item.get("answer", "")
            refs = qa_map.get(q, [])
            ok = is_correct_mechanical(pred, refs)
            item["is_correct"] = bool(ok)
            if ok:
                correct += 1

        acc = (correct / total) if total > 0 else 0.0
        summary[adaptor_name] = {
            "accuracy": acc,
            "correct_count": correct,
            "total_questions": total,
        }

    return summary


def evaluate_adaptor(name: str, adaptor, questions: list, limit: int) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        try:
            res: AdaptorResult = adaptor.run(q)
            results.append(
                {
                    "question": q,
                    "answer": res.answer,
                    "steps": res.steps_taken,
                    "tokens": res.token_consumption,
                    "replan": res.replan_count,
                }
            )
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_one_instance(
    instance_idx: int,
    adaptors_to_run: List[str],
    limit: int,
    storage_dir: str,
    mode: str,
    output_suffix: str = "",
    print_scores: bool = True,
):
    logger.info(f"=== Evaluating Instance {instance_idx} (LightRAG) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    questions = list(data["questions"])

    workspace = Path(storage_dir) / f"lightrag_acc_ret_{instance_idx}"
    if not workspace.exists():
        logger.error(f"LightRAG workspace not found: {workspace} (run ingest first)")
        return

    logger.info(f"Using LightRAG workspace: {workspace}")
    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)

    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"],
    )

    results = {}
    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        results["R1"] = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit)
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        results["R2"] = evaluate_adaptor("R2", IterativeAdaptor(llm, memory), questions, limit)
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        results["R3"] = evaluate_adaptor("R3", PlanAndActAdaptor(llm, memory), questions, limit)

    final_report = {
        "dataset": "Accurate_Retrieval",
        "instance_idx": instance_idx,
        "results": results,
    }

    # ✅ 机械评分：直接在 infer 后算并打印
    try:
        qa_map = load_ground_truth_acc_ret(instance_idx)
        score_summary = score_and_annotate(final_report["results"], qa_map)
        final_report["scores"] = {"mechanical_accuracy": score_summary}

        if print_scores:
            print(f"\n--- [On-the-fly Mechanical Score: acc_ret instance {instance_idx}] ---")
            for adaptor, stats in score_summary.items():
                print(
                    f"Adaptor {adaptor:3}: Accuracy = {stats['accuracy']:.2%} "
                    f"({stats['correct_count']}/{stats['total_questions']})"
                )
    except Exception as e:
        final_report["scores"] = {"mechanical_accuracy": "N/A", "error": str(e)}
        if print_scores:
            print(f"\n--- [On-the-fly Mechanical Score: N/A] ---\nReason: {e}")

    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    filename = f"acc_ret_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench (LightRAG)")
    parser.add_argument("--adaptor", nargs="+", default=["all"], choices=["R1", "R2", "R3", "all"])
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--storage_dir", type=str, default="out/lightrag_storage")
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "mix", "local", "global", "hybrid"])
    parser.add_argument("--output_suffix", type=str, default="lightrag")
    parser.add_argument("--no_print_scores", action="store_true", help="Disable printing mechanical scores")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(
            instance_idx=idx,
            adaptors_to_run=args.adaptor,
            limit=args.limit,
            storage_dir=args.storage_dir,
            mode=args.mode,
            output_suffix=args.output_suffix,
            print_scores=not args.no_print_scores,
        )


if __name__ == "__main__":
    main()