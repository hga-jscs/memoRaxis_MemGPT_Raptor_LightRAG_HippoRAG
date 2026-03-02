import argparse
import sys
import re
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def chunk_dialogues(context: str) -> List[str]:
    """
    策略 A: 针对 Dialogue N: 格式的正则切分
    """
    parts = re.split(r"\n(Dialogue \d+:)", "\n" + context)
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)
    return chunks


def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    """
    策略 B: 累积切分 (复用 Conflict Resolution 的逻辑)
    """
    lines = [line.strip() for line in context.split("\n") if line.strip()]
    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_length = 0

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


def ingest_one_instance(
    instance_idx: int,
    save_dir: str,
    mode: str,
    reset: bool,
):
    logger.info(f"=== Processing TTL Instance {instance_idx} (LightRAG) ===")

    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    context = data["context"]

    # 自适应选择策略
    if "Dialogue 1:" in context[:500]:
        logger.info("Strategy: Regex Split (Dialogue mode)")
        chunks = chunk_dialogues(context)
    else:
        logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
        chunks = chunk_accumulation(context, min_chars=800)

    logger.info(f"Preparing LightRAG workspace with {len(chunks)} chunks (instance={instance_idx})")

    # 每个 instance 一个 LightRAG 工作区目录
    workspace = Path(save_dir) / f"lightrag_ttl_{instance_idx}"
    workspace.mkdir(parents=True, exist_ok=True)

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    if reset:
        memory.reset()

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks) into {workspace} ...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 50 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    # LightRAG：build_index 会把索引持久化到 working_dir（无需 .pkl）
    memory.build_index(doc_id=f"ttl_{instance_idx}")

    print(f"\nInstance {instance_idx} complete. LightRAG workspace saved -> {workspace}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="Index range (e.g., '0-5')")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/lightrag_storage",
        help="Where to save LightRAG workspaces",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        choices=["naive", "mix", "local", "global", "hybrid"],
        help="LightRAG query mode (start with naive)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete workspace dir before ingest (for clean reruns)",
    )
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(
            instance_idx=idx,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
        )


if __name__ == "__main__":
    main()