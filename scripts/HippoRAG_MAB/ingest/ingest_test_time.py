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
from src.hipporag_memory import HippoRAGMemory

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
    save_root: str,
    openie_mode: str,
    force_rebuild: bool,
    max_chunks: int,
):
    logger.info(f"=== Processing TTL Instance {instance_idx} (HippoRAG) ===")

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

    # 自适应选择策略（保持与 RAPTOR ingest 一致）
    if "Dialogue 1:" in context[:500]:
        logger.info("Strategy: Regex Split (Dialogue mode)")
        chunks = chunk_dialogues(context)
    else:
        logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
        chunks = chunk_accumulation(context, min_chars=800)

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    index_dir = Path(save_root) / f"hipporag_ttl_{instance_idx}"

    memory = HippoRAGMemory(
        index_dir=str(index_dir),
        openie_mode=openie_mode,
        force_rebuild=force_rebuild,
    )

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 50 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    memory.build_index()

    print(f"\nInstance {instance_idx} complete. HippoRAG index saved -> {index_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="Index range (e.g., '0-5')")
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices", help="Where to save HippoRAG indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode")
    parser.add_argument("--force_rebuild", action="store_true", help="Delete existing index_dir and rebuild")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Debug: only index first N chunks (-1 = all)")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks)


if __name__ == "__main__":
    main()