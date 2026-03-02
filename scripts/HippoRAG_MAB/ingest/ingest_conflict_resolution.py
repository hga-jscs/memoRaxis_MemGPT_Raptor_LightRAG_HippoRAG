import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.hipporag_memory import HippoRAGMemory

logger = get_logger()


def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    """
    Conflict Resolution 专用切分策略：
    按行读取 Fact，累积直到缓冲区字符数 > min_chars，然后作为一个 Chunk。
    """
    lines = [line.strip() for line in context.split("\n") if line.strip()]

    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            # 形成一个 chunk
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            # 重置缓冲区
            current_chunk_lines = []
            current_length = 0

    # 处理剩余的缓冲区
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)

    return chunks


def ingest_one_instance(
    instance_idx: int,
    min_chars: int,
    save_root: str,
    openie_mode: str,
    force_rebuild: bool,
    max_chunks: int,
):
    logger.info(f"=== Processing Conflict_Resolution Instance {instance_idx} (HippoRAG) ===")

    # 注意：这里读取的是 JSON preview
    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"

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

    # 使用专用切分策略
    chunks = chunk_facts(data["context"], min_chars=min_chars)

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    # 每个 instance 独立索引目录
    index_dir = Path(save_root) / f"hipporag_conflict_{instance_idx}"

    memory = HippoRAGMemory(
        index_dir=str(index_dir),
        openie_mode=openie_mode,
        force_rebuild=force_rebuild,
    )

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    memory.build_index()

    print(f"\nInstance {instance_idx} complete. HippoRAG index saved -> {index_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--min_chars", type=int, default=800, help="Minimum chars per chunk")
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices", help="Where to save HippoRAG indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode")
    parser.add_argument("--force_rebuild", action="store_true", help="Delete existing index_dir and rebuild")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Debug: only index first N chunks (-1 = all)")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.min_chars, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks)


if __name__ == "__main__":
    main()