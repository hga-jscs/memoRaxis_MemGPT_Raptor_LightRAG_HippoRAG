import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.hipporag_memory import HippoRAGMemory

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    save_root: str,
    openie_mode: str,
    force_rebuild: bool,
    max_chunks: int,
):
    logger.info(f"=== Processing Instance {instance_idx} (HippoRAG) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size)

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    # 每个 instance 独立索引目录
    index_dir = Path(save_root) / f"hipporag_acc_ret_{instance_idx}"

    memory = HippoRAGMemory(
        index_dir=str(index_dir),
        openie_mode=openie_mode,
        force_rebuild=force_rebuild,
    )

    print(f"Starting ingestion of {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"Queued {i}/{len(chunks)} chunks...", end="\r", flush=True)

    # build index on disk
    memory.build_index()

    print(f"\nIngestion complete. HippoRAG index saved to {index_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices", help="Where to save HippoRAG indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode")
    parser.add_argument("--force_rebuild", action="store_true", help="Delete existing index_dir and rebuild")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Debug: only index first N chunks (-1 = all)")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks)


if __name__ == "__main__":
    main()