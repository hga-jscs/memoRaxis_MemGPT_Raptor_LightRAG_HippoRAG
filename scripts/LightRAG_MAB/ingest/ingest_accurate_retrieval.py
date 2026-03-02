import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    save_dir: str,
    mode: str,
    reset: bool,
):
    logger.info(f"=== Processing Instance {instance_idx} (LightRAG) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size)

    # LightRAG workspace dir for this instance
    out_dir = Path(save_dir) / f"lightrag_acc_ret_{instance_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LightRAG Memory
    memory = LightRAGMemory(working_dir=str(out_dir), mode=mode)
    if reset:
        memory.reset()

    print(f"Starting ingestion of {len(chunks)} chunks into {out_dir} ...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"Queued {i}/{len(chunks)} chunks...", end="\r", flush=True)

    # Build index & persist into working_dir (no .pkl file)
    memory.build_index(doc_id=f"acc_ret_{instance_idx}")

    print(f"\nIngestion complete. LightRAG workspace saved at: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (LightRAG)")
    parser.add_argument(
        "--instance_idx",
        type=str,
        default="0",
        help="Index range (e.g., '0', '0-5', '1,3')",
    )
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
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
            chunk_size=args.chunk_size,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
        )


if __name__ == "__main__":
    main()