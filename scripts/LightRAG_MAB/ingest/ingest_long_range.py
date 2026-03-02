import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, chunk_context
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    overlap: int,
    save_dir: str,
    mode: str,
    reset: bool,
):
    logger.info(f"=== Processing Long_Range_Understanding Instance {instance_idx} (LightRAG) ===")

    # 使用预先转换好的 JSON 文件
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"

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

    # 使用滑动窗口切分
    chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)

    logger.info(f"Preparing LightRAG workspace with {len(chunks)} chunks (instance={instance_idx})")

    # 每个 instance 一个 LightRAG 工作区目录
    workspace = Path(save_dir) / f"lightrag_long_range_{instance_idx}"
    workspace.mkdir(parents=True, exist_ok=True)

    # Initialize LightRAG Memory
    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    if reset:
        memory.reset()

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks) into {workspace} ...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 100 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    # LightRAG：build_index 会把索引持久化到 working_dir（无需 .pkl）
    memory.build_index(doc_id=f"long_range_{instance_idx}")

    print(f"\nInstance {instance_idx} complete. LightRAG workspace saved -> {workspace}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data (LightRAG)")
    # 默认 Top 40 (0-39)
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")

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
    logger.info(f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}, Mode={args.mode}")

    for idx in indices:
        ingest_one_instance(
            instance_idx=idx,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
        )


if __name__ == "__main__":
    main()