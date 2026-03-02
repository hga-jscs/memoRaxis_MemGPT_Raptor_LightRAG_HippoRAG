import argparse
import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, chunk_context
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()

def ingest_one_instance(instance_idx: int, chunk_size: int, overlap: int, save_dir: str, tb_num_layers: int):
    logger.info(f"=== Processing Long_Range_Understanding Instance {instance_idx} (RAPTOR) ===")
    
    # 使用预先转换好的 JSON 文件
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    # 使用滑动窗口切分
    chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")

    # Initialize RAPTOR Memory
    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 100 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)
            
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / f"raptor_long_range_{instance_idx}.pkl"
    memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data (RAPTOR)")
    # 默认 Top 40 (0-39)
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}, Layers={args.tb_num_layers}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.overlap, args.save_dir, args.tb_num_layers)

if __name__ == "__main__":
    main()
