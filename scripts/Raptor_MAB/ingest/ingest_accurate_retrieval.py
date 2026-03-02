import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()

def ingest_one_instance(instance_idx: int, chunk_size: int, save_dir: str, tb_num_layers: int):
    logger.info(f"=== Processing Instance {instance_idx} (RAPTOR) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"
    
    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size)

    # Initialize RAPTOR Memory
    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    print(f"Starting ingestion of {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"Queued {i}/{len(chunks)} chunks...", end="\r", flush=True)

    # Save tree to disk
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / f"raptor_acc_ret_{instance_idx}.pkl"
    memory.save_tree(str(tree_path))

    print(f"\nIngestion complete. RAPTOR tree saved to {tree_path}.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.save_dir, args.tb_num_layers)

if __name__ == "__main__":
    main()
