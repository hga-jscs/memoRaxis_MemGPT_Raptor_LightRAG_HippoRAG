import argparse
import sys
import logging
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()

def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    """
    Conflict Resolution 专用切分策略：
    按行读取 Fact，累积直到缓冲区字符数 > min_chars，然后作为一个 Chunk。
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    
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

def ingest_one_instance(instance_idx: int, min_chars: int, save_dir: str, tb_num_layers: int):
    logger.info(f"=== Processing Conflict_Resolution Instance {instance_idx} (RAPTOR) ===")
    
    # 注意：这里读取的是 JSON preview
    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
    
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

    # 使用专用切分策略
    chunks = chunk_facts(data["context"], min_chars=min_chars)

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")

    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / f"raptor_conflict_{instance_idx}.pkl"
    memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--min_chars", type=int, default=800, help="Minimum chars per chunk")
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.min_chars, args.save_dir, args.tb_num_layers)

if __name__ == "__main__":
    main()
