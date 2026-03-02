import argparse
import sys
import re
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

def chunk_dialogues(context: str) -> List[str]:
    """
    策略 A: 针对 Dialogue N: 格式的正则切分
    """
    parts = re.split(r'\n(Dialogue \d+:)', '\n' + context)
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)
    return chunks

def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    """
    策略 B: 累积切分 (复用 Conflict Resolution 的逻辑)
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
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

def ingest_one_instance(instance_idx: int, save_dir: str, tb_num_layers: int):
    logger.info(f"=== Processing TTL Instance {instance_idx} (RAPTOR) ===")
    
    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
    
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

    context = data["context"]
    
    # 自适应选择策略
    if "Dialogue 1:" in context[:500]:
        logger.info("Strategy: Regex Split (Dialogue mode)")
        chunks = chunk_dialogues(context)
    else:
        logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
        chunks = chunk_accumulation(context, min_chars=800)

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")

    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 50 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)
            
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / f"raptor_ttl_{instance_idx}.pkl"
    memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="Index range (e.g., '0-5')")
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.save_dir, args.tb_num_layers)

if __name__ == "__main__":
    main()
