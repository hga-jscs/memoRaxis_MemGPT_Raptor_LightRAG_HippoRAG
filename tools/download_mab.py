from pathlib import Path
from datasets import load_dataset

DS_NAME = "ai-hyz/MemoryAgentBench"

SPLIT_TO_FILE = {
    "Accurate_Retrieval": "Accurate_Retrieval-00000-of-00001.parquet",
    "Conflict_Resolution": "Conflict_Resolution-00000-of-00001.parquet",
    "Long_Range_Understanding": "Long_Range_Understanding-00000-of-00001.parquet",
    "Test_Time_Learning": "Test_Time_Learning-00000-of-00001.parquet",
}

out_dir = Path("MemoryAgentBench/data")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
ds = load_dataset(DS_NAME)  # default config
print("available splits =", list(ds.keys()))

for split, fname in SPLIT_TO_FILE.items():
    if split not in ds:
        raise RuntimeError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    print(f"\nExporting split={split} rows={len(ds[split])} ...")
    df = ds[split].to_pandas()
    out_path = out_dir / fname
    df.to_parquet(out_path, index=False)
    print("saved ->", out_path)

print("\n✅ All parquet files saved into MemoryAgentBench/data/")

