import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import END_OF_TEXT, BPETokenizer, train_bpe
from tests.test_tokenizer import save_tokenizer_files

BENCHMARK_BYTES = 100 * 1024 * 1024  # 100 MB
TOTAL_DATASET_GB = 2.2

def run_benchmark():
    train_path = PROJECT_ROOT / "data" / "TinyStories_train_part1.txt"
    vocab_file = PROJECT_ROOT / "data" / "vocab_4k.json"
    merges_file = PROJECT_ROOT / "data" / "merges_4k.txt"

    if not train_path.exists():
        print(f"Error: {train_path} missing.")
        return

    # Train and save V=4000 assets if missing
    if not (vocab_file.exists() and merges_file.exists()):
        print("Training V=4000 tokenizer on part1...")
        vocab, merges = train_bpe(
            str(train_path), vocab_size=4000, special_tokens=[END_OF_TEXT]
        )
        save_tokenizer_files(vocab, merges, str(vocab_file), str(merges_file))
        print("✓ Saved vocab_4k.json and merges_4k.txt to data/")

    print("Loading V=4000 Tokenizer...")
    tokenizer = BPETokenizer.from_files(
        str(vocab_file), str(merges_file), special_tokens=[END_OF_TEXT]
    )

    print(f"Reading first 100 MB from {train_path.name}...")
    with open(train_path, "r", encoding="utf-8") as f:
        raw_bytes = f.read(BENCHMARK_BYTES)

    actual_bytes = len(raw_bytes.encode("utf-8"))
    actual_mb = actual_bytes / (1024 * 1024)

    print(f"Benchmarking encoding on {actual_mb:.2f} MB of text...")
    t0 = time.time()
    tokens = tokenizer.encode(raw_bytes)
    elapsed_time = time.time() - t0

    throughput_mb_s = actual_mb / elapsed_time
    tokens_per_sec = len(tokens) / elapsed_time
    
    # Projection for 2.2 GB
    multiplier = (TOTAL_DATASET_GB * 1024) / actual_mb
    estimated_total_seconds = elapsed_time * multiplier
    estimated_total_minutes = estimated_total_seconds / 60

    print("=" * 60)
    print("ENCODING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Sample Size:         {actual_mb:.2f} MB")
    print(f"Tokens Generated:    {len(tokens):,}")
    print(f"Encoding Time:       {elapsed_time:.2f} seconds")
    print(f"Throughput:          {throughput_mb_s:.2f} MB/s ({tokens_per_sec:,.0f} tokens/s)")
    print("-" * 60)
    print(f"ESTIMATED FULL TIME ({TOTAL_DATASET_GB} GB): ~{estimated_total_minutes:.2f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()