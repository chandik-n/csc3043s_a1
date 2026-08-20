import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import END_OF_TEXT, BPETokenizer, train_bpe


def run_vocab_study():
    train_path = PROJECT_ROOT / "data" / "TinyStories_train_part1.txt"
    val_path = PROJECT_ROOT / "data" / "TinyStories_val.txt"
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = logs_dir / "vocab_study_results.csv"

    if not train_path.exists() or not val_path.exists():
        print(f"Error: Missing data files in {PROJECT_ROOT / 'data'}")
        return

    # Read validation text for evaluation
    with open(val_path, "r", encoding="utf-8") as f:
        val_text = f.read()

    total_bytes = len(val_text.encode("utf-8"))
    vocab_sizes = [1000, 2000, 4000, 8000, 16000]

    results = []

    print("=" * 70)
    print("VOCABULARY SIZE STUDY: BPE COMPRESSION ANALYSIS")
    print(f"Validation Dataset Size: {total_bytes / 1e6:.2f} MB ({total_bytes:,} bytes)")
    print("=" * 70)
    print(f"{'Vocab Size':<12} | {'Token Count':<14} | {'Bytes/Token':<12} | {'Compress %':<12} | {'Train Time':<10}")
    print("-" * 70)

    for v in vocab_sizes:
        t0 = time.time()
        vocab, merges = train_bpe(
            str(train_path), vocab_size=v, special_tokens=[END_OF_TEXT]
        )
        train_time = time.time() - t0

        tokenizer = BPETokenizer(vocab, merges, special_tokens=[END_OF_TEXT])

        tokens = tokenizer.encode(val_text)
        token_count = len(tokens)

        bytes_per_token = total_bytes / token_count
        sequence_reduction_pct = (1.0 - (token_count / total_bytes)) * 100

        print(
            f"{v:<12} | {token_count:<14,} | {bytes_per_token:<12.3f} | {sequence_reduction_pct:<11.2f}% | {train_time:<9.1f}s"
        )

        results.append({
            "vocab_size": v,
            "token_count": token_count,
            "bytes_per_token": bytes_per_token,
            "sequence_reduction_pct": sequence_reduction_pct,
            "train_time_seconds": round(train_time, 2)
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["vocab_size", "token_count", "bytes_per_token", "sequence_reduction_pct", "train_time_seconds"])
        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    run_vocab_study()