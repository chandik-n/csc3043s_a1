import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_plot():
    logs_dir = PROJECT_ROOT / "logs"
    csv_path = logs_dir / "vocab_study_results.csv"
    output_path = logs_dir / "vocab_compression_curve.png"

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist. Run scripts/vocab_study.py first!")
        return

    vocab_sizes = []
    bytes_per_token = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab_sizes.append(int(row["vocab_size"]))
            bytes_per_token.append(float(row["bytes_per_token"]))

    plt.figure(figsize=(8, 5))
    plt.plot(vocab_sizes, bytes_per_token, marker="o", linewidth=2, color="#1f77b4")

    # Annotate chosen vocabulary size
    plt.axvline(x=4000, color="r", linestyle="--", alpha=0.7, label="Chosen V = 4,000")

    plt.title("BPE Vocabulary Size vs. Compression Efficiency", fontsize=12, fontweight="bold")
    plt.xlabel("Vocabulary Size (V)", fontsize=10)
    plt.ylabel("Bytes / Token", fontsize=10)
    plt.xticks(vocab_sizes)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot generated directly from CSV and saved to: {output_path}")

if __name__ == "__main__":
    generate_plot()