import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import END_OF_TEXT, BPETokenizer, train_bpe
from tests.test_tokenizer import save_tokenizer_files


def encode_file(input_txt: Path, output_bin: Path, tokenizer: BPETokenizer):
    print(f"Encoding {input_txt.name}...")
    with open(input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    # uint16 supports vocabulary IDs up to 65,535 (V = 4,000 fits comfortably)
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(output_bin)
    print(
        f"Saved {len(arr):,} tokens to {output_bin.name} ({output_bin.stat().st_size / 1e6:.2f} MB)"
    )


def main():
    data_dir = PROJECT_ROOT / "data"
    train_p1 = data_dir / "TinyStories_train_part1.txt"
    train_p2 = data_dir / "TinyStories_train_part2.txt"
    val_txt = data_dir / "TinyStories_val.txt"

    vocab_file = data_dir / "vocab_4k.json"
    merges_file = data_dir / "merges_4k.txt"

    # Step 1: Ensure official V=4000 vocabulary and merges exist
    if not (vocab_file.exists() and merges_file.exists()):
        print("Training official V=4000 BPE Tokenizer...")
        vocab, merges = train_bpe(
            str(train_p1), vocab_size=4000, special_tokens=[END_OF_TEXT]
        )
        save_tokenizer_files(vocab, merges, str(vocab_file), str(merges_file))
        print("Saved vocab_4k.json and merges_4k.txt")

    # Step 2: Instantiate Tokenizer
    tokenizer = BPETokenizer.from_files(
        str(vocab_file), str(merges_file), special_tokens=[END_OF_TEXT]
    )

    # Step 3: Encode Datasets to .bin
    if val_txt.exists():
        encode_file(val_txt, data_dir / "val_4k.bin", tokenizer)

    if train_p1.exists():
        encode_file(train_p1, data_dir / "train_part1_4k.bin", tokenizer)

    if train_p2.exists():
        encode_file(train_p2, data_dir / "train_part2_4k.bin", tokenizer)


if __name__ == "__main__":
    main()