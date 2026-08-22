import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

#Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import (
    END_OF_TEXT,
    BPETokenizer,
    count_pretokens,
    merge_word,
    pretokenize,
    train_bpe,
)


#Reference (Slow) BPE Implementation for Verification
def train_bpe_reference(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Simple reference implementation that recounts pair frequencies from scratch every step."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pretoken_counts = count_pretokens(text, special_tokens=special_tokens)
    word_freqs = {
        tuple(bytes([b]) for b in pt.encode("utf-8")): count
        for pt, count in pretoken_counts.items()
    }

    vocab = {}
    next_id = 0
    for spec in special_tokens:
        vocab[next_id] = spec.encode("utf-8")
        next_id += 1

    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    merges = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        pair_counts = Counter()
        for symbols, freq in word_freqs.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += freq

        if not pair_counts:
            break

        best_count = max(pair_counts.values())
        best_pair = max(
            pair for pair, count in pair_counts.items() if count == best_count
        )

        word_freqs = {
            merge_word(word, best_pair): freq
            for word, freq in word_freqs.items()
        }

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

    return vocab, merges


#Serialization Helper for Testing from files
def save_tokenizer_files(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    vocab_json = {str(k): v.hex() for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")


#Test Suite
def test_1_pretokenisation():
    text = "Hello, world!"
    tokens = pretokenize(text)
    expected = ["Hello", ",", " world", "!"]
    assert tokens == expected, f"Expected {expected}, got {tokens}"
    print("✓ Test 1 Passed: Pretokenisation matches GPT-2 regex expectations.")


def test_2_special_tokens():
    text = f"Hello{END_OF_TEXT}world"
    counts = count_pretokens(text, special_tokens=[END_OF_TEXT])
    assert END_OF_TEXT not in counts, "Special token leaked into pre-token counts."
    assert set(counts.keys()) == {"Hello", "world"}, f"Unexpected pre-tokens: {counts}"
    print("✓ Test 2 Passed: Special tokens correctly isolated.")


def test_3_incremental_vs_reference_bpe():
    tiny_corpus = (
        "low low low low\n"
        "lower lower widest widest\n"
        "newest newest newest\n"
        "Hello world! Testing BPE optimization."
    )
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write(tiny_corpus)
        tmp_path = tmp.name

    try:
        ref_vocab, ref_merges = train_bpe_reference(
            tmp_path, vocab_size=280, special_tokens=[END_OF_TEXT]
        )
        opt_vocab, opt_merges = train_bpe(
            tmp_path, vocab_size=280, special_tokens=[END_OF_TEXT]
        )

        assert ref_vocab == opt_vocab, "Vocabularies differ between reference and incremental BPE!"
        assert ref_merges == opt_merges, "Merges differ between reference and incremental BPE!"
        print("✓ Test 3 Passed: Incremental BPE produced byte-identical merges to reference implementation.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_4_encode_decode_round_trip():
    tiny_corpus = "Hello world! This is a test. 12345 Unicode: 🚀 café"
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write(tiny_corpus)
        tmp_path = tmp.name

    try:
        vocab, merges = train_bpe(
            tmp_path, vocab_size=290, special_tokens=[END_OF_TEXT]
        )
        tokenizer = BPETokenizer(vocab, merges, special_tokens=[END_OF_TEXT])

        test_strings = [
            "Hello world!",
            "This is a test.",
            "12345",
            "Unicode: 🚀 café",
        ]
        for text in test_strings:
            decoded = tokenizer.decode(tokenizer.encode(text))
            assert decoded == text, f"Round-trip failed for '{text}'. Got: '{decoded}'"
        print("✓ Test 4 Passed: Encode/decode round trip verified (including multi-byte UTF-8).")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_5_special_token_round_trip():
    tiny_corpus = f"Hello{END_OF_TEXT}world"
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write(tiny_corpus)
        tmp_path = tmp.name

    try:
        vocab, merges = train_bpe(
            tmp_path, vocab_size=270, special_tokens=[END_OF_TEXT]
        )
        tokenizer = BPETokenizer(vocab, merges, special_tokens=[END_OF_TEXT])

        text = f"Hello{END_OF_TEXT}world"
        encoded = tokenizer.encode(text)
        special_id = tokenizer.special_to_id[END_OF_TEXT]

        assert special_id in encoded, "Special token ID missing from stream."
        assert encoded.count(special_id) == 1, "Special token count mismatch."
        assert tokenizer.decode(encoded) == text, "Special token round-trip mismatch."
        print("✓ Test 5 Passed: Special token round-trip verified.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_6_encode_iterable():
    tiny_corpus = "Hello world! Iteration streaming test."
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write(tiny_corpus)
        tmp_path = tmp.name

    try:
        vocab, merges = train_bpe(tmp_path, vocab_size=275, special_tokens=[END_OF_TEXT])
        tokenizer = BPETokenizer(vocab, merges, special_tokens=[END_OF_TEXT])

        chunks = ["Hello", " world!", f"{END_OF_TEXT}Streaming ", "123"]
        
        streamed_ids = list(tokenizer.encode_iterable(chunks))
        expected_ids = []
        for chunk in chunks:
            expected_ids.extend(tokenizer.encode(chunk))

        assert streamed_ids == expected_ids, f"Streamed IDs mismatched! Streamed: {streamed_ids}, Expected: {expected_ids}"
        print("✓ Test 6 Passed: encode_iterable streaming yields identical tokens.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_7_from_files():
    tiny_corpus = "Testing tokenizer serialization from files."
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write(tiny_corpus)
        corpus_path = tmp.name

    vocab_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
    merges_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
    vocab_path = vocab_file.name
    merges_path = merges_file.name
    vocab_file.close()
    merges_file.close()

    try:
        vocab, merges = train_bpe(corpus_path, vocab_size=275, special_tokens=[END_OF_TEXT])
        
        save_tokenizer_files(vocab, merges, vocab_path, merges_path)

        tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=[END_OF_TEXT])

        sample_text = "Testing serialization round trip!"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == sample_text, f"Deserialized tokenizer failed round-trip! Got '{decoded}'"
        print("✓ Test 7 Passed: BPETokenizer.from_files loading and encoding verified.")
    finally:
        Path(corpus_path).unlink(missing_ok=True)
        Path(vocab_path).unlink(missing_ok=True)
        Path(merges_path).unlink(missing_ok=True)

def test_8_validation_100_docs_round_trip():
    train_path = PROJECT_ROOT / "data" / "TinyStories_train_part1.txt"
    val_path = PROJECT_ROOT / "data" / "TinyStories_val.txt"

    if not val_path.exists() or not train_path.exists():
        print("Test 8 Skipped: data/TinyStories_train.txt or TinyStories_val.txt missing.")
        return

    print("Training temporary tokenizer for Test 8...")
    vocab, merges = train_bpe(
        str(train_path), vocab_size=1000, special_tokens=[END_OF_TEXT]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.json"
        merges_path = Path(tmpdir) / "merges.txt"
        save_tokenizer_files(vocab, merges, str(vocab_path), str(merges_path))

        tokenizer = BPETokenizer.from_files(
            str(vocab_path), str(merges_path), special_tokens=[END_OF_TEXT]
        )

        with open(val_path, "r", encoding="utf-8") as f:
            content = f.read()
            docs = [d.strip() for d in content.split("<|endoftext|>") if d.strip()][:100]

        assert len(docs) == 100, f"Expected 100 docs from validation set, found {len(docs)}"

        mismatches = 0
        for idx, doc in enumerate(docs):
            encoded = tokenizer.encode(doc)
            decoded = tokenizer.decode(encoded)
            if decoded != doc:
                mismatches += 1
                print(f"Mismatch at doc {idx}:")
                print(f"  Expected: {repr(doc[:60])}")
                print(f"  Got:      {repr(decoded[:60])}")

        assert mismatches == 0, f"{mismatches}/100 validation documents failed round-trip!"
        print("✓ Test 8 Passed: 100 real validation documents passed encode -> decode round-trip.")


if __name__ == "__main__":
    print("Running Tokenizer Test Suite...\n" + "=" * 40)
    test_1_pretokenisation()
    test_2_special_tokens()
    test_3_incremental_vs_reference_bpe()
    test_4_encode_decode_round_trip()
    test_5_special_token_round_trip()
    test_6_encode_iterable()
    test_7_from_files()
    test_8_validation_100_docs_round_trip()
    print("=" * 40 + "\nAll tests passed successfully!")