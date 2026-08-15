import re
import time
from collections import Counter

END_OF_TEXT = "<|endoftext|>"
WORD_START = "▁"
UNK = "<|unk|>"

PRETOKEN_RE = re.compile(r"(\s*)(\w+|[^\w\s])")

def pretokenize(text):
    pretokens = []
    for match in PRETOKEN_RE.finditer(text):
        whitespace, body = match.group(1), match.group(2)
        # A pre-token starts a new word if whitespace came before it. We treat the start of a
        # document as if it were preceded by a space, so that the first word is marked too.
        starts_word = bool(whitespace) or match.start() == 0
        pretokens.append(WORD_START + body if starts_word else body)
    return pretokens

def count_pretokens(text, special_tokens=(END_OF_TEXT,)):
    counts = Counter()
    split_pattern = "|".join(re.escape(s) for s in special_tokens)
    for document in re.split(split_pattern, text):
        counts.update(pretokenize(document))
    return counts

def count_pairs(word_freqs):
    pair_counts = Counter()
    for symbols, freq in word_freqs.items():
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freq
    return pair_counts

def merge_word(symbols, pair):
    merged = []
    i = 0
    while i < len(symbols):
        # If the pair starts here, append the concatenated symbol and skip forward by 2.
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        # Otherwise keep the current symbol and advance by 1.
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)

def train_bpe(text, vocab_size, special_tokens=(END_OF_TEXT, UNK), verbose=False):
    # Represent each distinct pre-token as a tuple of single characters, with its corpus count.
    pretoken_counts = count_pretokens(text)
    word_freqs = {tuple(bytes([b]) for b in pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}
    # Initial vocabulary: the special tokens, then every character seen in the corpus.
    base_bytes = [bytes([b]) for b in range(256)]
    vocab = list(special_tokens) + base_bytes
    merges = []

    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is smaller than the minimum "
                         f"vocabulary size of{len(vocab)} "
                         f"({len(special_tokens)} special tokens + 256 base bytes)"
                        )  
    if verbose:
        print(f"{len(word_freqs):,} distinct pre-tokens, 256 base bytes, "
              f"{num_merges} merges to learn")

    start = time.time()
    for step in range(num_merges):
        # Step 1: count every adjacent pair of symbols in the corpus.
        pair_counts = count_pairs(word_freqs)
        if not pair_counts:
            break

        # Step 2: pick the most frequent pair, breaking ties lexicographically-greatest.
        best_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == best_count)

        # Step 3: apply the merge everywhere in the corpus.
        word_freqs = {merge_word(word, best_pair): freq for word, freq in word_freqs.items()}

        # Step 4: record the merge and add the new symbol to the vocabulary.
        merges.append(best_pair)
        vocab.append(best_pair[0] + best_pair[1])

        if verbose and (step + 1) % 200 == 0:
            print(f"  merge {step + 1:5d}: {best_pair[0]!r} + {best_pair[1]!r} -> "
                  f"{vocab[-1]!r} (count {best_count:,})   [{time.time() - start:.1f}s]")

    return {i: token for i, token in enumerate(vocab)}, merges

toy_corpus = (
    "low low low low\n"
    "lower lower widest widest\n"
    "newest newest newest"
)

vocab, merges = train_bpe(
    toy_corpus,
    vocab_size=270,
    verbose=True
)

print("\nVocabulary:")
print(vocab)

print("\nMerges:")
print(merges)