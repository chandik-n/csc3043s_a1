import json
from collections import Counter, defaultdict
from typing import Iterator
import regex

#Appendix A: GPT-2 pre-tokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)

END_OF_TEXT = "<|endoftext|>"


def pretokenize(text):
    return [match.group(0) for match in PRETOKEN_RE.finditer(text)]


def count_pretokens(text, special_tokens=(END_OF_TEXT,)):
    counts = Counter()
    if special_tokens:
        split_pattern = "(" + "|".join(regex.escape(s) for s in special_tokens) + ")"
        documents = [p for p in regex.split(split_pattern, text) if p and p not in special_tokens]
    else:
        documents = [text]

    for document in documents:
        counts.update(pretokenize(document))
    return counts

def merge_word(symbols, pair):
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = (END_OF_TEXT,)):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pretoken_counts = count_pretokens(text, special_tokens=special_tokens)

    words = [tuple(bytes([b]) for b in pt.encode("utf-8")) for pt in pretoken_counts.keys()]
    freqs = list(pretoken_counts.values())

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

    pair_counts = Counter()
    where_pair = defaultdict(set)

    for word_idx, (word, count) in enumerate(zip(words, freqs)):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            where_pair[pair].add(word_idx)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == best_count)

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        affected_indices = list(where_pair.pop(best_pair, []))
        del pair_counts[best_pair]

        for word_idx in affected_indices:
            old_word = words[word_idx]
            count = freqs[word_idx]

            for i in range(len(old_word) - 1):
                old_p = (old_word[i], old_word[i + 1])
                pair_counts[old_p] -= count
                if pair_counts[old_p] <= 0:
                    del pair_counts[old_p]
                where_pair[old_p].discard(word_idx)

            new_word = merge_word(old_word, best_pair)
            words[word_idx] = new_word

            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i + 1])
                pair_counts[new_p] += count
                where_pair[new_p].add(word_idx)

    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_to_id = {s: self.bytes_to_id[s.encode("utf-8")] for s in self.special_tokens if s.encode("utf-8") in self.bytes_to_id}
        
        self.cache = {}

        if self.special_tokens:
            esc = [regex.escape(s) for s in self.special_tokens]
            self.special_regex = regex.compile("(" + "|".join(esc) + ")")
        else:
            self.special_regex = None

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] = None):
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p1_hex, p2_hex = line.strip().split()
                    merges.append((bytes.fromhex(p1_hex), bytes.fromhex(p2_hex)))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pt_bytes: bytes) -> list[int]:
        if pt_bytes in self.cache:
            return self.cache[pt_bytes]

        symbols = tuple(bytes([b]) for b in pt_bytes)

        while len(symbols) >= 2:
            min_pair = None
            min_rank = float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_rank == float("inf"):
                break

            symbols = merge_word(symbols, min_pair)

        ids = [self.bytes_to_id[sym] for sym in symbols]
        self.cache[pt_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        tokens = []
        if self.special_regex:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_to_id:
                tokens.append(self.special_to_id[chunk])
            else:
                for pt in pretokenize(chunk):
                    pt_bytes = pt.encode("utf-8")
                    tokens.extend(self._encode_pretoken(pt_bytes))

        return tokens

    def encode_iterable(self, iterable) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")