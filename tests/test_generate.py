import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import generate, apply_top_k, apply_top_p
from src.model import TransformerConfig, TransformerLM


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text):
        if not text:
            return []
        return [int(x) for x in text.split()]

    def decode(self, token_ids):
        return " ".join(str(x) for x in token_ids)


@pytest.fixture
def setup_model():
    torch.manual_seed(42)
    config = TransformerConfig(
        vocab_size=100, context_length=128, n_layers=2, d_model=64, n_heads=4
    )
    model = TransformerLM(config)
    tokenizer = DummyTokenizer()
    return model, tokenizer


def test_top_k_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = apply_top_k(logits, top_k=2)

    assert torch.isinf(filtered[0, 0])
    assert torch.isinf(filtered[0, 1])
    assert torch.isinf(filtered[0, 2])
    assert filtered[0, 3] == 4.0
    assert filtered[0, 4] == 5.0


def test_top_p_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = apply_top_p(logits, top_p=0.8)

    assert torch.isinf(filtered[0, 0])
    assert torch.isinf(filtered[0, 1])
    assert torch.isinf(filtered[0, 2])
    assert filtered[0, 3] == 4.0
    assert filtered[0, 4] == 5.0


def test_greedy_determinism(setup_model):
    model, tokenizer = setup_model
    prompt = "10 20 30"

    res1 = generate(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)
    res2 = generate(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)

    assert res1 == res2


def test_seed_reproducibility(setup_model):
    model, tokenizer = setup_model
    prompt = "10 20 30"

    res1 = generate(model, tokenizer, prompt, max_new_tokens=10, temperature=1.5, seed=123)
    res2 = generate(model, tokenizer, prompt, max_new_tokens=10, temperature=1.5, seed=123)
    res3 = generate(model, tokenizer, prompt, max_new_tokens=10, temperature=1.5, seed=999)

    assert res1 == res2
    assert res1 != res3


def test_sampling_options(setup_model):
    model, tokenizer = setup_model
    prompt = "5 15 25"

    res_k = generate(
        model, tokenizer, prompt, max_new_tokens=5, temperature=0.9, top_k=10
    )
    res_p = generate(
        model, tokenizer, prompt, max_new_tokens=5, temperature=0.9, top_p=0.85
    )
    res_kp = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=5,
        temperature=0.9,
        top_k=10,
        top_p=0.85,
    )

    for res in [res_k, res_p, res_kp]:
        assert isinstance(res, str)
        assert res.startswith(prompt)
        assert len(res.split()) == 8 


def test_cached_vs_uncached_agreement(setup_model):
    model, tokenizer = setup_model
    prompt = "1 2 3 4 5"

    res_cached = generate(
        model, tokenizer, prompt, max_new_tokens=100, temperature=0.0, use_cache=True
    )
    res_uncached = generate(
        model, tokenizer, prompt, max_new_tokens=100, temperature=0.0, use_cache=False
    )

    assert res_cached == res_uncached, "Cache vs No-Cache trajectory mismatch at step <= 100!"
    print("✓ Test Passed: 100-token greedy continuation is bitwise identical (cache == no-cache).")


def test_max_new_tokens_zero(setup_model):
    model, tokenizer = setup_model
    prompt = "1 2 3"

    res = generate(model, tokenizer, prompt, max_new_tokens=0)
    assert res == prompt


def test_eos_stopping(setup_model):
    model, tokenizer = setup_model
    prompt = "1 2 3"

    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[0, :] = 100.0

    res = generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.0)
    assert res.split()[-1] == "0"
    assert len(res.split()) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])