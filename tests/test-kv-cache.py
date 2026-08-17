import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TransformerConfig, TransformerLM


def test_kv_cache_correctness():
    # Set seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cpu")

    config = TransformerConfig()
    model = TransformerLM(config).to(device)
    model.eval()

    # 1. Stress test with a longer 32-token sequence
    tokens = torch.randint(0, config.vocab_size, (1, 32), device=device)

    # Test A: Standard uncached full sequence forward pass
    with torch.no_grad():
        normal_logits = model(tokens, use_cache=False)

    # Test B: Incremental cached forward pass
    model.reset_cache(batch_size=1, device=device)
    outputs = []

    with torch.no_grad():
        for i in range(tokens.size(1)):
            single_token = tokens[:, i : i + 1]
            logits = model(single_token, use_cache=True)
            outputs.append(logits)

    cached_logits = torch.cat(outputs, dim=1)

    # Verify output equivalence over 32 tokens
    max_diff = (normal_logits - cached_logits).abs().max().item()
    is_close = torch.allclose(normal_logits, cached_logits, atol=1e-5, rtol=1e-5)

    assert is_close, f"KV Cache logits mismatch! Maximum absolute difference: {max_diff}"
    print(f"✓ Test 1 Passed: 32-token uncached vs cached match (Max Diff: {max_diff:.2e}).")

    # 2. Test cache reset functionality to ensure state isn't polluted
    model.reset_cache(batch_size=1, device=device)

    with torch.no_grad():
        first_token_logits = model(tokens[:, :1], use_cache=True)

    reset_diff = (normal_logits[:, :1] - first_token_logits).abs().max().item()
    reset_is_close = torch.allclose(
        normal_logits[:, :1], first_token_logits, atol=1e-5, rtol=1e-5
    )

    assert reset_is_close, f"Cache reset failed! Difference on first token: {reset_diff}"
    print(f"✓ Test 2 Passed: Cache successfully reset state for new sequence.")


if __name__ == "__main__":
    print("Running KV Cache Correctness Test Suite...\n" + "=" * 50)
    test_kv_cache_correctness()
    print("=" * 50 + "\nAll KV cache tests passed successfully!")