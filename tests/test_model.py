import sys
from pathlib import Path
import torch

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TransformerConfig, TransformerLM

def test_1_model_forward_pass():
    config = TransformerConfig()
    model = TransformerLM(config)
    model.eval()

    batch_size, seq_len = 2, 16
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert (
        logits.shape == expected_shape
    ), f"Expected logits shape {expected_shape}, got {logits.shape}"
    print(
        f"✓ Test 1 Passed: Forward pass returned expected shape {logits.shape}."
    )


def test_2_parameter_count():
    config = TransformerConfig(vocab_size=4000, d_model=512)
    model = TransformerLM(config)

    emb_params = model.token_embeddings.weight.numel()
    head_params = model.lm_head.weight.numel()
    non_emb_params = model.num_parameters(non_embedding=True)
    total_params = model.num_parameters(non_embedding=False)

    expected_emb = config.vocab_size * config.d_model  
    expected_head = config.vocab_size * config.d_model  

    assert (
        emb_params == expected_emb
    ), f"Expected embedding params {expected_emb}, got {emb_params}"
    assert (
        head_params == expected_head
    ), f"Expected LM head params {expected_head}, got {head_params}"
    assert (
        total_params == non_emb_params + emb_params + head_params
    ), "Total parameter sum mismatch."

    print("✓ Test 2 Passed: Parameter breakdown verified for Q5.")
    print(f"   • Embedding Params:     {emb_params:,}")
    print(f"   • LM Head Params:       {head_params:,}")
    print(f"   • Non-Embedding Params: {non_emb_params:,}")
    print(f"   • Total Params:         {total_params:,}")


def test_3_ablations_forward_pass():
    batch_size, seq_len = 2, 8
    dummy_input = torch.randint(0, 4000, (batch_size, seq_len))

    cfg_no_norm = TransformerConfig(use_rmsnorm=False)
    model_no_norm = TransformerLM(cfg_no_norm)
    model_no_norm.eval()

    cfg_no_rope = TransformerConfig(use_rope=False)
    model_no_rope = TransformerLM(cfg_no_rope)
    model_no_rope.eval()

    cfg_relu = TransformerConfig(ffn_type="relu")
    model_relu = TransformerLM(cfg_relu)
    model_relu.eval()

    with torch.no_grad():
        logits_no_norm = model_no_norm(dummy_input)
        logits_no_rope = model_no_rope(dummy_input)
        logits_relu = model_relu(dummy_input)

    assert logits_no_norm.shape == (
        batch_size, seq_len, cfg_no_norm.vocab_size
    )

    assert logits_no_rope.shape == (
        batch_size, seq_len, cfg_no_rope.vocab_size
    )

    assert logits_relu.shape == (
        batch_size, seq_len, cfg_relu.vocab_size
    )

    print(
        "✓ Test 3 Passed: All ablation configurations "
        "(use_rmsnorm=False, use_rope=False, ffn_type='relu') "
        "run successfully."
    )


def test_4_swiglu_vs_relu_parameter_match():
    cfg_swiglu = TransformerConfig(ffn_type="swiglu")
    model_swiglu = TransformerLM(cfg_swiglu)

    cfg_relu = TransformerConfig(ffn_type="relu")
    model_relu = TransformerLM(cfg_relu)

    swiglu_non_emb = model_swiglu.num_parameters(non_embedding=True)
    relu_non_emb = model_relu.num_parameters(non_embedding=True)

    assert (
        swiglu_non_emb == relu_non_emb
    ), f"Parameter count mismatch! SwiGLU: {swiglu_non_emb:,}, ReLU: {relu_non_emb:,}"

    print("✓ Test 4 Passed: SwiGLU and ReLU non-embedding parameter counts match exactly!")
    print(f"   • SwiGLU Non-Embedding Params: {swiglu_non_emb:,}")
    print(f"   • ReLU Non-Embedding Params:   {relu_non_emb:,}")

def test_5_kv_cache_matches_full_forward():
    config = TransformerConfig()
    model = TransformerLM(config)
    model.eval()

    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, 16)
    )

    with torch.no_grad():
        full_logits = model(tokens)

        model.reset_cache(
            batch_size=1,
            device=tokens.device,
            dtype=model.token_embeddings.weight.dtype,
        )

        cached_outputs = []

        for i in range(tokens.size(1)):
            logits = model(
                tokens[:, i:i+1],
                use_cache=True
            )
            cached_outputs.append(logits)

        cached_logits = torch.cat(cached_outputs, dim=1)

    assert torch.allclose(
        full_logits,
        cached_logits,
        atol=1e-5,
        rtol=1e-5,
    ), "KV-cache output differs from normal forward pass."

    print("✓ Test 5 Passed: KV-cache matches full forward pass.")


if __name__ == "__main__":
    print("Running TransformerLM Model Test Suite...\n" + "=" * 50)
    test_1_model_forward_pass()
    test_2_parameter_count()
    test_3_ablations_forward_pass()
    test_4_swiglu_vs_relu_parameter_match()
    test_5_kv_cache_matches_full_forward()
    print("=" * 50 + "\nAll model tests passed successfully!")