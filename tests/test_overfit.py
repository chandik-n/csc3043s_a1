import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TransformerLM, TransformerConfig
from src.data import MemmapDataset

 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TransformerConfig(
        vocab_size=4000,
        context_length=256,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=352,
        use_rmsnorm=True,
        use_rope=True,
        ffn_type="swiglu",
    )

    model = TransformerLM(config).to(device)

    dataset = MemmapDataset(
        "data/train_4k.bin",
        context_length=256,
    )

    batch_indices = np.arange(8)
    x, y = dataset[batch_indices]
    x = x.to(device)
    y = y.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model.train()

    for step in range(1, 501):
        optimizer.zero_grad()

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")

        if loss.item() < 0.1:
            print(f"\nPASS: loss reached {loss.item():.6f} at step {step}")
            return

    raise AssertionError(
        f"Overfit test failed: loss was only {loss.item():.6f} after 500 steps"
    )


if __name__ == "__main__":
    main()