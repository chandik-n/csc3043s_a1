import argparse
from dataclasses import asdict
import math
import os
import numpy as np
import torch
import torch.nn as nn

from src.model import TransformerLM, TransformerConfig


def compute_metrics_non_overlapping(model, memmap_data, context_length, batch_size, device, use_autocast, autocast_dtype, chars_per_token):
    model.eval()
    total_tokens_in_file = len(memmap_data)
    num_windows = (total_tokens_in_file - 1) // context_length

    if num_windows == 0:
        raise ValueError(
            f"Data file length ({total_tokens_in_file}) is too short for context_length={context_length}."
        )

    xs, ys = [], []
    for k in range(num_windows):
        start = k * context_length
        chunk = memmap_data[start : start + context_length + 1].astype(np.int64)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    xs = torch.from_numpy(np.array(xs))
    ys = torch.from_numpy(np.array(ys))

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            x_batch = xs[i : i + batch_size].to(device, non_blocking=True)
            y_batch = ys[i : i + batch_size].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = model(x_batch)
                loss_sum = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="sum",
                )

            total_nll += loss_sum.item()
            total_tokens += y_batch.numel()

    #1. Mean NLL per token (in nats)
    mean_nll_per_token = total_nll / total_tokens if total_tokens > 0 else 0.0

    #2. Perplexity (PPL)
    ppl = math.exp(mean_nll_per_token)

    #3. Bits-Per-Character (BPC)
    total_bits = total_nll / math.log(2)
    total_chars = total_tokens * chars_per_token
    bpc = total_bits / total_chars if total_chars > 0 else 0.0

    return {
        "loss": mean_nll_per_token,
        "ppl": ppl,
        "bpc": bpc,
        "total_tokens": total_tokens,
        "total_windows": num_windows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer LM over non-overlapping windows")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to held-out memmap binary file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Context length for non-overlapping windows. Defaults to model training context_length.",
    )

    parser.add_argument(
        "--chars_per_token",
        type=float,
        required=True,
        help="Average characters/bytes per token measured on your dataset (e.g. 3.98)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        autocast_dtype = torch.bfloat16
        use_autocast = True
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        use_autocast = False

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config_dict = ckpt["config"]
    config = TransformerConfig(**config_dict)

    if args.context_length is not None:
        if args.context_length != config.context_length:
            print(
                f"Notice: Overriding trained context_length ({config.context_length}) "
                f"with eval context_length ({args.context_length})."
            )
        config.context_length = args.context_length

    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    eval_memmap = np.memmap(args.eval_data, dtype=np.uint16, mode="r")

    print(f"Evaluating {len(eval_memmap):,} raw tokens from {args.eval_data}...")
    print(f"Using non-overlapping windows of length {config.context_length}")

    metrics = compute_metrics_non_overlapping(
        model=model,
        memmap_data=eval_memmap,
        context_length=config.context_length,
        batch_size=args.batch_size,
        device=device,
        use_autocast=use_autocast,
        autocast_dtype=autocast_dtype,
        chars_per_token=args.chars_per_token,
    )

    print("\n" + "=" * 45)
    print("           EVALUATION RESULTS          ")
    print("=" * 45)
    print(f"Evaluated Windows       : {metrics['total_windows']:,}")
    print(f"Total Evaluated Tokens  : {metrics['total_tokens']:,}")
    print(f"Validation Loss (NLL)   : {metrics['loss']:.4f}")
    print(f"Perplexity (PPL)        : {metrics['ppl']:.4f}")
    print(f"Bits-Per-Character (BPC): {metrics['bpc']:.4f}")
    print("=" * 45)

    if metrics["bpc"] > 1.0:
        print("\nWARNING: BPC is above 1.0. Verify convergence or check your tokenizer settings.")


if __name__ == "__main__":
    main()