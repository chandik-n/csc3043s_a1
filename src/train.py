import argparse
from dataclasses import asdict
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import TransformerLM, TransformerConfig
from src.data import MemmapDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, lr, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def get_lr_scheduler(optimizer, warmup_steps, max_steps, base_lr):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_steps = max(1, max_steps - warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=0.1 * base_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )


def evaluate(model, val_loader, device, use_autocast, autocast_dtype, eval_batches=20):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= eval_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / count if count > 0 else 0.0


def get_epoch_indices(seed, epoch, dataset_len):
    """Deterministically generate dataset indices for any given epoch."""
    epoch_g = torch.Generator()
    epoch_g.manual_seed(seed + epoch)
    return torch.randperm(dataset_len, generator=epoch_g).tolist()


def fetch_batch(dataset, indices, batch_idx, batch_size):
    start = batch_idx * batch_size
    end = min(start + batch_size, len(indices))
    batch_indices = indices[start:end]
    return dataset[batch_indices]


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    #Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    #Architecture Flags
    parser.add_argument("--vocab_size", type=int, default=1500)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--use_rmsnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "relu"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        autocast_dtype = torch.bfloat16
        use_autocast = True
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        use_autocast = False

    print(f"Device: {device} | Autocast: {use_autocast} | RMSNorm: {args.use_rmsnorm} | RoPE: {args.use_rope}")

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        use_rmsnorm=args.use_rmsnorm,
        use_rope=args.use_rope,
        ffn_type=args.ffn_type,
    )
    model = TransformerLM(config).to(device)

    optimizer = get_optimizer(model, args.lr, args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_steps, base_lr=args.lr)

    train_dataset = MemmapDataset(args.train_data, context_length=args.context_length)
    val_dataset = MemmapDataset(args.val_data, context_length=args.context_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start_step = 0
    tokens_processed = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = ckpt["step"]
        tokens_processed = ckpt.get("tokens_processed", 0)

        torch.set_rng_state(ckpt["rng_torch"].cpu())
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_random"])

    total_samples = len(train_dataset)
    epoch_size_in_batches = max(1, total_samples // args.batch_size)

    start_time = time.time()
    log_path = os.path.join(args.out_dir, "train_log.jsonl")

    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        step_idx = step - 1
        current_epoch = step_idx // epoch_size_in_batches
        batch_idx_in_epoch = step_idx % epoch_size_in_batches

        epoch_indices = get_epoch_indices(args.seed, current_epoch, total_samples)
        x, y = fetch_batch(train_dataset, epoch_indices, batch_idx_in_epoch, args.batch_size)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()

        optimizer.step()
        scheduler.step()

        tokens_processed += args.batch_size * args.context_length
        current_lr = scheduler.get_last_lr()[0]

        val_loss_to_log = None
        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss_to_log = evaluate(
                model, val_loader, device, use_autocast, autocast_dtype, eval_batches=args.eval_batches
            )
            print(
                f"Step {step}/{args.max_steps} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss_to_log:.4f} | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        log_data = {
            "step": step,
            "current_epoch": current_epoch,
            "wall_clock": round(time.time() - start_time, 2),
            "tokens_processed": tokens_processed,
            "lr": current_lr,
            "train_loss": round(loss.item(), 4),
            "val_loss": round(val_loss_to_log, 4) if val_loss_to_log is not None else None,
            "grad_norm": round(grad_norm, 4),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "tokens_processed": tokens_processed,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_numpy": np.random.get_state(),
                    "rng_random": random.getstate(),
                    "config": asdict(config),
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()