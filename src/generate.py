import torch
import torch.nn.functional as F


def apply_top_k(logits, top_k):
    if top_k >= logits.size(-1):
        return logits

    top_k_val, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
    min_val = top_k_val[:, -1:]

    return logits.masked_fill(logits < min_val, -float("Inf"))


def apply_top_p(logits, top_p):
    if top_p >= 1.0 or top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    return logits.masked_fill(indices_to_remove, -float("Inf"))


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=1.0,
    top_k=None,
    top_p=None,
    seed=None,
    use_cache=True,
):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        return prompt

    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = list(input_ids)

    if max_new_tokens == 0:
        return tokenizer.decode(generated_ids)

    if use_cache:
        model.reset_cache(batch_size=1, device=device)
        logits = model(tokens, use_cache=True)
    else:
        logits = model(tokens, use_cache=False)

    next_token_logits = logits[:, -1, :]

    for step in range(max_new_tokens):
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature

            if top_k is not None:
                scaled_logits = apply_top_k(scaled_logits, top_k)

            if top_p is not None:
                scaled_logits = apply_top_p(scaled_logits, top_p)

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        if step < max_new_tokens - 1:
            if use_cache:
                logits = model(next_token, use_cache=True)
            else:
                tokens_in = torch.tensor(
                    [generated_ids], dtype=torch.long, device=device
                )
                logits = model(tokens_in, use_cache=False)

            next_token_logits = logits[:, -1, :]

    return tokenizer.decode(generated_ids)