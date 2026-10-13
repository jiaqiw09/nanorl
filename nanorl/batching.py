"""Right-padded response microbatches for a causal, single-rank actor."""
from __future__ import annotations


def collate_responses(samples, *, vocab_size: int, max_length: int, device):
    import torch

    lengths = [len(s.prompt_ids) + len(s.response_ids) for s in samples]
    if not lengths or max(lengths) > max_length:
        raise ValueError("Empty microbatch or sequence exceeds model context")
    # Token zero is only a filler; it is never a loss target. No extra vocab entry is needed.
    tokens = torch.zeros((len(samples), max(lengths)), dtype=torch.long, device=device)
    response_mask = torch.zeros((len(samples), max(lengths) - 1), dtype=torch.bool, device=device)
    for row, (sample, length) in enumerate(zip(samples, lengths)):
        ids = sample.prompt_ids + sample.response_ids
        if not sample.prompt_ids or not sample.response_ids or any(type(t) is not int or not 0 <= t < vocab_size for t in ids):
            raise ValueError("Invalid prompt/response tokens in microbatch")
        tokens[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        # Position P-1 predicts first response token P; stop before the padded suffix.
        response_mask[row, len(sample.prompt_ids) - 1 : length - 1] = True
    positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand_as(tokens)
    return tokens, positions, response_mask


def response_log_probs(logits, tokens, response_mask, vocab_size: int):
    import torch

    if logits.ndim != 3 or logits.shape[:2] != tokens.shape or not 0 < vocab_size <= logits.shape[-1]:
        raise ValueError("Expected batch-first logits aligned with padded tokens")
    # Select BEFORE softmax: prompt and padding logits have no loss or gradient.
    selected = logits[:, :-1, :vocab_size][response_mask].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite response logits")
    targets = tokens[:, 1:][response_mask]
    return selected.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
