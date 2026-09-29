# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from typing import Any, Dict, List, Optional

import torch

from gpatch_v4.utils.dynamic_cp_utils import packed_to_response_padded


def unpack_thd_log_probs(
    packed_log_probs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> list[torch.Tensor]:
    """Restore per-sample logprobs from a ``[1, T]`` padded THD axis."""
    assert packed_log_probs.dim() == 2 and packed_log_probs.shape[0] == 1
    cu = cu_seqlens.detach().cpu().tolist()
    cu_padded = cu_seqlens_padded.detach().cpu().tolist()
    assert len(cu) == len(cu_padded) and len(cu) >= 2
    restored = []
    flat = packed_log_probs.squeeze(0)
    assert cu[0] == 0 and cu_padded[0] == 0
    assert cu_padded[-1] == flat.numel(
    ), (f"padded THD boundary {cu_padded[-1]} must match packed logprob "
        f"length {flat.numel()}")
    for sample_idx in range(len(cu) - 1):
        actual_len = cu[sample_idx + 1] - cu[sample_idx]
        start = cu_padded[sample_idx]
        padded_len = cu_padded[sample_idx + 1] - start
        assert 0 <= actual_len <= padded_len
        restored.append(flat[start:start + actual_len].contiguous())
    return restored


def _optional_prompt_len(sample: Dict[str, Any]) -> int | None:
    if "prompt_lengths" not in sample:
        return None
    value = sample["prompt_lengths"]
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def compute_thd_response_span(
    *,
    prompt_len: int | None,
    sequence_len: int,
    actual_len: int,
    response_mask: Optional[torch.Tensor] = None,
) -> tuple[int, int]:
    """Next-token response ``(start, length)`` on a shifted S-1 sequence."""
    if prompt_len is not None:
        response_start = max(prompt_len - 1, 0)
        response_len = max(min(sequence_len - prompt_len, actual_len - response_start), 0)
        return response_start, response_len

    if response_mask is None:
        return actual_len, 0
    nonzero_mask = torch.nonzero(response_mask.reshape(-1).bool(), as_tuple=False)
    response_start = int(nonzero_mask[0].item()) if nonzero_mask.numel() else actual_len
    response_len = max(actual_len - response_start, 0)
    return response_start, response_len


def thd_response_spans_from_samples(
    samples: List[Dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts: list[int] = []
    lengths: list[int] = []
    for sample in samples:
        tokens = sample["tokens"]
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.reshape(-1)
        sequence_len = int(torch.as_tensor(sample["sequence_lengths"]).item())
        response_start, response_len = compute_thd_response_span(
            prompt_len=_optional_prompt_len(sample),
            sequence_len=sequence_len,
            actual_len=tokens.numel() - 1,
            response_mask=sample.get("mask"),
        )
        starts.append(response_start)
        lengths.append(response_len)
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )


def response_pad_fsdp2_thd_loss_tensors(
    tensors: Dict[str, Optional[torch.Tensor]],
    cu_seqlens_padded: torch.Tensor,
    response_starts: torch.Tensor,
    response_lengths: torch.Tensor,
) -> Dict[str, Optional[torch.Tensor]]:
    """Slice full packed ``[1, T]`` policy tensors to response-padded ``[B, R_max]``.

    FSDP2 THD logprobs/entropy are already CP-gathered onto the global packed
    axis, so this path does not reconstruct CP shards.
    """
    padded = packed_to_response_padded(
        tensors,
        cu_seqlens_padded,
        response_starts,
        response_lengths,
    )
    mask = padded["mask"]
    entropy = padded["per_token_entropy"]
    assert mask is not None
    padded["scaled_entropy"] = (
        (entropy * mask).sum() / mask.sum().clamp(min=1) if entropy is not None else None
    )
    return padded
