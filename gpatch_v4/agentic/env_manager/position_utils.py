from typing import Optional

import torch

from gpatch_v4.core.constants import MODEL_ARCH

# Qwen3VLPrepareDataForward / WemmVideoPrepareDataForward assert
# position_ids >= smart-pad seqlen and refuse to 0-pad (0 is a valid
# mRoPE id). Qwen3.8-27B dense registers as ``qwen3_5``.
_MROPE_FULL_PAD_ARCHS = (
    MODEL_ARCH.QWEN3_5,
    MODEL_ARCH.QWEN3_5_MOE,
    MODEL_ARCH.QWEN3_VL,
    MODEL_ARCH.QWEN3_VL_MOE,
    MODEL_ARCH.QWEN3_OMNI_MOE,
    MODEL_ARCH.QWEN3_5_WEMM,
    MODEL_ARCH.QWEN3_5_MOE_WEMM,
    MODEL_ARCH.QWEN3_VL_WEMM,
)


def agentic_mrope_placeholders(
    token_len: int,
    model_arch: Optional[str],
    seq_length: Optional[int],
    pad_multi: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad mRoPE ids to the smart-pad bucket for VL / qwen3.5 arches.

    Parameters
    ----------
    token_len : int
        Unpadded token length of this sample.
    model_arch : str or None
    seq_length : int or None
        Required for mRoPE VL / qwen3.5 / wemm arches.
    pad_multi : int or None
        ``training.pad_to_mulitiple_of``. Required with ``seq_length``.

    Returns
    -------
    position_ids : Tensor
        Shape ``(3, 1, pos_len)``. mRoPE arches use
        ``max(seq_length, ceil(token_len / pad_multi) * pad_multi)``;
        others use ``token_len``.
    image_input_mask : Tensor
        Shape ``(1, pos_len)``, all False.
    """
    if model_arch in _MROPE_FULL_PAD_ARCHS:
        assert seq_length is not None and pad_multi is not None, (
            f"{model_arch} position_ids need seq_length and pad_to_mulitiple_of"
        )
        pad_to_len = ((token_len + pad_multi - 1) // pad_multi) * pad_multi
        pos_len = max(seq_length, pad_to_len)
    else:
        pos_len = token_len
    position_ids = (
        torch.arange(pos_len, dtype=torch.long).view(1, 1, pos_len).expand(3, 1,
                                                                           pos_len).contiguous()
    )
    image_input_mask = torch.zeros(1, pos_len, dtype=torch.bool)
    return position_ids, image_input_mask
