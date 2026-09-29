import torch
import torch.nn.functional as F


def compute_dpo_loss_core(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
):
    """Backend-agnostic DPO loss from pre-computed per-sequence log-probs.

    Parameters
    ----------
    policy_chosen_logps, policy_rejected_logps : Tensor ``[rbs]``
    ref_chosen_logps, ref_rejected_logps : Tensor ``[rbs]``
    beta : float
    label_smoothing : float
    loss_type : ``"sigmoid"``

    Returns
    -------
    losses : Tensor ``[rbs]``
    chosen_rewards : Tensor ``[rbs]``  (detached)
    rejected_rewards : Tensor ``[rbs]``  (detached)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        raise ValueError(f"unknown DPO loss type {loss_type}")

    return losses, chosen_rewards.detach(), rejected_rewards.detach()
