import torch
import torch.distributed as dist


class CpMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, cp_group):
        ctx.local_nel = tensor.shape[dim]
        nel = torch.tensor(tensor.shape[dim], device=tensor.device)
        dist.all_reduce(nel, group=cp_group)
        ctx.nel = int(nel.item())
        ctx.dim = dim
        out = tensor.sum(dim=dim)
        dist.all_reduce(out, group=cp_group)
        out = out / ctx.nel
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output / ctx.nel
        grad_output = grad_output.unsqueeze(
            ctx.dim
        ).expand(*grad_output.shape[:ctx.dim], ctx.local_nel,
                 *grad_output.shape[ctx.dim:]).contiguous()
        return grad_output, None, None


def load_balancing_loss_func(
    *,
    gate_logits: tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
    cp_group=None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """

    # [t, e], t=l*b*s
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    # [t, e]
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    # [t, top_k]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # [t, top_k, e]
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if cp_group is None or dist.get_world_size(cp_group) == 1:
        # Compute the percentage of tokens routed to each experts
        # [top_k, e]
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        # [e]
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # tokens_per_expert comes from topk/one_hot and carries no gradient,
        # so a plain (detached) CP all-reduce global mean is enough.
        # [top_k, e]
        t_local = expert_mask.shape[0]
        tokens_per_expert = expert_mask.float().sum(dim=0)
        dist.all_reduce(tokens_per_expert, group=cp_group)
        t_global = torch.tensor(t_local, device=tokens_per_expert.device)
        dist.all_reduce(t_global, group=cp_group)
        tokens_per_expert = tokens_per_expert / t_global.item()

        # router_prob_per_expert is a global average over CP tokens; route it
        # through the autograd-aware mean so gradients reach the local
        # router_logits with the correct 1/T scale.
        # [e]
        router_prob_per_expert = CpMean.apply(routing_weights, 0, cp_group)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
