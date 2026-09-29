import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


def gather_ep_partition(
    param: DTensor,
    local_tensor: torch.Tensor,
    ep_fsdp_mesh: DeviceMesh,
) -> torch.Tensor:
    """Restore one EP partition without gathering across EP ranks.

    Parameters
    ----------
    param : DTensor
        Its logical shape and stride describe this EP partition, not all experts.
    local_tensor : torch.Tensor
        FSDP-local shard, already unwrapped and on the collective's device.
    ep_fsdp_mesh : DeviceMesh
        One-dimensional FSDP mesh for this EP partition; all its ranks must call.

    Returns
    -------
    torch.Tensor
        Complete EP-local expert tensor on every rank in the FSDP mesh.
    """
    return DTensor.from_local(
        local_tensor.contiguous(),
        device_mesh=ep_fsdp_mesh,
        placements=[Shard(0)],
        shape=param.shape,
        stride=param.stride(),
    ).full_tensor()
