import asyncio
from typing import Any, Dict, List

import torch
from megatron.core import mpu
from typing_extensions import override

from gpatch_v4.actor.distill_teacher_actor import DistillTeacherActor
from gpatch_v4.core.parallel_state import (
    get_model_and_context_parallel_group_gloo,
    is_tp_and_cp_head,
)
from gpatch_v4.transfer.transfer_queue_interface import TQInterface, TqPayloadType
from gpatch_v4.utils import clear_memory
from gpatch_v4.utils.training_utils import expand_rollout_batch


class EmbeddingDistillTeacherActor(DistillTeacherActor):
    """Independent-topo teacher for embedding OffPD.

    Reuses DistillTeacherActor PG / TQ / sleep-wake and the hidden-state RPC
    (``issue_calc_hidden_states`` / ``get_calc_hidden_states_result``). Forward
    is packed-THD embedding, published as ``teacher_eos_embeddings``.
    """

    @override
    async def setup_model(self):
        await super().setup_model()
        device = torch.cuda.current_device()
        for model_chunk in self.teacher_engine.model:
            for _, param in model_chunk.named_parameters():
                if param.device.type == "cpu":
                    param.data = param.data.to(device)

    @override
    async def _compute_hidden_states(
        self,
        batched_data: List[tuple[Dict[str, Any], asyncio.Task]],
    ) -> None:
        batched_data = sorted(
            batched_data,
            key=lambda item: (
                int(item[0]["actor_dp_rank"]),
                int(item[0]["sample_idx"]),
                int(item[0]["ppo_step"]),
            ),
        )
        request_meta = [metadata for metadata, _ in batched_data]
        payload_batches = list(
            await asyncio.gather(*[restore_task for _, restore_task in batched_data])
        )

        publish = mpu.is_pipeline_last_stage() and is_tp_and_cp_head()
        for metadata, payload_batch in zip(request_meta, payload_batches, strict=True):
            samples = expand_rollout_batch(payload_batch)
            outputs = self.teacher_engine._embedding_forward_only(samples, len(samples), 0)
            if publish:
                embeddings = [
                    output["embedding"].detach().contiguous().cpu() for output in outputs
                ]
                embedding_payload = await TQInterface.async_offload_to_tq(
                    {"teacher_eos_embeddings": embeddings},
                    int(metadata["ppo_step"]),
                    TqPayloadType.ONE_KEY_PER_DICT,
                    fields=["teacher_eos_embeddings"],
                )
                actor_dp_rank = metadata["actor_dp_rank"]
                ppo_step = metadata["ppo_step"]
                sample_idx = metadata["sample_idx"]
                step_results = self.compute_logps_results.setdefault(actor_dp_rank, {}).setdefault(
                    ppo_step, {}
                )
                step_results[sample_idx] = {"teacher_eos_embeddings": embedding_payload}

        group = get_model_and_context_parallel_group_gloo()
        gathered = [None] * torch.distributed.get_world_size(group=group)
        torch.distributed.all_gather_object(
            gathered,
            self.compute_logps_results if publish else None,
            group=group,
        )
        filled = [result for result in gathered if result is not None]
        assert len(filled) == 1
        self.compute_logps_results = filled[0]
        self.batching_reqs = []
        clear_memory()
