import asyncio
from typing import Any, Dict, List

from typing_extensions import override

from gpatch_v4.rollout_generator.off_policy_distill_generator import (
    OffPolicyDistillRolloutGenerator,
)
from gpatch_v4.transfer.transfer_queue_interface import TQInterface
from gpatch_v4.utils import BroadcastUtils, TimerSingleton


class EmbeddingOffPolicyDistillRolloutGenerator(OffPolicyDistillRolloutGenerator):
    """Embedding OffPD: dataloader GBS + optional independent-topo teacher embeddings."""
    @override
    async def get_teacher_hidden_states(
        self,
        rbs: List[Dict[str, List[Any]]],
        curr_ppo_step,
    ):
        """Fetch Teacher TQ slots and restore embeddings into ``rbs``."""
        s_idx = self.sample_idx

        async def get_one(rollout_batch, sample_idx):
            result = await self.teacher_client.get_calc_hidden_states_result(
                curr_ppo_step, sample_idx
            )
            restored = await TQInterface.async_restore_from_tq(result["teacher_eos_embeddings"])
            embeddings = restored["teacher_eos_embeddings"]
            assert len(embeddings) == len(next(iter(rollout_batch.values())))
            rollout_batch["teacher_eos_embeddings"] = embeddings

        return await asyncio.gather(
            *[get_one(rollout_batch, s_idx + rbi) for rbi, rollout_batch in enumerate(rbs)]
        )

    @override
    async def __call__(
        self,
        data_iter,
        num_microbatches,
        curr_train_step,
        prefetch_num_gb=1,
    ):
        timers = TimerSingleton.get_timer()
        rbs_list = []
        timers("get_batched_data", log_level=0).start(barrier=True)
        for _ in range(prefetch_num_gb):
            rbs = self.gen_data_from_data_iter(data_iter, num_microbatches, curr_train_step)
            rbs_list.append(BroadcastUtils.broadcast_rollout_batch(rbs))
        timers("get_batched_data").stop()

        if (
            self.config.training.enable_teacher_kl_loss and
            self.config.training.setup_teacher_in_independent_topo
        ):
            timers("teacher_infer", log_level=0).start(barrier=True)
            flat_rbs = [rollout_batch for rbs in rbs_list for rollout_batch in rbs]
            await self.calc_teacher_hidden_states(flat_rbs, curr_train_step)
            timers("teacher_infer").stop()

        self.sample_idx += prefetch_num_gb * num_microbatches
        return rbs_list
