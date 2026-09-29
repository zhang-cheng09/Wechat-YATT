"""Controller helpers that turn collected rollout batches into train refs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import ray

from gpatch_v4.core.dynamic_batch_reward_normalize import (
    prepare_sample_masks,
    reward_normalize,
    uses_group_reward_normalization,
)
from gpatch_v4.transfer.tq_connector import KVBatchMeta
from gpatch_v4.transfer.transfer_queue_interface import TQInterface, TqSampleOperateType
from gpatch_v4.utils import Envelope, log
from gpatch_v4.utils.dynamic_batch_utils import (
    TRAIN_STEP_ID_KEY,
    assign_train_steps,
    compute_rollout_metrics,
    convert_samples_to_train_data,
    split_train_steps_by_dp,
    validate_dynamic_batch_samples,
)
from gpatch_v4.utils.filter_samplings import filter_rollout_samples
from gpatch_v4.utils.training_utils import expand_rollout_batches

if TYPE_CHECKING:
    from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
        GenerateResult,
    )


def sample_object_indices(
    source: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
) -> List[int]:
    index_by_id = {id(sample): index for index, sample in enumerate(source)}
    assert len(index_by_id) == len(source), "source samples contain duplicate objects"
    indices = []
    for sample in selected:
        sample_id = id(sample)
        assert sample_id in index_by_id, (
            "dynamic-batch hooks must return existing sample objects to preserve TQ row alignment"
        )
        indices.append(index_by_id[sample_id])
    assert len(set(indices)) == len(indices), "selected samples contain duplicate objects"
    return indices


class PrepareResultMixin:
    """Split collected rollout batches into per-DP train refs.

    Expects the consuming controller to provide:
    - ``self.config`` / ``self.training_config``
    - ``self._dp_size``
    - ``self.custom_filter_samples`` / ``self.custom_reward_normalize`` /
      ``self.custom_convert_samples_to_train_data`` /
      ``self.custom_compute_rollout_metrics``
    - ``self._split_train_data_by_dp``
    """
    async def _prepare_variable_sample_count_train_data(
        self,
        rbs: List[Dict[str, List[Any]]],
        tq_metas: Optional[Sequence[KVBatchMeta]] = None,
        tq_original_state=None,
        tq_clear_refs=None,
    ) -> GenerateResult:
        """Prepare variable-sample-count train data for distributed training.

        Here, ``dynamic`` means that sample-level filtering and conversion may
        leave each train step with a different number and structure of samples.
        The controller must therefore plan train-step boundaries from the full
        rollout view and partition the actual post-processed samples, instead
        of deriving fixed batches from nominal GBS. This is independent of
        dynamic microbatch sizing and Dynamic Context Parallel; Dynamic CP is
        an optional downstream mechanism for routing variable-length sequences.

        The preparation pipeline performs the following operations:

        1. Expand each batched dictionary into per-sample dictionaries and
           assign complete prompts to train steps.
        2. Build sample masks, filter samples, normalize rewards, and compute
           rollout metrics.
        3. Convert the samples to the policy training format and validate the
           resulting fields and train-step grouping.
        4. Balance every train step by token count across DP ranks and hand off
           each rank through a Ray envelope or TQ key view.

        Parameters
        ----------
        rbs : List[Dict[str, List[Any]]]
            Collected rollout batches. Each element is a batched dictionary
            that maps a field name to an aligned list of field values. Position
            ``i`` across all fields represents one rollout sample. Expanding
            the outer list produces ``List[Dict[str, Any]]``.
        tq_metas : Sequence[KVBatchMeta], optional
            TQ rows aligned with the unprocessed expanded samples.
        tq_original_state
            Tensor snapshot captured before controller-side mutation.
        tq_clear_refs
            Complete pre-filter metadata batches retained for final cleanup.

        Returns
        -------
        GenerateResult
            Prepared training data and metrics. ``dp_refs`` has length
            ``self._dp_size``. Each element resolves to
            ``List[List[Dict[str, Any]]]`` through either the Ray envelope or
            TQ key-view path. The outer list is ordered by train step, and each
            inner list contains the converted samples assigned to that DP rank.
            ``metrics`` maps metric names to scalar values computed from the
            filtered samples.

        Raises
        ------
        AssertionError
            If the rollout or training configuration cannot form complete
            train steps, filtering or conversion removes an entire train step,
            converted samples violate dynamic-batch invariants, or a train
            step cannot be partitioned across DP ranks.
        """
        from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
            GenerateResult,
        )

        training_config = self.training_config
        samples = expand_rollout_batches(rbs)
        source_samples = list(samples)
        # expand flattens all microbatches; later UPDATE needs one aligned meta.
        flat_meta = (TQInterface.concat_kv_batch_metas(tq_metas) if tq_metas is not None else None)
        # 1. stamp _train_step_id by prompt_idx
        samples = assign_train_steps(samples, training_config)
        expected_train_step_ids = {sample[TRAIN_STEP_ID_KEY] for sample in samples}
        # 2. prepare train masks
        prepare_sample_masks(samples)
        # 3. filter samples
        samples = filter_rollout_samples(
            self.config,
            samples,
            custom_filter_fn=self.custom_filter_samples,
        )
        # 4. normalize rewards
        pre_advantage_metrics = reward_normalize(
            self.config,
            samples,
            custom_reward_normalize_fn=self.custom_reward_normalize,
        )
        requires_normalized_rewards = uses_group_reward_normalization(self.config)
        assert {sample[TRAIN_STEP_ID_KEY]
                for sample in samples
               } == expected_train_step_ids, ("filter removed an entire train step")
        # 5. compute rollout metrics
        metrics = compute_rollout_metrics(
            self.config,
            samples,
            training_config.metrics_report or [],
            custom_metrics_fn=self.custom_compute_rollout_metrics,
        )
        metrics.update(pre_advantage_metrics)
        # 6. convert samples to train data
        samples = convert_samples_to_train_data(
            self.config,
            samples,
            custom_convert_fn=self.custom_convert_samples_to_train_data,
        )
        validate_dynamic_batch_samples(
            samples,
            require_normalized_rewards=requires_normalized_rewards,
        )
        assert {
            sample[TRAIN_STEP_ID_KEY]
            for sample in samples
        } == expected_train_step_ids, ("custom train-data conversion removed an entire train step")
        updated_meta = None
        if self.config.tq.enable:
            assert flat_meta is not None
            assert tq_original_state is not None
            assert tq_clear_refs is not None
            # filter/convert may drop rows; UPDATE needs meta 1:1 with survivors.
            selected_meta = TQInterface.subset_kv_batch_meta(
                flat_meta,
                sample_object_indices(source_samples, samples),
            )
            updated_meta = await TQInterface.async_put_samples(
                samples,
                TqSampleOperateType.UPDATE,
                meta=selected_meta,
                original_state=tq_original_state,
            )
        # 7. re-group by _train_step_id and balance each step by DP rank
        dp_partitions = split_train_steps_by_dp(
            samples,
            self._dp_size,
            training_config.train_mbs,
            self.config.policy.dist_config.dynamic_context_parallel,
        )
        if self.config.tq.enable:
            assert updated_meta is not None
            # One KVBatchMeta per train step: subset updated rows for that DP shard.
            refs = [
                tuple(
                    TQInterface.subset_kv_batch_meta(
                        updated_meta,
                        sample_object_indices(samples, step_samples),
                    ) for step_samples in partition
                ) for partition in dp_partitions
            ]
        else:
            refs = [Envelope(ray.put(partition)) for partition in dp_partitions]
        num_train_steps = len(dp_partitions[0]) if dp_partitions else 0
        log(
            f"[RolloutController] split {len(samples)} samples in "
            f"{num_train_steps} train steps across {self._dp_size} DP ranks"
        )
        return GenerateResult(
            dp_refs=refs,
            metrics=metrics,
            tq_clear_refs=tq_clear_refs or [],
        )

    async def _prepare_fixed_batch_train_data(
        self,
        rbs: List[Dict[str, List[Any]]],
        tq_metas: Optional[Sequence[KVBatchMeta]] = None,
        tq_original_state=None,
        tq_clear_refs=None,
    ) -> GenerateResult:
        """Split fixed microbatches by DP; optionally write TQ updates first.

        When TQ is disabled, returns Ray envelopes from
        ``_split_train_data_by_dp``. When enabled, microbatch boundaries stay
        intact: each original field-major batch becomes one ``KVBatchMeta``.
        Round-robin assignment matches ``_split_train_data_by_dp``
        (microbatch ``i`` → DP ``i % dp_size``).

        Parameters
        ----------
        rbs : List[Dict[str, List[Any]]]
            Clean field-major batches; aligned with ``tq_metas`` when TQ is on.
        tq_metas : Sequence[KVBatchMeta], optional
            Original per-microbatch metadata in the same order as ``rbs``.
        tq_original_state
            Snapshot from ``capture_tq_sample_tensor_state`` before BT-RM
            and rollout-attr mutation.
        tq_clear_refs
            Original metadata batches whose exact keys are cleared after training.

        Returns
        -------
        GenerateResult
            With TQ, ``dp_refs[dp]`` is ``tuple[KVBatchMeta, ...]``; without TQ,
            ``dp_refs`` are Ray object refs from ``_split_train_data_by_dp``.
        """
        from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
            GenerateResult,
        )

        if not self.config.tq.enable:
            assert tq_metas is None
            return GenerateResult(dp_refs=self._split_train_data_by_dp(rbs))

        assert tq_original_state is not None
        assert tq_clear_refs is not None
        samples = expand_rollout_batches(rbs)
        # expand flattens all microbatches; UPDATE needs one aligned meta.
        flat_meta = TQInterface.concat_kv_batch_metas(tq_metas)
        updated_meta = await TQInterface.async_put_samples(
            samples,
            TqSampleOperateType.UPDATE,
            meta=flat_meta,
            original_state=tq_original_state,
        )
        batch_sizes = [len(meta) for meta in tq_metas]
        tq_batches: List[KVBatchMeta] = []
        offset = 0
        # Slice the flat updated meta back into original microbatch boundaries.
        for batch_size in batch_sizes:
            tq_batches.append(
                TQInterface.subset_kv_batch_meta(
                    updated_meta,
                    range(offset, offset + batch_size),
                )
            )
            offset += batch_size
        assert offset == len(samples)

        dp_size = self._dp_size
        partitions: List[List[KVBatchMeta]] = [[] for _ in range(dp_size)]
        for i, batch in enumerate(tq_batches):
            partitions[i % dp_size].append(batch)
        refs = [tuple(part) for part in partitions]

        log(f"[RolloutController] split {len(rbs)} TQ micro-batches across "
            f"{dp_size} DP ranks")
        return GenerateResult(
            dp_refs=refs,
            tq_clear_refs=tq_clear_refs,
        )
