import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.parallel_state import cpu_barrier, is_mp_and_cp_head
from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator
from gpatch_v4.teacher_routing import plan_balanced_teacher_endpoints
from gpatch_v4.utils import (
    TimerSingleton,
    clear_memory,
    destroy_process_groups,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
    reload_process_groups,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


class OnPolicyDistillRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for on-policy distillation with multi-teacher support.

    Legacy fan-out computes every teacher. Routed mode computes only the
    selected teacher while preserving ``teacher_logprobs_{name}`` compatibility.

    Parameters
    ----------
    config : OnPolicyDistillConfig
    sampler_client : object
    gen_rm_client : object
    bt_rm_client : object
    run_eval : bool, optional
    teacher_clients : dict[str, object]
    """
    def __init__(
        self,
        config: OnPolicyDistillConfig,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        teacher_clients=None,
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)
        assert teacher_clients and len(teacher_clients) > 0, "At least one teacher is required"
        self.teacher_clients: Dict[str, Any] = teacher_clients

    async def calc_all_teacher_logps(
        self, rbs, num_microbatches, curr_ppo_step, sample_idx_base: int = None
    ):
        """Compute log probs from all teachers in parallel.

        Fans out requests to every teacher concurrently per phase, with a
        single ``cpu_barrier`` between phases for deterministic ordering.

        ``sample_idx_base`` overrides ``self.sample_idx`` when log_prob_top_k>0.
        当log_prob_top_k>0时， compute_teacher_logps需要student logits topk-ids,
        所以需要在 __call__ 之外，compute_student_logps之后再调用这个函数，但这时，
        self.sample_idx 已经被加过了，不能直接用，所以需要传一个 sample_idx_base 来覆盖它.
        """
        teacher_items = list(self.teacher_clients.items())
        routed_mode = self.config.ppo.g_opd_compute_only_routed_teacher_logps
        routed_requests = []
        active_teacher_names = set(self.teacher_clients)
        if routed_mode:
            assert self.config.ppo.log_prob_top_k == 0, (
                "routed teacher log-probs currently require log_prob_top_k=0"
            )
            assert num_microbatches == len(rbs), (
                f"num_microbatches={num_microbatches} != len(rbs)={len(rbs)}"
            )
            if self.is_mp_and_cp_head:
                s_idx = self.sample_idx if sample_idx_base is None else sample_idx_base
                routed_requests, active_teacher_names = self._build_routed_teacher_requests(
                    rbs, curr_ppo_step, s_idx
                )

        # Only global torch rank 0 actually sends begin/end RPCs. On routed
        # head ranks this also avoids onloading a teacher with zero samples in
        # the current PPO step.
        active_teacher_items = [
            item for item in teacher_items if item[0] in active_teacher_names
        ]

        # Phase 1: wake up all teachers concurrently
        await asyncio.gather(
            *[
                t_client.mark_ppo_step_begin(0, curr_ppo_step)
                for _, t_client in active_teacher_items
            ]
        )
        cpu_barrier()
        logging_memory_usage_details(
            f"memory tracking actor after teacher wake_up",
            rank=0,
        )

        # Phase 2: issue calc_logps requests for all teachers concurrently
        # Only send keys that teacher actually needs to reduce RPC transfer size.
        if self.is_mp_and_cp_head and routed_mode:
            await asyncio.gather(
                *[
                    self.teacher_clients[request["teacher_name"]].issue_calc_logps(
                        request["batched_data"],
                        curr_ppo_step,
                        request["sample_idx"],
                        endpoint_idx=request["endpoint_idx"],
                    )
                    for request in routed_requests
                ]
            )
        elif self.is_mp_and_cp_head:
            s_idx = self.sample_idx if sample_idx_base is None else sample_idx_base
            # prompt_lengths is required by dyn-CP reroute
            # (compute_dyn_cp_response_span); non-dyn-CP teacher fwd ignores it.
            teacher_keys = {
                "tokens",
                "prompt_lengths",
                "sequence_lengths",
                "stu_topk_ids",
                "position_ids",
                "image_input_mask",
                "vision_data",
                "vision_grid_thw",
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
                "audio_feature",
            }
            all_issue_cos = []
            for _, t_client in teacher_items:
                for rbi, rollout_batch in enumerate(rbs):
                    lightweight_batch = self._get_lightweight_teacher_batch(rollout_batch)
                    all_issue_cos.append(
                        t_client.issue_calc_logps(
                            lightweight_batch,
                            curr_ppo_step,
                            s_idx + rbi,
                        )
                    )
            await asyncio.gather(*all_issue_cos)
        cpu_barrier()

        # Phase 3: collect results for all teachers concurrently, then
        # update rollout_batches in deterministic order
        if self.is_mp_and_cp_head and routed_mode:
            all_results = await asyncio.gather(
                *[
                    self.teacher_clients[request["teacher_name"]].get_calc_logps_result(
                        curr_ppo_step,
                        request["sample_idx"],
                        endpoint_idx=request["endpoint_idx"],
                    )
                    for request in routed_requests
                ]
            )
            for request, response in zip(routed_requests, all_results, strict=True):
                if request["is_padding"]:
                    continue
                rollout_batch = rbs[request["rollout_batch_idx"]]
                self.fill_routed_teacher_logps(
                    rollout_batch,
                    response,
                    request["teacher_name"],
                )
            assert check_rollout_batches(rbs), "rbs format error after routed teachers"
        elif self.is_mp_and_cp_head:
            s_idx = self.sample_idx if sample_idx_base is None else sample_idx_base
            all_result_cos = []
            for _, t_client in teacher_items:
                for rbi in range(num_microbatches):
                    all_result_cos.append(
                        t_client.get_calc_logps_result(curr_ppo_step, s_idx + rbi)
                    )
            all_results = await asyncio.gather(*all_result_cos)

            idx = 0
            for t_name, _ in teacher_items:
                for rbi, rollout_batch in enumerate(rbs):
                    resp = all_results[idx]
                    self.fillback_teacher_logps(rollout_batch, resp, t_name)
                    idx += 1
                assert check_rollout_batches(rbs), \
                    f"rbs format error after teacher {t_name}"
        cpu_barrier()

        # Phase 4: put all teachers to sleep concurrently
        await asyncio.gather(
            *[
                t_client.mark_ppo_step_end(0, curr_ppo_step)
                for _, t_client in active_teacher_items
            ]
        )
        cpu_barrier()

        logging_memory_usage_details("memory tracking actor after teacher sleep", rank=0)

        return rbs

    def _get_routed_teacher_name(self, rollout_batch: Dict[str, List[Any]]) -> str:
        """Return the single teacher selected by a homogeneous microbatch."""

        if len(self.teacher_clients) == 1:
            return next(iter(self.teacher_clients))

        routing_field = self.config.ppo.g_opd_teacher_routing_field
        assert routing_field in rollout_batch, (
            f"routed teacher log-probs require '{routing_field}' in rollout batch; "
            f"got keys={list(rollout_batch)}"
        )
        route_values = rollout_batch[routing_field]
        if not isinstance(route_values, (list, tuple)):
            route_values = [route_values]
        teacher_names = {str(value) for value in route_values}
        assert len(teacher_names) == 1, (
            "routed teacher log-probs currently require one teacher per rollout "
            f"microbatch, got {routing_field}={route_values}. Keep rollout_mbs=1 "
            "or add per-sample sub-batch routing."
        )
        teacher_name = next(iter(teacher_names))
        assert teacher_name in self.teacher_clients, (
            f"sample routed to unknown teacher {teacher_name!r}; configured "
            f"teachers={list(self.teacher_clients)}"
        )
        return teacher_name

    def _get_lightweight_teacher_batch(
        self, rollout_batch: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """Strip rollout-only fields before sending a teacher RPC."""

        teacher_keys = {
            "tokens",
            "prompt_lengths",
            "sequence_lengths",
            "stu_topk_ids",
            "position_ids",
            "image_input_mask",
            "vision_data",
            "vision_grid_thw",
            "input_features",
            "feature_attention_mask",
            "audio_feature",
        }
        teacher_batch = self.get_teacher_rollout_batch(rollout_batch)
        lightweight_batch = {
            key: teacher_batch[key] for key in teacher_keys if key in teacher_batch
        }
        if "stu_topk_ids" in lightweight_batch:
            seq_lens = lightweight_batch["sequence_lengths"]
            lightweight_batch["stu_topk_ids"] = [
                ids[:int(seq_len.item()) - 1]
                for ids, seq_len in zip(
                    lightweight_batch["stu_topk_ids"], seq_lens, strict=True
                )
            ]
        return lightweight_batch

    def _gather_routed_teacher_counts(self, local_counts: List[int]) -> List[List[int]]:
        """Gather per-teacher request counts across policy DP head ranks."""

        dp_group = mpu.get_data_parallel_group()
        group_size = torch.distributed.get_world_size(group=dp_group)
        count_tensor = torch.tensor(
            local_counts,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        gathered = [torch.empty_like(count_tensor) for _ in range(group_size)]
        torch.distributed.all_gather(gathered, count_tensor, group=dp_group)
        return [row.cpu().tolist() for row in gathered]

    def _build_routed_teacher_requests(self, rbs, curr_ppo_step: int, s_idx: int):
        """Build real and padding RPC requests for routed teacher computation.

        Requests are globally striped over every endpoint of their selected
        teacher. If a teacher's sample count is not divisible by its endpoint
        count, its first active policy DP rank duplicates one real request into
        the remaining endpoint slots. Padding results are fetched to trigger
        every endpoint's forward pass, then discarded.
        """

        teacher_names = list(self.teacher_clients)
        local_by_teacher = defaultdict(list)
        for rollout_batch_idx, rollout_batch in enumerate(rbs):
            teacher_name = self._get_routed_teacher_name(rollout_batch)
            local_by_teacher[teacher_name].append(
                {
                    "teacher_name": teacher_name,
                    "rollout_batch_idx": rollout_batch_idx,
                    "sample_idx": s_idx + rollout_batch_idx,
                    "batched_data": self._get_lightweight_teacher_batch(rollout_batch),
                    "endpoint_idx": None,
                    "is_padding": False,
                }
            )

        gathered_rows = self._gather_routed_teacher_counts(
            [len(local_by_teacher[name]) for name in teacher_names]
        )
        group_rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        requests = []
        active_teacher_names = set()

        for teacher_index, teacher_name in enumerate(teacher_names):
            teacher_client = self.teacher_clients[teacher_name]
            teacher_config = self.config.teachers[teacher_name]
            if isinstance(teacher_config, dict):
                forward_only_mbs = int(teacher_config.get("forward_only_mbs", 1))
            else:
                forward_only_mbs = int(getattr(teacher_config, "forward_only_mbs", 1))
            all_rank_counts = [row[teacher_index] for row in gathered_rows]
            endpoint_plan = plan_balanced_teacher_endpoints(
                all_rank_counts,
                group_rank=group_rank,
                num_endpoints=teacher_client.svr_cluster,
                requests_per_endpoint_multiple=forward_only_mbs,
            )
            if endpoint_plan.global_request_count == 0:
                continue
            active_teacher_names.add(teacher_name)

            real_requests = local_by_teacher[teacher_name]
            assert len(real_requests) == len(endpoint_plan.real_endpoint_indices)
            for request, endpoint_idx in zip(
                real_requests, endpoint_plan.real_endpoint_indices, strict=True
            ):
                request["endpoint_idx"] = endpoint_idx
                requests.append(request)

            if endpoint_plan.padding_endpoint_indices:
                assert real_requests, (
                    "padding owner must have at least one real request to duplicate"
                )
                template = real_requests[0]
                for padding_index, endpoint_idx in enumerate(
                    endpoint_plan.padding_endpoint_indices
                ):
                    requests.append(
                        {
                            "teacher_name": teacher_name,
                            "rollout_batch_idx": None,
                            # Results are scoped by actor_dp_rank and ppo_step;
                            # negative IDs cannot collide with real sample IDs.
                            "sample_idx": -(padding_index + 1),
                            "batched_data": template["batched_data"],
                            "endpoint_idx": endpoint_idx,
                            "is_padding": True,
                        }
                    )

            padding_request_count = (
                endpoint_plan.requests_per_endpoint * teacher_client.svr_cluster
                - endpoint_plan.global_request_count
            )
            log(
                f"routed teacher {teacher_name}: global_real_requests="
                f"{endpoint_plan.global_request_count}, endpoints="
                f"{teacher_client.svr_cluster}, requests_per_endpoint="
                f"{endpoint_plan.requests_per_endpoint}, padding_requests="
                f"{padding_request_count}, ppo_step={curr_ppo_step}",
                rank=0,
            )

        return requests, active_teacher_names

    def fill_routed_teacher_logps(
        self,
        rollout_batch: Dict[str, List[Any]],
        response: Dict[str, List[Any]],
        teacher_name: str,
    ) -> Dict[str, List[Any]]:
        """Fill the routed result while preserving the target branch schema.

        Only ``teacher_name`` was computed. Zero-valued entries for the other
        configured teachers make rollout schemas identical across DP ranks;
        routing ensures those placeholders are never selected by the loss.
        """
        self.fillback_teacher_logps(rollout_batch, response, teacher_name)
        selected = rollout_batch[f"teacher_logprobs_{teacher_name}"]
        for configured_name in self.teacher_clients:
            key = f"teacher_logprobs_{configured_name}"
            if key not in rollout_batch:
                rollout_batch[key] = [
                    torch.zeros_like(logprobs) for logprobs in selected
                ]
        return rollout_batch

    @override
    def _post_process_rm_rollout_batch(self, rollout_batches: List[Dict[str, List[Any]]]):
        return self.apply_sampling_rollout_attr.add_back_rollout_attr(rollout_batches)

    def get_teacher_rollout_batch(self, rollout_batch: Dict[str,
                                                            List[Any]]) -> Dict[str, List[Any]]:
        if "teacher_tokens" in rollout_batch:
            teacher_rollout_batch = {}
            teacher_rollout_batch["tokens"] = rollout_batch["teacher_tokens"]
            teacher_rollout_batch["prompt_lengths"] = rollout_batch["teacher_prompt_lengths"]
            teacher_rollout_batch["sequence_lengths"] = rollout_batch["teacher_sequence_lengths"]
            if "teacher_res" in rollout_batch:
                for rbs in rollout_batch["teacher_res"]:
                    for k, v in rbs.items():
                        if k not in teacher_rollout_batch:
                            teacher_rollout_batch[k] = []
                        teacher_rollout_batch[k].append(v)
            if self.config.ppo.log_prob_top_k > 0:
                teacher_rollout_batch["stu_topk_ids"] = []
                for i in range(len(rollout_batch["stu_topk_ids"])):
                    tea_prompt_len = teacher_rollout_batch["prompt_lengths"][i]
                    stu_prompt_len = rollout_batch["prompt_lengths"][i]
                    tea_seq_len = teacher_rollout_batch["sequence_lengths"][i]
                    stu_seq_len = rollout_batch["sequence_lengths"][i]
                    tea_response_len = tea_seq_len - tea_prompt_len
                    stu_response_len = stu_seq_len - stu_prompt_len
                    assert tea_response_len == stu_response_len
                    input = rollout_batch["stu_topk_ids"][i]
                    stu_topk_ids = torch.zeros(
                        (tea_seq_len - 1, input.shape[1]),
                        dtype=input.dtype,
                        layout=input.layout,
                        device=input.device
                    )
                    stu_topk_ids[tea_prompt_len - 1:tea_seq_len - 1] = input[stu_prompt_len -
                                                                             1:stu_seq_len - 1]
                    teacher_rollout_batch["stu_topk_ids"].append(stu_topk_ids)
        else:
            teacher_rollout_batch = rollout_batch

        return teacher_rollout_batch

    def fillback_teacher_logps(
        self,
        rollout_batch: Dict[str, List[Any]],
        response: Dict[str, List[Any]],
        t_name: str,
        use_canonical_keys: bool = False,
    ) -> Dict[str, List[Any]]:
        def output_key(base_key: str) -> str:
            return base_key if use_canonical_keys else f"{base_key}_{t_name}"

        teacher_logprobs_key = output_key("teacher_logprobs")
        teacher_on_stu_topk_logprobs_key = output_key("teacher_on_stu_topk_logprobs")
        teacher_topk_ids_key = output_key("teacher_topk_ids")
        teacher_topk_logprobs_key = output_key("teacher_topk_logprobs")

        if "teacher_tokens" in rollout_batch:
            rollout_batch[teacher_logprobs_key] = []
            if "teacher_on_stu_topk_logprobs" in response:
                rollout_batch[teacher_on_stu_topk_logprobs_key] = []
            if "teacher_topk_ids" in response:
                rollout_batch[teacher_topk_ids_key] = []
            if "teacher_topk_logprobs" in response:
                rollout_batch[teacher_topk_logprobs_key] = []

            for i in range(len(rollout_batch["teacher_tokens"])):
                tea_prompt_len = rollout_batch["teacher_prompt_lengths"][i]
                stu_prompt_len = rollout_batch["prompt_lengths"][i]
                tea_seq_len = rollout_batch["teacher_sequence_lengths"][i]
                stu_seq_len = rollout_batch["sequence_lengths"][i]
                input = response["teacher_logprobs"][i]
                assert input.shape[0] == tea_seq_len - 1
                assert tea_seq_len - tea_prompt_len == stu_seq_len - stu_prompt_len
                logprobs = torch.zeros(
                    (stu_seq_len - 1), dtype=input.dtype, layout=input.layout, device=input.device
                )
                logprobs[stu_prompt_len - 1:] = input[tea_prompt_len - 1:]
                rollout_batch[teacher_logprobs_key].append(logprobs)
                if "teacher_on_stu_topk_logprobs" in response:
                    input = response["teacher_on_stu_topk_logprobs"][i]
                    assert input.shape[0] == tea_seq_len - 1
                    logprobs = torch.zeros(
                        (stu_seq_len - 1, input.shape[1]),
                        dtype=input.dtype,
                        layout=input.layout,
                        device=input.device
                    )
                    logprobs[stu_prompt_len - 1:, :] = input[tea_prompt_len - 1:, :]
                    rollout_batch[teacher_on_stu_topk_logprobs_key].append(logprobs)
                if "teacher_topk_ids" in response:
                    input = response["teacher_topk_ids"][i]
                    assert input.shape[0] == tea_seq_len - 1
                    ids = torch.full(
                        (stu_seq_len - 1, input.shape[1]),
                        -1,
                        dtype=input.dtype,
                        layout=input.layout,
                        device=input.device
                    )
                    ids[stu_prompt_len - 1:, :] = input[tea_prompt_len - 1:, :]
                    rollout_batch[teacher_topk_ids_key].append(ids)
                if "teacher_topk_logprobs" in response:
                    input = response["teacher_topk_logprobs"][i]
                    assert input.shape[0] == tea_seq_len - 1
                    logprobs = torch.zeros(
                        (stu_seq_len - 1, input.shape[1]),
                        dtype=input.dtype,
                        layout=input.layout,
                        device=input.device
                    )
                    logprobs[stu_prompt_len - 1:, :] = input[tea_prompt_len - 1:, :]
                    rollout_batch[teacher_topk_logprobs_key].append(logprobs)
        else:
            rollout_batch[teacher_logprobs_key] = response["teacher_logprobs"]
            if "teacher_on_stu_topk_logprobs" in response:
                rollout_batch[teacher_on_stu_topk_logprobs_key] = (
                    response["teacher_on_stu_topk_logprobs"]
                )
            if "teacher_topk_ids" in response:
                rollout_batch[teacher_topk_ids_key] = response["teacher_topk_ids"]
            if "teacher_topk_logprobs" in response:
                rollout_batch[teacher_topk_logprobs_key] = (
                    response["teacher_topk_logprobs"]
                )

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        timers = TimerSingleton.get_timer()
        dp_rank = mpu.get_data_parallel_rank()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        timers("sampler_generate", log_level=0).start(barrier=True)
        rbs = await self.rollout_samples(
            data_iter, num_microbatches, curr_ppo_step, dp_rank=dp_rank
        )
        if self.is_mp_and_cp_head:
            rbs = self._hook_after_sampling(rbs, curr_ppo_step)
            assert check_rollout_batches(rbs), f"rbs format error, may need pop('ready'): {rbs=}"
        cpu_barrier()
        timers("sampler_generate").stop()

        timers("gen_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_gen_rm_reward:
            rbs = await self.generate_gen_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("gen_rm_generate").stop()

        timers("bt_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_bt_rm_reward:
            rbs = await self.calc_bt_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("bt_rm_generate").stop()

        if self.is_mp_and_cp_head:
            rbs = self._post_process_rm_rollout_batch(rbs)

        # Teacher runs here when it does NOT depend on student's topk_ids.
        # only_tch: teacher produces its own topk (no dependency on student).
        # only_stu/intersection/union: teacher needs stu_topk_ids → deferred to actor.
        if self.config.ppo.log_prob_top_k == 0 or self.config.ppo.opd_top_k_strategy == "only_tch":
            timers("compute_teacher_logps", log_level=0).start(barrier=True)
            rbs = await self.calc_all_teacher_logps(rbs, num_microbatches, curr_ppo_step)
            cpu_barrier()
            timers("compute_teacher_logps").stop()

        if self.is_mp_and_cp_head:
            for rb in rbs:
                self.apply_sampling_rollout_attr.remove_rollout_attr(rb)
        self.sample_idx += num_microbatches

        if offload_process_group:
            reload_process_groups()
        return rbs
