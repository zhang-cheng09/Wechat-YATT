import asyncio
import time

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.embedding_actor import EmbeddingActor
from gpatch_v4.actor.mixin import ProfileMixin, TestActorMixin, TrainingPltMixin
from gpatch_v4.actor.off_policy_distill_student_actor import (
    OffloadStudentEngineManager,
    OnloadStudentEngineManager,
)
from gpatch_v4.client import TeacherClient
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank, is_mp_and_cp_head
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.transfer import get_tq_connector, init_tq_connector
from gpatch_v4.utils import (
    TimerSingleton,
    TrainReporterSingleton,
    is_same_tokenizer,
    log,
)
from gpatch_v4.utils.training_utils import expand_rollout_batches


class EmbeddingOffPolicyDistillStudentActor(EmbeddingActor, TestActorMixin, ProfileMixin):
    """Embedding off-policy distill: prefetch + teacher embeddings.

    Same-topo teacher is in-process and time-shares GPUs via offload.
    Independent topo keeps the frozen teacher resident on its own actors
    so the next window's teacher forward overlaps student train.
    """

    train_log_tag = "EMB_OFFPD"

    async def init(self, config):
        await super().init(config)
        self.teacher_engine = None
        self.teacher_client = None
        if self.config.tq.enable:
            init_tq_connector(self.config.tq)
        if (
            self.config.training.enable_teacher_kl_loss and
            not self.config.training.setup_teacher_in_independent_topo
        ):
            assert self.config.training.prefetch_num_gb >= 1, (
                "prefetch_num_gb must be >= 1; colocate embedding offpd amortizes "
                "student/teacher swap over this many GBS"
            )
            extra_args = {
                "policy_config": config.teacher,
                "tokenizer": self.teacher_tokenizer,
            }
            self.teacher_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)

    @override
    def validated_config(self):
        super().validated_config()
        self.config.teacher.without_ref = True
        self.config.teacher.without_optim = True
        self.config.policy.without_ref = True
        if self.config.training.enable_teacher_kl_loss:
            is_same_tokenizer(
                self.actor_tokenizer, self.teacher_tokenizer, skip_eos_token_judge=True
            )

    @override
    async def setup_model_and_optimizer(self):
        await super().setup_model_and_optimizer()
        # Embedding distill uses teacher embeddings, not vocab KL weights.

    async def setup_client(self):
        self.gen_rm_client = None
        self.bt_rm_client = None
        self.sampler_client = None
        self.teacher_client = None
        if not self.config.training.enable_teacher_kl_loss:
            return
        if self.config.training.setup_teacher_in_independent_topo:
            self.teacher_client = TeacherClient(
                self.config,
                teacher_name=None,
                teacher_config=self.config.teacher,
            )
            return
        self.teacher_engine.setup_model_and_get_optimizer()
        # wrap_with_ddp=False: PEFT LoRA stays on CPU after kaiming init.
        device = torch.cuda.current_device()
        for model_chunk in self.teacher_engine.model:
            for _, param in model_chunk.named_parameters():
                if param.device.type == "cpu":
                    param.data = param.data.to(device)
        self.teacher_engine.offload_model()

    async def setup_rollout_generator(self):
        self.train_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
            teacher_client=self.teacher_client,
        )

    @override
    def _training_plt_report(self, train_state: TrainingPltMixin.TrainState, data: dict):
        self.training_plt_report(
            "EmbeddingOffPolicyDistillStudentActor",
            self.config.policy.model_arch,
            TrainingPltMixin.TrainType.SFT,
            train_state,
            data,
        )

    def _prefetch_num_gb(self, cur_epoch_train_step, train_step, training_config):
        prefetch_n = max(1, int(training_config.prefetch_num_gb or 1))
        remaining = training_config.train_step_per_epoch - cur_epoch_train_step
        exit_step = training_config.exit_step
        if exit_step is not None and exit_step > 0:
            if train_step >= exit_step:
                return 0
            remaining = min(remaining, exit_step - train_step)
        return max(0, min(prefetch_n, remaining))

    def _prepare_teacher_batches(self, prefetch_rbs_list):
        self.teacher_engine.onload_model()
        try:
            self.teacher_engine.set_model_eval()
            for expanded_rbs in prefetch_rbs_list:
                self.attach_teacher_embeddings(expanded_rbs)
        finally:
            self.teacher_engine.offload_model()
        return prefetch_rbs_list

    def _log_teacher_embeddings_once(self, prefetch_rbs_list):
        if (
            not self.config.training.enable_teacher_kl_loss or not is_last_rank() or
            hasattr(self, "_teacher_embeddings_logged")
        ):
            return
        first = prefetch_rbs_list[0][0].get("teacher_eos_embeddings")
        via = ("TQ" if self.config.training.setup_teacher_in_independent_topo else "colocate")
        log(
            f"[EmbeddingOffPD] teacher embeddings attached via {via} "
            f"shape={tuple(first.shape) if first is not None else None} "
            f"prefetch_num_gb={self.config.training.prefetch_num_gb}"
        )
        self._teacher_embeddings_logged = True

    @torch.no_grad()
    def attach_teacher_embeddings(self, expanded_rbs):
        """Teacher packed-THD forward; teacher must already be onloaded."""
        num_microbatches = len(expanded_rbs)
        outputs = self.teacher_engine._embedding_forward_only(expanded_rbs, num_microbatches, 0)
        if not mpu.is_pipeline_last_stage():
            return
        for sample, output in zip(expanded_rbs, outputs):
            sample["teacher_eos_embeddings"] = output["embedding"].detach()

    @override
    async def _train_loop(self):
        """Prefetch N GBS, one swap, then reuse EmbeddingActor.train_one_step."""
        self.setup_profile()
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch
        collected_metrics = []

        cpu_barrier()
        for epoch in range(init_epoch, training_config.num_train_epoches):
            self.maybe_set_epoch(epoch)

            self.train_iter = iter(self.train_dataloader)
            cur_epoch_train_step = init_step if epoch == init_epoch else 0
            epoch_exit = False
            while cur_epoch_train_step < training_config.train_step_per_epoch:
                await asyncio.sleep(0.01)
                self.last_progress_time = time.time()

                prefetch_num_gb = self._prefetch_num_gb(
                    cur_epoch_train_step, train_step, training_config
                )
                if prefetch_num_gb <= 0:
                    break

                prefetch_train_step = train_step
                with OffloadStudentEngineManager(self.model_engine):
                    prefetch_rbs_list = await self.train_rollout_generator(
                        self.train_iter,
                        num_microbatches,
                        prefetch_train_step,
                        prefetch_num_gb,
                    )
                    prefetch_rbs_list = [expand_rollout_batches(rbs) for rbs in prefetch_rbs_list]
                    if (
                        self.config.training.enable_teacher_kl_loss and
                        not self.config.training.setup_teacher_in_independent_topo and
                        self.teacher_engine is not None
                    ):
                        timers("teacher_infer", log_level=0).start(barrier=True)
                        prefetch_rbs_list = self._prepare_teacher_batches(prefetch_rbs_list)
                        timers("teacher_infer").stop()
                    self._log_teacher_embeddings_once(prefetch_rbs_list)
                prefetch_data_time = timers("get_batched_data",
                                            log_level=0).elapsed(reset=True) / prefetch_num_gb
                teacher_infer_time = timers("teacher_infer",
                                            log_level=0).elapsed(reset=True) / prefetch_num_gb
                extra_time_metrics = {
                    "time_perf/get_batched_data": prefetch_data_time,
                    "time_perf/teacher_infer_time": teacher_infer_time,
                    "time_perf/train_step_total": prefetch_data_time + teacher_infer_time,
                }

                with OnloadStudentEngineManager(self.model_engine):
                    for expanded_rbs in prefetch_rbs_list:
                        if train_step == training_config.exit_step:
                            epoch_exit = True
                            break

                        metric = self.train_one_step(
                            epoch,
                            train_step,
                            expanded_rbs,
                            extra_time_metrics=extra_time_metrics,
                        )
                        if self.config.debug.trainer_return_ppo_step_metrics:
                            collected_metrics.append(metric)
                        cpu_barrier()

                        train_step += 1
                        cur_epoch_train_step += 1
                        if (
                            train_step % training_config.save_interval == 0 and
                            not self.config.debug.disable_save_checkpoint
                        ):
                            self.model_engine.save_checkpoint(
                                train_step, dataloader=self.train_dataloader
                            )

                if self.config.tq.enable:
                    cpu_barrier()
                    if is_mp_and_cp_head() and mpu.get_data_parallel_rank() == 0:
                        await get_tq_connector().async_clear_step(prefetch_train_step)
                    cpu_barrier()

                if epoch_exit:
                    break

            if epoch_exit:
                break

        self.train_step_finished = True

        cpu_barrier()
        if not self.config.debug.disable_save_checkpoint:
            if train_step % training_config.save_interval != 0:
                self.model_engine.save_checkpoint(train_step, dataloader=self.train_dataloader)

        if is_last_rank():
            TrainReporterSingleton.finish()

        return collected_metrics
