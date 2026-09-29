# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from pathlib import Path

from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.training_backend.fsdp2_backend.checkpoint import (
    save_checkpoint,
    save_hf_checkpoint,
)
from gpatch_v4.utils import log


class CheckpointMixin:
    def save_checkpoint(self, global_step: int, dataloader=None):
        if isinstance(self.model, HpModule):
            # HF export is collective; DCP stores only training state to avoid duplicate weights.
            ck = self.config.checkpoint
            if ck.export_hf_save_path is not None:
                save_path = str(Path(ck.export_hf_save_path) / f"{global_step}")
            else:
                save_path = str(Path(ck.save_ckpt_path).expanduser() / "hf" / f"{global_step}")
            preserve_mtp = self.config.policy.model_arch == "deepseek_v4" or (
                self.config.policy.model_arch == "qwen4_exp" and self.model.mtp is not None
            )
            self.model.save_checkpoint_hp(
                save_path,
                orig_ckpt_dir=self.policy_config.hf_model_path,
                preserve_mtp=preserve_mtp,
                save_distributed_ep_shard=ck.save_distributed_ep_shard,
            )
            log(f"save_checkpoint_hp wrote {save_path}", rank=0)
            save_checkpoint(
                self.config,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                global_step=global_step,
                save_dcp_model=False,
                ep_rank=self.model._ep_rank,
            )
            return
        save_checkpoint(
            self.config, self.model, self.optimizer, self.lr_scheduler, global_step=global_step
        )
        if self.config.checkpoint.convert_mcore_to_hf_online:
            save_hf_checkpoint(self.config, self.model, self.tokenizer, global_step=global_step)
        # TODO: Save dataloader state if provided
