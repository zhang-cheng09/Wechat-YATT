"""Ray actor for Qwen3.8-Flash-Next (qwen4_exp) FSDP2 THD GSPO CP tests."""

import ray


@ray.remote(num_gpus=1)
class Qwen4ExpPolicyLossWorker:
    def get_master_addr_and_port(self):
        import socket

        ip = socket.gethostbyname(socket.gethostname())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return ip, s.getsockname()[1]

    def run_gspo(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        ep_size: int,
        cp_size: int,
        fake_seq_lens: list,
        replay_indices=None,
        packed_prev_log_probs=None,
    ):
        from contextlib import nullcontext
        import math
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        from megatron.core import parallel_state as ps
        from gpatch_v4.configs.debug_config import DebugConfig
        from gpatch_v4.configs.ppo_config import PpoConfig
        from gpatch_v4.extended_model.qwen4_exp import Qwen4ExpPrepareDataForwardLLM
        from gpatch_v4.models.comm.router_replay import capture_routing_decisions
        from gpatch_v4.models.qwen4_exp import (
            Qwen4ExpConfig,
            Qwen4ExpHpForConditionalGeneration,
            Qwen4ExpTextConfig,
            apply_hp,
        )
        from gpatch_v4.models.qwen4_exp.router_replay import router_replay_ctx
        from gpatch_v4.training_backend.fsdp2_backend.mixin import (
            ForwardStepMixin,
            response_pad_fsdp2_thd_loss_tensors,
            thd_response_spans_from_samples,
        )
        from gpatch_v4.training_backend.loss_factory import PolicyLossInput, gspo_loss_func

        import torch
        from torch import distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        tiny_kwargs = dict(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=8,
            full_attention_interval=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            output_gate_type="sigmoid",
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=16,
            indexer_budget=16,
            indexer_compress_ratio=4,
            hc_count=4,
            hc_lowrank=8,
            ple_layer_ids=[2],
            ple_embed_dim=64,
            ple_conv_kernel_size=4,
            ngram_size=3,
            heads_per_ngram=2,
            ngram_vocab_size_base=512,
            make_ngram_vocab_size_divisible_by=128,
            split_ngram_parts=4,
            eos_token_id=1,
            tie_word_embeddings=False,
            use_cache=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
        )

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        try:
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=cp_size,
            )
            ep_2d_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // ep_size, ep_size),
                mesh_dim_names=("ep_fsdp", "ep"),
            )
            if cp_size > 1:
                cp_mesh = init_device_mesh(
                    "cuda",
                    mesh_shape=(world_size // cp_size, cp_size),
                    mesh_dim_names=("dp", "cp"),
                )["cp"]
            else:
                cp_mesh = None

            torch.manual_seed(0)
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                config = Qwen4ExpConfig(
                    text_config=Qwen4ExpTextConfig(**tiny_kwargs),
                    vision_config=dict(
                        depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                        patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                        out_hidden_size=tiny_kwargs["hidden_size"], num_position_embeddings=16,
                    ),
                )
                model = Qwen4ExpHpForConditionalGeneration(config)
            finally:
                torch.set_default_dtype(prev_dtype)
            model = model.to("cuda")
            model = apply_hp(
                model,
                ep_2d_mesh,
                cp_mesh=cp_mesh,
                attn_backend="dense",
                ep_backend="eager",
            )

            pad_id = 0
            device = torch.device("cuda")
            batches = []
            for i, seq_len in enumerate(fake_seq_lens):
                rng = torch.Generator(device="cpu").manual_seed(20260530 + i)
                tokens = torch.randint(2, config.text_config.vocab_size, (seq_len,), generator=rng)
                shifted = seq_len - 1
                batches.append(
                    {
                        "tokens": tokens,
                        "sequence_lengths": torch.tensor(seq_len),
                        "prompt_lengths": torch.tensor(1),
                        "mask": torch.ones(shifted),
                        "advantages": torch.full((shifted,), 0.2),
                        "logprobs": torch.zeros(shifted),
                        "ref_logprobs": torch.zeros(shifted),
                        "rollout_log_probs": torch.zeros(seq_len),
                    }
                )

            prepare = Qwen4ExpPrepareDataForwardLLM.__new__(Qwen4ExpPrepareDataForwardLLM)
            prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True))
            prepare._pad_each_doc_to_multi_of = config.text_config.indexer_compress_ratio
            packed, fwd_kwargs = prepare._grpo_train_thd(
                batches,
                seq_len=64,
                pad_token_id=pad_id,
                include_rl_fields=True,
            )

            mixin = ForwardStepMixin.__new__(ForwardStepMixin)
            mixin.prepare_data = prepare
            mixin.training_config = SimpleNamespace(use_linear_ce=False)

            if replay_indices is not None and len(replay_indices) > 0:
                s_local = fwd_kwargs["input_ids"].shape[1]
                cp_rank = ps.get_context_parallel_rank()
                cp_replay = [
                    t[cp_rank * s_local:(cp_rank + 1) * s_local].to(device=device).long()
                    for t in replay_indices
                ]
                replay_ctx = router_replay_ctx(model, cp_replay)
            else:
                replay_ctx = nullcontext()

            model.train()
            with replay_ctx:
                with capture_routing_decisions(
                    model, router_class_names=("Qwen4ExpTextTopKRouter",)
                ) as recorded:
                    outputs = model(**fwd_kwargs)
                    curr_log_probs, entropy = mixin._rl_logprobs_and_entropy_from_head_output(
                        outputs.logits,
                        packed["target"],
                        packed["mask"],
                        pre_shifted=True,
                        allow_compile=False,
                    )
                    prev_packed = (
                        packed_prev_log_probs.to(
                            device=curr_log_probs.device, dtype=curr_log_probs.dtype
                        ) if packed_prev_log_probs is not None else curr_log_probs.detach()
                    )
                    if prev_packed.shape != curr_log_probs.shape:
                        raise RuntimeError(
                            f"prev {tuple(prev_packed.shape)} vs curr {tuple(curr_log_probs.shape)}"
                        )
                    starts, lengths = thd_response_spans_from_samples(
                        batches, device=curr_log_probs.device
                    )
                    padded = response_pad_fsdp2_thd_loss_tensors(
                        {
                            "curr_log_probs": curr_log_probs,
                            "per_token_entropy": entropy,
                            "mask": packed["mask"],
                            "advantages": packed["advantages"],
                            "prev_log_probs": prev_packed,
                            "ref_log_probs": packed["ref_log_probs"],
                            "rollout_log_probs": packed["rollout_log_probs"],
                        },
                        packed["cu_seqlens_padded"],
                        starts,
                        lengths,
                    )
                    ppo_config = SimpleNamespace(
                        ppo=PpoConfig(
                            loss_func="gspo",
                            ppo_entropy_bonus=0.0,
                            grpo_kl_loss_beta=0.0,
                            enable_off_policy_correction=False,
                            ppo_ratio_eps=0.2,
                        ),
                        debug=DebugConfig(),
                    )
                    _reduce = (
                        "gpatch_v4.training_backend.loss_factory."
                        "reduce_metrics_across_data_parallel_group"
                    )
                    with patch(_reduce):
                        loss, _metrics = gspo_loss_func(
                            ppo_config,
                            PolicyLossInput(
                                advantages=padded["advantages"],
                                prev_log_probs=padded["prev_log_probs"],
                                ref_log_probs=padded["ref_log_probs"],
                                curr_log_probs=padded["curr_log_probs"],
                                response_mask=padded["mask"],
                                scaled_entropy=padded["scaled_entropy"],
                                rollout_log_probs=padded["rollout_log_probs"],
                                per_token_entropy=padded["per_token_entropy"],
                                cu_seqlens_padded=None,
                                local_cp_size=1,
                            ),
                        )
                    reported_loss = loss.detach().clone()
                    loss.backward()

            total_grad_norm = model.clip_grad_norm_(2.0)
            print(
                f"[qwen4_exp_gspo_cp{cp_size}] rank {rank}: "
                f"loss={reported_loss.item():.6f} grad_norm={total_grad_norm:.6f} "
                f"local_tokens={fwd_kwargs['input_ids'].shape[1]} "
                f"packed_T={packed['target'].shape[1]}"
            )
            recorded_routing = None
            packed_prev_out = None
            if replay_indices is None:
                for i, t in enumerate(recorded):
                    if t is None:
                        raise RuntimeError(f"layer {i} routing not captured")
                recorded_routing = [t.detach().cpu() for t in recorded]
                packed_prev_out = curr_log_probs.detach().cpu()

            result = {
                "rank": rank,
                "loss": reported_loss.item(),
                "total_grad_norm": float(total_grad_norm),
                "recorded_routing": recorded_routing,
                "packed_prev": packed_prev_out,
                "logits_finite": bool(torch.isfinite(outputs.logits).all()),
                "has_nan": math.isnan(float(total_grad_norm)),
            }
        finally:
            if hasattr(ps, "destroy_model_parallel"):
                ps.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()
        return result
