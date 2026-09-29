import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from gpatch_v4.configs.config import FinetuneConfig, OffPolicyDistillConfig
from gpatch_v4.extended_model.qwen3_vl import (
    Qwen3VLOffPoilicyDistillPrepareDataForward,
    Qwen3VLPrepareDataForward,
)
from gpatch_v4.configs.config import FinetuneConfig, OffPolicyDistillConfig
from gpatch_v4.utils.dynamic_cp_utils import (
    flatten_teacher_hidden_for_sft_pack,
    restore_packed_teacher_hidden,
    reverse_reroute_hidden_states,
)


class TestQwen3VLDynamicCp(unittest.TestCase):

    def _reroute_rl_text_only_batch(self, token_weights=None, routed_experts=None):
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            ),
            training=SimpleNamespace(moe_router_replay=routed_experts is not None),
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batch = {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "prompt_lengths": torch.tensor(1),
            "sequence_lengths": torch.tensor(4),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "mask": torch.tensor([True, True, True]),
            "sample_mask": torch.tensor(1.0),
        }
        if token_weights is not None:
            batch["token_weights"] = token_weights
        if routed_experts is not None:
            batch["routed_experts"] = routed_experts
        captured = {}

        def schedule(samples, *args, **kwargs):
            captured["packed_keys"] = kwargs["packed_keys"]
            return samples, 1, 3.0, 9.0, {}

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_rank",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_default",
            side_effect=schedule,
        ):
            samples, _, _, _, _ = prepare_data.rl_reroute_data_for_dynamic_cp(
                [batch],
                pad_token_id=0,
            )

        return samples[0], captured["packed_keys"]

    def test_rl_text_only_batch_expands_scalar_token_weight(self):
        sample, packed_keys = self._reroute_rl_text_only_batch(torch.tensor([2.5]))

        self.assertIn("token_weights", packed_keys)
        self.assertEqual(sample["token_weights"].tolist(), [2.5, 2.5, 2.5])

    def test_rl_text_only_batch_preserves_per_token_weights(self):
        sample, packed_keys = self._reroute_rl_text_only_batch(
            torch.tensor([0.5, 1.0, 1.5])
        )

        self.assertIn("token_weights", packed_keys)
        self.assertEqual(sample["token_weights"].tolist(), [0.5, 1.0, 1.5])

    def test_rl_text_only_batch_keeps_token_weights_optional(self):
        sample, packed_keys = self._reroute_rl_text_only_batch()

        self.assertNotIn("token_weights", packed_keys)
        self.assertNotIn("token_weights", sample)

    def test_rl_reroute_packs_routed_experts(self):
        """moe_router_replay 时 routed_experts 要进 packed_keys，并和 shifted tokens 对齐。"""
        routed = torch.arange(4 * 2 * 2).view(4, 2, 2)
        sample, packed_keys = self._reroute_rl_text_only_batch(routed_experts=routed)

        self.assertIn("routed_experts", packed_keys)
        self.assertEqual(sample["routed_experts"].shape, (3, 2, 2))
        self.assertTrue(torch.equal(sample["routed_experts"], routed[:3]))

    def test_grpo_dynamic_cp_shards_routed_experts_with_labels(self):
        """R3 metadata must follow the CP-local router token stream."""
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        cp_group = MagicMock()
        cp_group.size.return_value = 2
        cp_group.rank.return_value = 0
        tp_group = MagicMock()
        tp_group.size.return_value = 1
        routed = torch.arange(8 * 2 * 2).view(8, 2, 2)
        cp_index = torch.tensor([0, 1, 6, 7])
        batch = {
            "tokens": torch.arange(8),
            "labels": 10 + torch.arange(8),
            "position_ids": torch.arange(24),
            "image_input_mask": torch.zeros(8, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "routed_experts": routed.clone(),
            "local_cp_size": torch.tensor(2),
            "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
            "max_seqlen": torch.tensor(8),
        }

        with patch.object(
            torch.Tensor,
            "cuda",
            lambda tensor, *args, **kwargs: tensor,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_dynamic_data_context_parallel_groups",
            return_value=cp_group,
            create=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_tensor_model_parallel_group",
            return_value=tp_group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.get_thd_partitioned_indices",
            return_value=cp_index,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.PackedSeqParams",
        ):
            prepared, _ = prepare_data.grpo_train_with_dynamic_cp(
                [batch],
                seqlen=8,
                pad_token_id=0,
                ppo_pack_seq=True,
            )

        self.assertEqual(prepared["tokens"].shape, (1, 8))
        self.assertEqual(prepared["target"].shape, (1, 4))
        self.assertEqual(prepared["routed_experts"].shape, (4, 2, 2))
        self.assertTrue(
            torch.equal(prepared["routed_experts"], routed.index_select(0, cp_index))
        )

    def test_grpo_dynamic_cp_keeps_token_weights_for_loss(self):
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batch = {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, 4]),
            "position_ids": torch.arange(9),
            "image_input_mask": torch.zeros(3, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "token_weights": torch.tensor([2.5, 2.5, 2.5]),
            "local_cp_size": torch.tensor(1),
            "cu_seqlens_padded": torch.tensor([0, 3], dtype=torch.int32),
            "max_seqlen": torch.tensor(3),
        }

        with patch.object(
            torch.Tensor,
            "cuda",
            lambda tensor, *args, **kwargs: tensor,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_dynamic_data_context_parallel_groups",
            return_value=group,
            create=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.PackedSeqParams",
        ):
            prepared, _ = prepare_data.grpo_train_with_dynamic_cp(
                [batch],
                seqlen=3,
                pad_token_id=0,
                ppo_pack_seq=True,
            )

        self.assertEqual(prepared["token_weights"].shape, (1, 3))
        self.assertEqual(prepared["token_weights"].tolist(), [[2.5, 2.5, 2.5]])

    def test_sft_text_only_batch_accepts_none_vision_fields(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="smart_padding",
            max_seqlen_per_dp_cp_rank=8192,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_vl",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
        }]

        def schedule(samples, *args, **kwargs):
            return samples, 1, 4.0, 16.0

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_smart_padding",
            side_effect=schedule,
        ):
            samples, _, _, _ = prepare_data.sft_reroute_data_for_dynamic_cp(
                batches,
                pad_token_id=0,
            )

        self.assertIsNone(samples[0]["vision_data"])
        self.assertIsNone(samples[0]["vision_grid_thw"])

    def test_sft_reroute_returns_routing_when_requested(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="default",
            max_seqlen_per_dp_cp_rank=8192,
            min_dynamic_context_parallel_size=1,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(dist_config=dist_config, model_arch="qwen3_vl")
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
        }]
        routing = {"global_ids_this_rank": torch.tensor([0])}

        def schedule(*args, **kwargs):
            self.assertTrue(kwargs.get("need_routing_info"))
            return args[0], 1, 4.0, 16.0, routing

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_default",
            side_effect=schedule,
        ):
            samples, n_mb, _, _, routing_info = prepare_data.sft_reroute_data_for_dynamic_cp(
                batches,
                pad_token_id=0,
                need_routing_info=True,
            )

        self.assertEqual(n_mb, 1)
        self.assertIs(routing_info, routing)
        self.assertEqual(len(samples), 1)

    def test_default_vision_type_fp32_for_sft_and_opd(self):
        prepare_data = Qwen3VLPrepareDataForward(
            SimpleNamespace(policy=SimpleNamespace(model_arch="qwen3_vl"))
        )
        self.assertEqual(
            prepare_data._get_default_vision_type(object.__new__(FinetuneConfig)),
            torch.float32,
        )
        self.assertEqual(
            prepare_data._get_default_vision_type(object.__new__(OffPolicyDistillConfig)),
            torch.float32,
        )
        self.assertEqual(
            prepare_data._get_default_vision_type(SimpleNamespace()),
            torch.bfloat16,
        )

    def test_sft_reroute_keeps_fp32_vision_data_for_opd(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="default",
            max_seqlen_per_dp_cp_rank=8192,
            min_dynamic_context_parallel_size=1,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(dist_config=dist_config, model_arch="qwen3_vl")
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        captured = {}
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": torch.randn(4, 8, dtype=torch.float32),
            "vision_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        }]

        def schedule(*args, **kwargs):
            captured["samples"] = args[0]
            return args[0], 1, 4.0, 16.0, {}

        with patch.object(
            prepare_data,
            "_get_default_vision_type",
            return_value=torch.float32,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_default",
            side_effect=schedule,
        ):
            samples, _, _, _, _ = prepare_data.sft_reroute_data_for_dynamic_cp(
                batches,
                pad_token_id=0,
                need_routing_info=True,
            )

        packed = captured["samples"][0]["vision_data"]
        self.assertEqual(packed.dtype, torch.float32)
        self.assertEqual(samples[0]["vision_data"].dtype, torch.float32)

    def test_sft_reroute_rejects_fp32_vision_when_expected_bf16(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="default",
            max_seqlen_per_dp_cp_rank=8192,
            min_dynamic_context_parallel_size=1,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(dist_config=dist_config, model_arch="qwen3_vl")
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": torch.randn(4, 8, dtype=torch.float32),
            "vision_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        }]

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ):
            with self.assertRaises(AssertionError):
                prepare_data.sft_reroute_data_for_dynamic_cp(
                    batches,
                    pad_token_id=0,
                    need_routing_info=True,
                )

    def test_distill_sft_reroute_packs_teacher_hidden(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="default",
            max_seqlen_per_dp_cp_rank=8192,
            min_dynamic_context_parallel_size=1,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(dist_config=dist_config, model_arch="qwen3_vl")
        )
        prepare_data = Qwen3VLOffPoilicyDistillPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        hidden_size = 6
        hidden = torch.arange(4 * hidden_size).view(4, hidden_size).float()
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "teacher_hidden_states": hidden.clone(),
        }]
        captured = {}

        def schedule(*args, **kwargs):
            packed_keys = kwargs.get("packed_keys", args[8])
            captured["packed_keys"] = packed_keys
            captured["gid_keys"] = kwargs.get("global_id_seqlens_keys", args[10])
            return args[0], 1, 4.0, 16.0, {}

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.is_pipeline_last_stage",
            return_value=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_default",
            side_effect=schedule,
        ):
            samples, _, _, _ = prepare_data.sft_reroute_data_for_dynamic_cp(
                batches,
                pad_token_id=0,
            )

        self.assertIn("teacher_hidden_states", captured["packed_keys"])
        self.assertIn("teacher_hidden_states", captured["gid_keys"])
        packed_hidden = samples[0]["teacher_hidden_states"]
        tokens = samples[0]["tokens"]
        self.assertEqual(packed_hidden.numel(), tokens.numel() * hidden_size)
        self.assertEqual(packed_hidden.shape[-1], hidden_size)

    def test_distill_sft_train_dynamic_cp_slices_hidden_with_labels(self):
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(dist_config=dist_config, model_arch="qwen3_vl")
        )
        prepare_data = Qwen3VLOffPoilicyDistillPrepareDataForward(config)
        cp_group = MagicMock()
        cp_group.size.return_value = 2
        cp_group.rank.return_value = 0
        tp_group = MagicMock()
        tp_group.size.return_value = 1
        hidden_size = 5
        hidden = torch.arange(8 * hidden_size).view(8, hidden_size).float()
        cp_index = torch.tensor([0, 1, 6, 7])
        batch = {
            "tokens": torch.arange(8),
            "labels": 10 + torch.arange(8),
            "loss_mask": torch.ones(8),
            "position_ids": torch.arange(24),
            "image_input_mask": torch.zeros(8, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "teacher_hidden_states": hidden.clone(),
            "local_cp_size": torch.tensor(2),
            "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
            "max_seqlen": torch.tensor(8),
        }

        with patch.object(
            torch.Tensor,
            "cuda",
            lambda tensor, *args, **kwargs: tensor,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_dynamic_data_context_parallel_groups",
            return_value=cp_group,
            create=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_tensor_model_parallel_group",
            return_value=tp_group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.get_thd_partitioned_indices",
            return_value=cp_index,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.is_pipeline_last_stage",
            return_value=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.PackedSeqParams",
        ):
            prepared, _ = prepare_data.sft_train_with_dynamic_cp(
                [batch],
                seq_len=8,
                pad_token_id=0,
                input_teacher_hidden_states=True,
            )

        self.assertEqual(prepared["tokens"].shape, (1, 8))
        self.assertEqual(prepared["labels"].shape, (1, 4))
        self.assertEqual(prepared["teacher_hidden_states"].shape, (1, 4, hidden_size))
        self.assertTrue(
            torch.equal(
                prepared["teacher_hidden_states"][0],
                hidden.index_select(0, cp_index),
            )
        )

    def test_flatten_and_restore_teacher_hidden(self):
        hidden = torch.arange(12).view(3, 4).float()
        flat = flatten_teacher_hidden_for_sft_pack(hidden, pad_len=5)
        self.assertEqual(flat.numel(), 20)
        sample = {
            "tokens": torch.arange(5),
            "teacher_hidden_states": flat,
        }
        restore_packed_teacher_hidden(sample)
        self.assertEqual(sample["teacher_hidden_states"].shape, (5, 4))
        self.assertTrue(torch.equal(sample["teacher_hidden_states"][:3], hidden))

    def test_reverse_reroute_2d_hidden_keeps_seq_len(self):
        hidden = torch.arange(12, dtype=torch.float32).view(3, 4)
        group = MagicMock()
        group.size.return_value = 1
        group.rank.return_value = 0

        with patch(
            "gpatch_v4.utils.dynamic_cp_utils.torch.cuda.current_device",
            return_value="cpu",
        ), patch(
            "gpatch_v4.utils.dynamic_cp_utils.dist.all_to_all_single",
        ), patch(
            "gpatch_v4.utils.dynamic_cp_utils.check_all_to_all_single_legal",
        ):
            restored = reverse_reroute_hidden_states(
                {7: hidden},
                torch.tensor([7]),
                [(7, 3)],
                {7: 0},
                {7: [0]},
                group,
                hidden_size=4,
                transport_dtype=torch.float32,
            )

        self.assertEqual(restored[7].shape[0], 3)
        self.assertEqual(restored[7].shape, (3, 4))
        self.assertTrue(torch.equal(restored[7], hidden))


if __name__ == "__main__":
    unittest.main()
