"""Unit tests for packed Qwen4-Exp SGLang weight updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.generation_backend.sglang_model_specific.sglang_weight_update_qwen4_exp import (  # noqa: E501
    QWEN4_EXP_PACKED_WEIGHT_LOADER,
    install_packed_weight_loader,
    load_qwen4_exp_packed_weights,
)


class _Experts(torch.nn.Module):
    def __init__(
        self, intermediate_dim=3, hidden_dim=4, tp_size=1, tp_rank=0,
        use_triton_kernels=False,
    ):
        super().__init__()
        w13_shape = (2, 2 * intermediate_dim // tp_size, hidden_dim)
        w2_shape = (2, hidden_dim, intermediate_dim // tp_size)
        if use_triton_kernels:
            w13_shape = (2, hidden_dim, 2 * intermediate_dim // tp_size)
            w2_shape = (2, intermediate_dim // tp_size, hidden_dim)
        self.w13_weight = torch.nn.Parameter(torch.zeros(w13_shape, dtype=torch.bfloat16))
        self.w2_weight = torch.nn.Parameter(torch.zeros(w2_shape, dtype=torch.bfloat16))
        self.quant_config = None
        self.moe_ep_size = 1
        self.num_local_experts = 2
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        self.moe_tp_size = tp_size
        self.moe_tp_rank = tp_rank
        self.use_presharded_weights = False
        self.use_padded_loading = False
        self.use_triton_kernels = use_triton_kernels
        self.calls = []

    def weight_loader_fused(self, param, weight, weight_name, shard_id):
        self.calls.append((param, weight, weight_name, shard_id))
        param.copy_(weight)


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


class _CoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CoreModel()
        self.fallback = []

    def load_weights(self, named_tensors):
        self.fallback.extend(named_tensors)
        return "fallback-result"


class Qwen4ExpSglangWeightUpdateTest(unittest.TestCase):
    def setUp(self):
        self.location = patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=None,
        )
        self.backend = patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            return_value=SimpleNamespace(
                is_triton=lambda: True, is_triton_kernels=lambda: False,
            ),
        )
        self.location_mock = self.location.start()
        self.backend_mock = self.backend.start()
        self.addCleanup(self.location.stop)
        self.addCleanup(self.backend.stop)

    def test_loads_each_packed_expert_tensor_in_one_call(self):
        model = _Model()
        gate_up = torch.ones(2, 6, 4, dtype=torch.bfloat16)
        down = torch.full((2, 4, 3), 2, dtype=torch.bfloat16)

        result = load_qwen4_exp_packed_weights(
            model,
            [
                ("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.language_model.layers.0.mlp.experts.down_proj", down),
            ],
        )

        calls = model.model.layers[0].mlp.experts.calls
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2:], ("model.layers.0.mlp.experts.w2_weight", "w2"))
        self.assertTrue(torch.equal(model.model.layers[0].mlp.experts.w13_weight, gate_up))
        self.assertTrue(torch.equal(model.model.layers[0].mlp.experts.w2_weight, down))

    def test_gate_up_tp8_matches_separate_gate_and_up_slices(self):
        gate_up = torch.arange(2 * 32 * 4, dtype=torch.float32).reshape(2, 32, 4)
        gate_up = gate_up.to(torch.bfloat16)
        for rank in (0, 3, 7):
            with self.subTest(rank=rank):
                model = _Model()
                experts = _Experts(intermediate_dim=16, tp_size=8, tp_rank=rank)
                model.model.layers[0].mlp.experts = experts
                load_qwen4_exp_packed_weights(
                    model,
                    [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)],
                )
                start = rank * 2
                expected = torch.cat(
                    (gate_up[:, start:start + 2], gate_up[:, 16 + start:18 + start]),
                    dim=1,
                )
                self.assertTrue(torch.equal(experts.w13_weight, expected))
                self.assertEqual(experts.calls, [])
                self.assertEqual(model.fallback, [])

    def test_gate_up_triton_kernel_transposed_layout(self):
        model = _Model()
        experts = _Experts(
            intermediate_dim=16, tp_size=8, tp_rank=5, use_triton_kernels=True,
        )
        model.model.layers[0].mlp.experts = experts
        self.backend_mock.return_value = SimpleNamespace(
            is_triton=lambda: False, is_triton_kernels=lambda: True,
        )
        gate_up = torch.arange(2 * 32 * 4, dtype=torch.float32).reshape(2, 32, 4)
        gate_up = gate_up.to(torch.bfloat16)
        load_qwen4_exp_packed_weights(
            model,
            [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)],
        )
        expected = torch.cat((gate_up[:, 10:12], gate_up[:, 26:28]), dim=1)
        expected = expected.transpose(-2, -1)
        self.assertTrue(torch.equal(experts.w13_weight, expected))

    def test_gate_up_tp8_matches_sglang_fused_moe_loader(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        gate_up = torch.arange(2 * 32 * 4, dtype=torch.float32).reshape(2, 32, 4)
        gate_up = gate_up.to(torch.bfloat16)
        gate, up = gate_up.chunk(2, dim=1)

        for transposed in (False, True):
            self.backend_mock.return_value = SimpleNamespace(
                is_triton=lambda: not transposed,
                is_triton_kernels=lambda: transposed,
            )
            for rank in (0, 3, 7):
                with self.subTest(transposed=transposed, rank=rank):
                    model = _Model()
                    experts = _Experts(
                        intermediate_dim=16,
                        tp_size=8,
                        tp_rank=rank,
                        use_triton_kernels=transposed,
                    )
                    model.model.layers[0].mlp.experts = experts
                    load_qwen4_exp_packed_weights(
                        model,
                        [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)],
                    )

                    reference_w13 = torch.zeros_like(experts.w13_weight)
                    sglang_loader = SimpleNamespace(
                        moe_runner_config=SimpleNamespace(is_gated=True),
                        moe_tp_size=8,
                        quant_method=SimpleNamespace(),
                        quant_config=None,
                        use_padded_loading=False,
                        use_presharded_weights=False,
                        use_triton_kernels=transposed,
                    )
                    w13_dim = 2 if transposed else 1
                    FusedMoE._load_w13(
                        sglang_loader, reference_w13, w13_dim, "w1", gate, rank
                    )
                    FusedMoE._load_w13(
                        sglang_loader, reference_w13, w13_dim, "w3", up, rank
                    )
                    self.assertTrue(torch.equal(experts.w13_weight, reference_w13))

    def test_eplb_delegates_to_model_loader(self):
        model = _Model()
        self.location_mock.return_value = SimpleNamespace(
            ep_size=1,
            physical_to_logical_map_cpu=torch.tensor([[1, 0]], dtype=torch.long),
        )
        gate_up = torch.ones(2, 6, 4, dtype=torch.bfloat16)
        named_tensors = [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)]

        self.assertEqual(load_qwen4_exp_packed_weights(model, named_tensors), "fallback-result")
        self.assertEqual(model.fallback, named_tensors)
        self.assertEqual(model.model.layers[0].mlp.experts.calls, [])

    def test_trivial_expert_location_uses_fast_path(self):
        model = _Model()
        self.location_mock.return_value = SimpleNamespace(
            ep_size=1,
            physical_to_logical_map_cpu=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        )
        gate_up = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4).bfloat16()

        load_qwen4_exp_packed_weights(
            model,
            [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)],
        )

        self.assertEqual(model.fallback, [])
        self.assertTrue(torch.equal(model.model.layers[0].mlp.experts.w13_weight, gate_up))

    def test_other_backend_delegates_to_model_loader(self):
        model = _Model()
        self.backend_mock.return_value = SimpleNamespace(
            is_triton=lambda: False, is_triton_kernels=lambda: False,
        )
        gate_up = torch.ones(2, 6, 4, dtype=torch.bfloat16)
        named_tensors = [("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)]

        self.assertEqual(load_qwen4_exp_packed_weights(model, named_tensors), "fallback-result")
        self.assertEqual(model.fallback, named_tensors)
        self.assertEqual(model.model.layers[0].mlp.experts.calls, [])

    def test_delegates_non_expert_weights(self):
        model = _Model()
        dense_weight = torch.ones(4, 4)

        result = load_qwen4_exp_packed_weights(
            model,
            [("model.layers.0.self_attn.q_proj.weight", dense_weight)],
        )

        self.assertEqual(result, "fallback-result")
        self.assertEqual(
            model.fallback,
            [("model.layers.0.self_attn.q_proj.weight", dense_weight)],
        )

    def test_delegates_packed_experts_when_ep_is_enabled(self):
        model = _Model()
        model.model.layers[0].mlp.experts.moe_ep_size = 2
        gate_up = torch.ones(2, 6, 4, dtype=torch.bfloat16)
        named_tensors = [
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", gate_up)
        ]

        result = load_qwen4_exp_packed_weights(model, named_tensors)

        self.assertEqual(result, "fallback-result")
        self.assertEqual(model.fallback, named_tensors)
        self.assertEqual(model.model.layers[0].mlp.experts.calls, [])


class InstallPackedWeightLoaderTest(unittest.TestCase):
    def test_none_string_and_existing_list(self):
        # 缺省、单个路径、已有列表三种形态都追加且不改原列表
        loader = QWEN4_EXP_PACKED_WEIGHT_LOADER

        missing = {}
        install_packed_weight_loader(missing)
        self.assertEqual(missing["custom_weight_loader"], [loader])

        explicit_none = {"custom_weight_loader": None}
        install_packed_weight_loader(explicit_none)
        self.assertEqual(explicit_none["custom_weight_loader"], [loader])

        one_path = {"custom_weight_loader": "other.loader"}
        install_packed_weight_loader(one_path)
        self.assertEqual(one_path["custom_weight_loader"], ["other.loader", loader])

        existing = ["other.loader"]
        many = {"custom_weight_loader": existing}
        install_packed_weight_loader(many)
        self.assertEqual(many["custom_weight_loader"], ["other.loader", loader])
        self.assertEqual(existing, ["other.loader"])

        install_packed_weight_loader(many)
        self.assertEqual(many["custom_weight_loader"], ["other.loader", loader])


if __name__ == "__main__":
    unittest.main()
