from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.models.qwen4_exp.cp import (
    Qwen4ExpCPContext,
    Qwen4ExpPackedSeqParams,
)
from gpatch_v4.models.qwen4_exp.diagnostics import (
    compact_tensor_fingerprint,
    packed_microbatch_fingerprint,
    parameter_fingerprint,
    tensor_fingerprint,
)


def test_tensor_and_parameter_fingerprints_detect_bit_changes() -> None:
    tensor = torch.arange(16, dtype=torch.int64).reshape(2, 8)
    same = tensor.clone()
    changed = tensor.clone()
    changed[0, 0] = -1

    assert tensor_fingerprint(tensor) == tensor_fingerprint(same)
    assert tensor_fingerprint(tensor)["sha256"] != tensor_fingerprint(changed)["sha256"]
    assert compact_tensor_fingerprint(tensor)["hash_kind"] == "full"
    assert (
        compact_tensor_fingerprint(tensor)["sha256"] !=
        compact_tensor_fingerprint(changed)["sha256"]
    )

    model = torch.nn.Linear(8, 4, bias=False)
    before = parameter_fingerprint(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    after = parameter_fingerprint(model)
    assert before["sample_sha256"] != after["sample_sha256"]
    assert before["value_sum_fp64"] != after["value_sum_fp64"]


def test_packed_microbatch_fingerprint_covers_cp_metadata() -> None:
    input_ids = torch.tensor([[10, 11, 12, 20, 21, 0, 0, 0]])
    padding_mask = input_ids == 0
    boundaries = torch.tensor([0, 3, 5], dtype=torch.long)
    padded_boundaries = torch.tensor([0, 3, 8], dtype=torch.long)
    packed = Qwen4ExpPackedSeqParams(
        cu_seqlens_q=boundaries,
        cu_seqlens_q_padded=padded_boundaries,
        max_seqlen_q=5,
        total_seqlen=8,
    )
    assert packed.cp_partition_mode == "contiguous"
    cp_context = Qwen4ExpCPContext(
        group=None,
        rank=0,
        size=1,
        global_input_ids=input_ids,
        global_padding_mask=padding_mask,
        local_sequence_start=0,
        local_sequence_length=8,
        global_cu_seqlens=boundaries,
    )

    record = packed_microbatch_fingerprint(
        {
            "labels": input_ids.clone(),
            "full_packed_seq_params": packed,
        },
        {
            "input_ids": input_ids,
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]]),
            "_qwen4_exp_cp_context": cp_context,
        },
        microbatch_index=2,
    )

    assert record["microbatch_index"] == 2
    assert record["cp_metadata"]["global_document_boundaries"] == [0, 3, 5]
    assert record["cp_metadata"]["global_segment_boundary_values"] == [0, 3, 5, 8]
    assert "packed.cu_seqlens_q_padded" in record["tensors"]
    assert "cp.global_segment_boundaries" in record["tensors"]


def test_model_diagnostic_hooks_configure_record_and_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from gpatch_v4.models.qwen4_exp.vl import Qwen4ExpHpForConditionalGeneration

    model_cls = Qwen4ExpHpForConditionalGeneration
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_type="qwen4_exp")
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Linear(2, 2, bias=False)
    layer.mlp.experts._deterministic_deepep_reduce = False
    layer.mlp.experts._deepep_probe_dir = None
    text_model = torch.nn.Module()
    text_model.layers = torch.nn.ModuleList([layer])
    model.model = torch.nn.Module()
    model.model.language_model = text_model
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    debug = SimpleNamespace(
        deterministic_deepep_reduce=True,
        step0_diagnostics=True,
        step0_diagnostics_dir=str(tmp_path),
        deepep_probe_layer=0,
    )

    model.configure_debug_diagnostics(debug, SimpleNamespace(ep_backend="deepep"))
    assert layer.mlp.experts._deterministic_deepep_reduce is True
    assert layer.mlp.experts._deepep_probe_dir == str(tmp_path)
    tokens = torch.tensor([[1, 2]])
    model.maybe_dump_prepared_microbatch_diagnostics({"labels": tokens}, {"input_ids": tokens}, 0)
    model.finish_step0_diagnostics()
    assert layer.mlp.experts._deepep_probe_dir is None
    model.maybe_dump_prepared_microbatch_diagnostics({"labels": tokens}, {"input_ids": tokens}, 1)

    records = [json.loads(line) for line in (tmp_path / "rank_00000.jsonl").read_text().splitlines()]
    assert [record["kind"] for record in records] == ["initial_parameters", "packed_microbatch"]
    assert records[1]["microbatch_index"] == 0
