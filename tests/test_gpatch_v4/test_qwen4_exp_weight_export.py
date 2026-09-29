# coding=utf-8
"""qwen4_exp weight exportor registration and Engram HF-shard packing tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.models.qwen4_exp.engram import (
    OwnerShardedNGramEmbedding,
    iter_engram_hf_shards,
)
from gpatch_v4.models.qwen4_exp.weight_export import (
    export_qwen4_exp_bf16_weights,
)
from gpatch_v4.training_backend.fsdp2_backend.weight_exportor import get_weight_exportor


class Qwen4ExpWeightExportTest(unittest.TestCase):
    def test_vlm_export_preserves_vision_and_engram_names(self):
        model = torch.nn.Module()
        model._ep_size = 1
        model.config = SimpleNamespace(text_config=SimpleNamespace(split_ngram_parts=2))
        model.model = torch.nn.Module()
        model.model.visual = torch.nn.Linear(2, 2, bias=False)
        model.model.language_model = torch.nn.Module()
        model.model.language_model.embed_tokens = torch.nn.Embedding(4, 2)
        model.model.language_model.ngram_embedding = OwnerShardedNGramEmbedding(4, 2)
        model.lm_head = torch.nn.Linear(2, 4, bias=False)

        def cpu_shards(table, prefix, parts, *, out_device, yield_on_all_ranks):
            return iter_engram_hf_shards(
                table, prefix, parts, out_device="cpu", yield_on_all_ranks=yield_on_all_ranks
            )

        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor), patch(
            "gpatch_v4.models.qwen4_exp.weight_export._to_export_tensor",
            side_effect=lambda tensor: tensor.detach().to(torch.bfloat16),
        ), patch(
            "gpatch_v4.models.qwen4_exp.weight_export.iter_engram_hf_shards", side_effect=cpu_shards
        ):
            exported = dict(export_qwen4_exp_bf16_weights(model))
        assert set(exported) == {
            "model.visual.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.ngram_embedding.shard_0.weight",
            "model.language_model.ngram_embedding.shard_1.weight",
            "lm_head.weight",
        }
        torch.testing.assert_close(exported["model.visual.weight"], model.model.visual.weight.bfloat16())
        torch.testing.assert_close(
            torch.cat([
                exported[f"model.language_model.ngram_embedding.shard_{index}.weight"]
                for index in range(2)
            ]),
            model.model.language_model.ngram_embedding.weight.bfloat16(),
        )

    def test_exportor_registered_for_qwen4_exp(self):
        exportor = get_weight_exportor(MODEL_ARCH.QWEN4_EXP)
        self.assertIs(exportor, export_qwen4_exp_bf16_weights)

    def test_iter_engram_hf_shards_single_rank_matches_split_parts(self):
        """world_size=1 still emits split_ngram_parts shards (not one owner blob)."""
        torch.manual_seed(0)
        rows, dim, parts = 16, 4, 4
        table = OwnerShardedNGramEmbedding(rows, dim)
        table.weight.data.copy_(
            torch.arange(rows * dim, dtype=torch.float32).view(rows, dim)
        )

        shards = list(
            iter_engram_hf_shards(
                table,
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight",
                parts,
                out_device="cpu",
                yield_on_all_ranks=True,
            )
        )
        self.assertEqual(len(shards), parts)
        rows_per = (rows + parts - 1) // parts
        rebuilt = []
        for i, (key, tensor) in enumerate(shards):
            self.assertEqual(
                key,
                f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight",
            )
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertEqual(tensor.shape[0], rows_per)
            rebuilt.append(tensor.float())
        torch.testing.assert_close(
            torch.cat(rebuilt, dim=0), table.weight.data.float(), rtol=0, atol=0
        )


def _hf_shard_worker(rank: int, world_size: int, init_file: str, full: torch.Tensor, parts: int):
    """world_size != split_ngram_parts：export 必须按 parts 切，且能还原整表。"""
    import datetime
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        rows, dim = full.shape
        table = OwnerShardedNGramEmbedding(rows, dim)
        local_rows = rows // world_size
        table.weight.data.copy_(full[rank * local_rows:(rank + 1) * local_rows])

        shards = list(
            iter_engram_hf_shards(
                table,
                "ngram_embedding.weight",
                parts,
                out_device="cpu",
                yield_on_all_ranks=True,
            )
        )
        assert len(shards) == parts, [k for k, _ in shards]
        rows_per = (rows + parts - 1) // parts
        for i, (key, tensor) in enumerate(shards):
            assert key == f"ngram_embedding.shard_{i}.weight"
            assert tensor.shape == (rows_per, dim)
            expected = full[i * rows_per:(i + 1) * rows_per].bfloat16()
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class Qwen4ExpEngramHfShardMultiprocessTest(unittest.TestCase):
    def test_iter_engram_hf_shards_world_size_ne_split_parts(self):
        # Mirrors EP32 vs HF128: owner count != split_ngram_parts.
        world_size, parts, rows, dim = 2, 4, 16, 3
        full = torch.arange(rows * dim, dtype=torch.float32).view(rows, dim)
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(pathlib.Path(tmp) / "pg_init")
            pathlib.Path(init_file).touch()
            mp.start_processes(
                _hf_shard_worker,
                args=(world_size, init_file, full, parts),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )


if __name__ == "__main__":
    unittest.main()
