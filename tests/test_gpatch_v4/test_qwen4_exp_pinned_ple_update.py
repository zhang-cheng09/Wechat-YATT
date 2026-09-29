import unittest
from types import SimpleNamespace

import torch

from gpatch_v4.generation_backend.sglang_model_specific.sglang_weight_update_qwen4_exp import (  # noqa: E501
    load_qwen4_exp_packed_weights,
)


class _PleModel(torch.nn.Module):
    def __init__(self, rows, dim, start, end, parts, pinned=True, dtype=torch.bfloat16):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(split_ngram_parts=parts))
        self.model = torch.nn.Module()
        self.model.ple = torch.nn.Module()
        embedding = torch.nn.Module()
        embedding.org_vocab_size = rows
        embedding.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end,
        )
        embedding.weight = torch.nn.Parameter(
            torch.full((end - start + 2, dim), -9, dtype=dtype, pin_memory=pinned),
            requires_grad=False,
        )
        self.model.ple.ngram_embedding = embedding
        self.fallback = []

    def load_weights(self, weights):
        self.fallback.extend(weights)
        return 'fallback'


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and pinned host memory')
class Qwen4PinnedPleUpdateTest(unittest.TestCase):
    def test_full_shards_match_tp_rows_with_uneven_boundaries(self):
        rows, dim, parts = 23, 5, 4
        full = torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim).bfloat16()
        rows_per_shard = (rows + parts - 1) // parts
        for tp in (1, 2, 3, 8):
            rows_per_rank = (rows + tp - 1) // tp
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    start, end = rank * rows_per_rank, min((rank + 1) * rows_per_rank, rows)
                    model = _PleModel(rows, dim, start, end, parts)
                    original = model.model.ple.ngram_embedding.weight
                    original_pointer = original.data_ptr()
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        weights = [
                            (f'model.language_model.ple.ngram_embedding.shard_{i}.weight',
                             full[i * rows_per_shard:min((i + 1) * rows_per_shard, rows)].cuda())
                            for i in range(parts)
                        ]
                        load_qwen4_exp_packed_weights(model, weights)
                    target = model.model.ple.ngram_embedding.weight
                    self.assertIs(target, original)
                    self.assertEqual(target.data_ptr(), original_pointer)
                    self.assertTrue(target.is_pinned())
                    self.assertTrue(torch.equal(target[:end - start], full[start:end]))
                    self.assertTrue(torch.all(target[end - start:] == -9))
                    self.assertEqual(model.fallback, [])

    def test_unpinned_or_dtype_conversion_uses_native_loader(self):
        for pinned, dtype in ((False, torch.bfloat16), (True, torch.float32)):
            with self.subTest(pinned=pinned, dtype=dtype):
                model = _PleModel(8, 3, 0, 8, 2, pinned=pinned, dtype=dtype)
                weight = torch.ones(4, 3, device='cuda', dtype=torch.bfloat16)
                named = [('model.ple.ngram_embedding.shard_0.weight', weight)]
                self.assertEqual(load_qwen4_exp_packed_weights(model, named), 'fallback')
                self.assertEqual(model.fallback, named)

    def test_invalid_shard_layout_uses_native_loader(self):
        model = _PleModel(8, 3, 0, 8, 2)
        for key, shape in (('shard_0', (3, 3)), ('shard_2', (4, 3)), ('shard_0', (4, 2))):
            with self.subTest(key=key, shape=shape):
                model.fallback.clear()
                named = [(f'model.ple.ngram_embedding.{key}.weight',
                          torch.ones(shape, device='cuda', dtype=torch.bfloat16))]
                self.assertEqual(load_qwen4_exp_packed_weights(model, named), 'fallback')
                self.assertEqual(model.fallback, named)
