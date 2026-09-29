import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)

from gpatch_v4.generation_backend.sglang_engine import SglangEngine


class SglangEngineCudaGraphLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tokenizer_manager = SimpleNamespace(
            flush_cache=AsyncMock(),
            release_memory_occupation=AsyncMock(),
            resume_memory_occupation=AsyncMock(),
        )
        infer_engine = SimpleNamespace(tokenizer_manager=self.tokenizer_manager)
        with patch.dict(
            "os.environ", {"SGLANG_MEMORY_SAVER_CUDA_GRAPH": "1"}
        ):
            self.engine = SglangEngine(
                infer_engine=infer_engine,
                model_path="/tmp/model",
                infer_engine_role="sampler",
                placement_type="colocate",
            )

    async def test_disabled_offload_keeps_cuda_graph_resident(self) -> None:
        with patch.dict(
            "os.environ", {"SGLANG_MEMORY_SAVER_CUDA_GRAPH": "0"}
        ):
            engine = SglangEngine(
                infer_engine=SimpleNamespace(
                    tokenizer_manager=self.tokenizer_manager
                ),
                model_path="/tmp/model",
                infer_engine_role="sampler",
                placement_type="colocate",
            )

        await engine.sleep()

        req = self.tokenizer_manager.release_memory_occupation.await_args.args[0]
        self.assertEqual(
            req.tags,
            [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE],
        )
        self.assertNotIn(GPU_MEMORY_TYPE_CUDA_GRAPH, engine.wake_up_tag)

    async def test_default_sleep_releases_all_memory_types(self) -> None:
        await self.engine.sleep()

        req = self.tokenizer_manager.release_memory_occupation.await_args.args[0]
        self.assertEqual(
            req.tags,
            [
                GPU_MEMORY_TYPE_WEIGHTS,
                GPU_MEMORY_TYPE_KV_CACHE,
                GPU_MEMORY_TYPE_CUDA_GRAPH,
            ],
        )
        self.assertFalse(any(self.engine.wake_up_tag.values()))

    async def test_weight_only_wake_keeps_kv_cache_and_cuda_graph_asleep(
        self,
    ) -> None:
        await self.engine.sleep()
        self.tokenizer_manager.resume_memory_occupation.reset_mock()

        await self.engine.wake_up(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        req = self.tokenizer_manager.resume_memory_occupation.await_args.args[0]
        self.assertEqual(req.tags, [GPU_MEMORY_TYPE_WEIGHTS])
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_WEIGHTS])
        self.assertFalse(self.engine.wake_up_tag[GPU_MEMORY_TYPE_KV_CACHE])
        self.assertFalse(self.engine.wake_up_tag[GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def test_kv_cache_wake_also_resumes_cuda_graph(self) -> None:
        await self.engine.sleep()
        self.tokenizer_manager.resume_memory_occupation.reset_mock()

        await self.engine.wake_up(tags=[GPU_MEMORY_TYPE_KV_CACHE])

        req = self.tokenizer_manager.resume_memory_occupation.await_args.args[0]
        self.assertEqual(
            req.tags,
            [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
        )
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_KV_CACHE])
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def test_final_sleep_after_weight_only_wake_does_not_flush_cache(self) -> None:
        await self.engine.sleep()
        await self.engine.wake_up(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        self.tokenizer_manager.flush_cache.reset_mock()
        self.tokenizer_manager.release_memory_occupation.reset_mock()

        await self.engine.sleep()

        self.tokenizer_manager.flush_cache.assert_not_awaited()
        self.tokenizer_manager.release_memory_occupation.assert_awaited_once()
        req = self.tokenizer_manager.release_memory_occupation.await_args.args[0]
        self.assertEqual(req.tags, [GPU_MEMORY_TYPE_WEIGHTS])
        self.assertFalse(any(self.engine.wake_up_tag.values()))

        await self.engine.sleep()
        self.tokenizer_manager.release_memory_occupation.assert_awaited_once()

    async def test_weight_only_sleep_flushes_when_kv_cache_is_awake(self) -> None:
        events = []

        async def flush_cache(**kwargs):
            events.append("flush")

        async def release_memory_occupation(req, _):
            events.append("release")
            self.assertEqual(req.tags, [GPU_MEMORY_TYPE_WEIGHTS])
            self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_WEIGHTS])

        self.tokenizer_manager.flush_cache.side_effect = flush_cache
        self.tokenizer_manager.release_memory_occupation.side_effect = release_memory_occupation

        await self.engine.sleep(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        self.assertEqual(events, ["flush", "release"])
        self.assertFalse(self.engine.wake_up_tag[GPU_MEMORY_TYPE_WEIGHTS])
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_KV_CACHE])
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def test_sleep_waits_for_release_before_marking_weights_asleep(self) -> None:
        await self.engine.sleep()
        await self.engine.wake_up(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        release_started = asyncio.Event()
        release_done = asyncio.Event()

        async def release_memory_occupation(req, _):
            release_started.set()
            await release_done.wait()

        self.tokenizer_manager.release_memory_occupation.side_effect = release_memory_occupation
        task = asyncio.create_task(self.engine.sleep())
        try:
            await asyncio.wait_for(release_started.wait(), timeout=5)
            self.assertFalse(task.done())
            self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_WEIGHTS])
        finally:
            release_done.set()
            await asyncio.wait_for(task, timeout=5)

        self.assertFalse(any(self.engine.wake_up_tag.values()))

    async def test_flush_failure_keeps_tags_awake_and_does_not_release(self) -> None:
        initial_tags = self.engine.wake_up_tag.copy()
        self.tokenizer_manager.flush_cache.side_effect = RuntimeError("flush failed")

        with self.assertRaisesRegex(RuntimeError, "flush failed"):
            await self.engine.sleep()

        self.tokenizer_manager.release_memory_occupation.assert_not_awaited()
        self.assertEqual(self.engine.wake_up_tag, initial_tags)

    async def test_release_failure_keeps_tags_awake(self) -> None:
        initial_tags = self.engine.wake_up_tag.copy()
        self.tokenizer_manager.release_memory_occupation.side_effect = RuntimeError("release failed")

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            await self.engine.sleep()

        self.assertEqual(self.engine.wake_up_tag, initial_tags)

    async def test_weight_only_release_rejection_is_not_swallowed(self) -> None:
        await self.engine.sleep()
        await self.engine.wake_up(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        self.tokenizer_manager.flush_cache.reset_mock()
        self.tokenizer_manager.release_memory_occupation.side_effect = AssertionError("not idle")

        with self.assertRaisesRegex(AssertionError, "not idle"):
            await self.engine.sleep()

        self.tokenizer_manager.flush_cache.assert_not_awaited()
        self.assertTrue(self.engine.wake_up_tag[GPU_MEMORY_TYPE_WEIGHTS])
        self.assertFalse(self.engine.wake_up_tag[GPU_MEMORY_TYPE_KV_CACHE])
        self.assertFalse(self.engine.wake_up_tag[GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def test_weight_update_kv_lifecycle_includes_cuda_graph(self) -> None:
        await self.engine.release_kv_cache_for_weight_update()

        release_req = (
            self.tokenizer_manager.release_memory_occupation.await_args.args[0]
        )
        self.assertEqual(
            release_req.tags,
            [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
        )

        await self.engine.resume_kv_cache_after_weight_update()

        resume_req = (
            self.tokenizer_manager.resume_memory_occupation.await_args.args[0]
        )
        self.assertEqual(
            resume_req.tags,
            [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
        )
