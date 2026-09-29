"""DAPO-Math rule-based external reward for AgentLoopActor.

Same packaging as ``math_external_reward.MathRuleExternalReward``
(``BaseExternalReward`` for the async rollout pipeline), but grading
follows ``bt_reward_dapo``:

* ``mathruler.grader`` for boxed-answer accuracy (fractions / radicals / …)
* decode only the response span ``[prompt_len:seq_len]``
* ground-truth kept as strings (not forced to float)

Runs in a single-process AgentLoopActor (0 GPU, no torch.distributed),
so it does **not** use ``BroadcastUtils`` or ``is_mp_and_cp_head``.
"""

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional

import torch

from tasks.math_rl_v4.bt_reward_dapo import math_accuracy_reward, math_format_reward

from gpatch_v4.reward.base_external_reward import BaseExternalReward
from gpatch_v4.utils.training_utils import list_of_tensor_to_list

_EVAL_DELAY_RANDOM_MIN_S = 50.0
_EVAL_DELAY_RANDOM_MAX_S = 200.0


def _resolve_gt_answer(label: Any) -> str:
    """Normalize one gt_label entry to a ground-truth answer string."""
    if isinstance(label, str):
        try:
            parsed = json.loads(label)
            if isinstance(parsed, dict) and "answer" in parsed:
                return str(parsed["answer"])
            return label
        except (json.JSONDecodeError, TypeError):
            return label
    if isinstance(label, torch.Tensor):
        val = label.item()
        return str(int(val)) if val == int(val) else str(val)
    return str(label)


class DapoMathRuleExternalReward(BaseExternalReward):
    """DAPO-Math rule reward (accuracy + format) for AgentLoopActor.

    Parameters
    ----------
    config : RlConfig or None
        Supplies the optional evaluation reward delay.
    tokenizer : PreTrainedTokenizer or None
        Actor tokenizer used to decode generated tokens.
    """

    supports_concurrent_calls = True

    def __init__(self, config=None, tokenizer=None):
        super().__init__(config=config, tokenizer=tokenizer)
        assert tokenizer is not None, "DapoMathRuleExternalReward requires a tokenizer"

    async def calc_external_reward(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
        _started_event: Optional[asyncio.Event] = None,
    ) -> List[Dict[str, Any]]:
        """Compute per-batch reward updates.

        Returns one update dict per input batch; the caller is
        responsible for merging each update back into its rollout batch
        (e.g. ``rb.update(updates[i])``).
        """
        if _started_event is not None:
            _started_event.set()

        delay_s = 0.0
        eval_delay_s = self.config.external_reward.eval_delay_s if self.config else 0.0
        if is_eval and eval_delay_s != 0:
            if eval_delay_s == -1:
                delay_s = random.uniform(
                    _EVAL_DELAY_RANDOM_MIN_S, _EVAL_DELAY_RANDOM_MAX_S
                )
                logging.info("[external_reward] eval random delay=%.3fs", delay_s)
            elif eval_delay_s > 0:
                delay_s = float(eval_delay_s)
            else:
                raise ValueError(
                    f"external_reward.eval_delay_s must be >= 0 or -1, got {eval_delay_s}"
                )
        score_task = asyncio.create_task(
            asyncio.to_thread(self._compute_updates, rollout_batches)
        )
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            return await score_task
        except BaseException:
            score_task.cancel()
            await asyncio.gather(score_task, return_exceptions=True)
            raise

    def _compute_updates(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> List[Dict[str, Any]]:
        return [self._compute_update(batch) for batch in rollout_batches]

    def _compute_update(self, batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Return a dict of fields to merge into ``batch``."""
        tokens_list = batch["tokens"]
        seq_lens = batch["sequence_lengths"]
        prompt_lens = batch["prompt_lengths"]
        gt_labels = batch.get("gt_label", batch.get("labels"))
        assert gt_labels is not None, "gt_label or labels must be present"

        tokens_cpu = list_of_tensor_to_list(tokens_list, False)
        seq_len_cpu = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in seq_lens]
        prompt_len_cpu = [
            p.item() if isinstance(p, torch.Tensor) else int(p) for p in prompt_lens
        ]
        assert len(tokens_cpu) == len(seq_len_cpu) == len(prompt_len_cpu) == len(gt_labels)

        # Decode only the response portion (after prompt), matching bt_reward_dapo.
        resp_token_ids = [
            tokens_cpu[i][prompt_len_cpu[i]:seq_len_cpu[i]] for i in range(len(tokens_cpu))
        ]
        resp_strs = self.tokenizer.batch_decode(resp_token_ids, skip_special_tokens=True)
        gt_answers = [_resolve_gt_answer(label) for label in gt_labels]

        acc_reward = [
            math_accuracy_reward(resp, gt) for resp, gt in zip(resp_strs, gt_answers)
        ]
        fmt_reward = [math_format_reward(resp) for resp in resp_strs]

        acc_t = torch.tensor(acc_reward, dtype=torch.float32).view(-1, 1)
        fmt_t = torch.tensor(fmt_reward, dtype=torch.float32).view(-1, 1)
        combined = acc_t + fmt_t
        n = combined.size(0)

        external_reward = [combined[i] for i in range(n)]
        acc_reward_list = [acc_t[i] for i in range(n)]
        fmt_reward_list = [fmt_t[i] for i in range(n)]

        upd: Dict[str, Any] = {
            "external_reward": external_reward,
            "acc_reward": acc_reward_list,
            "fmt_reward": fmt_reward_list,
        }
        prev_rewards = batch.get("rewards")
        if prev_rewards is None:
            upd["rewards"] = external_reward
        else:
            upd["rewards"] = [prev_rewards[i] + external_reward[i] for i in range(n)]
        return upd
