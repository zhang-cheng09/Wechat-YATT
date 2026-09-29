# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""HpModule marker class.

Kept in its own module to avoid importing the parent package's heavy
``__init__.py`` (diffusers, etc.) and to prevent circular imports.
"""

from typing import Any, Mapping


class HpModule:
    """Marker base class for hybrid-parallel models.

    Subclasses opt into the alternative construction path in
    ``Fsdp2EngineMixin``: meta-device init -> ``apply_hp`` -> binds
    ``clip_grad_norm_`` and ``load_checkpoint_hp`` /
    ``load_state_dict_hp``, instead of the default ``from_pretrained``
    + FSDP2 wrap flow.
    """
    def configure_debug_diagnostics(self, debug_config: Any, policy_config: Any) -> None:
        return

    def maybe_dump_prepared_microbatch_diagnostics(
        self,
        batch: Mapping[str, Any],
        forward_kwargs: Mapping[str, Any],
        microbatch_index: int,
    ) -> None:
        return

    def finish_step0_diagnostics(self) -> None:
        return
