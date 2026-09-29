import asyncio

import hydra
import torch

# isort: off
import gpatch_v4.core.device  # noqa: F401  # ensure device backend is initialized early
# isort: on

from gpatch_v4.configs.config import EmbeddingOffPolicyDistillConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import OffPolicyDistillTrainer
from gpatch_v4.transfer import close_tq_connector


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: EmbeddingOffPolicyDistillConfig):
    """Entry point for embedding off-policy distillation training."""
    merged_obj = merge_hydra_config(EmbeddingOffPolicyDistillConfig, cfg)

    trainer = OffPolicyDistillTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    try:
        if merged_obj.debug.test_ray_rpc:
            asyncio.run(trainer.test_ray_rpc(merged_obj))
            return
        asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))
    finally:
        close_tq_connector()


if __name__ == "__main__":
    main()
