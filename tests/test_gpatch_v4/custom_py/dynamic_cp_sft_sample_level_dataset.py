"""Deterministic unequal-length text-only SFT data for Qwen3.6 GPU tests."""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class DynamicCpSftSampleLevelDataset(Dataset):
    _LENGTHS = (256, 512, 768, 1024, 320, 576, 832, 1088)

    def __init__(self, config, tokenizer):
        self.config = config
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = config.training.seq_length + 1

    def __len__(self):
        return len(self._LENGTHS)

    def __getitem__(self, index):
        seq_len = min(self._LENGTHS[index], self.max_seq_len)
        generator = torch.Generator().manual_seed(20260904 + index)
        tokens = torch.randint(
            0,
            self.vocab_size,
            (seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        labels = tokens.clone()
        # Unequal answer lengths are required to distinguish sequence mean from
        # token mean. Keep at least one supervised next-token target.
        prompt_len = max(1, seq_len - (32 + index * 29))
        labels[:prompt_len] = -100
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long)
        position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1).contiguous()
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "position_ids": position_ids,
            "image_input_mask": torch.zeros(seq_len, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
        }


def _collate(examples):
    return {
        key: [example[key] for example in examples]
        for key in examples[0]
    }


def get_dataset_and_dataloader(config=None, tokenizer=None, dp_rank=0, dp_size=1):
    dataset = DynamicCpSftSampleLevelDataset(config, tokenizer)
    sampler = DistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=False,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.training.train_mbs,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    return {
        "train_dataset": dataset,
        "train_sampler": sampler,
        "train_dataloader": dataloader,
    }
