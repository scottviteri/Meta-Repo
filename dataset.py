from typing import List, Dict
import torch
from torch.utils.data import Dataset


class TokenRewardDataset(Dataset):
    """Simple dataset that holds tokenized input sequences plus per-token rewards.

    Each example is a dict with:
    - `input_ids`: List[int] of token ids (length L)
    - `rewards`: List[float] of per-token rewards (length L)
    """

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
                "rewards": torch.tensor(ex["rewards"], dtype=torch.float)}


def collate_fn(batch, pad_token_id=0):
    """Pad input_ids and rewards to max length in batch."""
    input_ids = [b["input_ids"] for b in batch]
    rewards = [b["rewards"] for b in batch]

    lengths = [x.size(0) for x in input_ids]
    max_len = max(lengths)

    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_rewards = torch.zeros((len(batch), max_len), dtype=torch.float)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, r) in enumerate(zip(input_ids, rewards)):
        L = ids.size(0)
        padded_input_ids[i, :L] = ids
        padded_rewards[i, :L] = r
        attention_mask[i, :L] = 1

    return {"input_ids": padded_input_ids, "rewards": padded_rewards, "attention_mask": attention_mask}
