import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class GPT2WithQ(nn.Module):
    """GPT-2 backbone with two output heads:
    - LM head (logits over vocabulary) for next-token prediction (causal LM)
    - Q head (vocab-sized) predicting expected future return if next token == a

    The Q head produces a scalar Q(s, a) for each possible next-token a.
    In training we typically supervise only the taken (observed) next-token's Q.
    """

    def __init__(self, pretrained_model_name_or_config="gpt2"):
        super().__init__()
        if isinstance(pretrained_model_name_or_config, GPT2Config):
            config = pretrained_model_name_or_config
        else:
            config = GPT2Config.from_pretrained(pretrained_model_name_or_config)

        # Use the core transformer (no LM head) and add our own heads
        self.transformer = GPT2Model(config)
        self.vocab_size = config.vocab_size
        hidden_size = config.hidden_size

        # LM head (tie weights later optionally)
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)

        # Q head: outputs a predicted return for each action (token)
        # This is intentionally vocab-sized so we can query Q(s, a) for any a.
        self.q_head = nn.Linear(hidden_size, self.vocab_size)

        # Optionally initialize lm_head weights from transformer wte
        try:
            # tie weights if transformer has wte
            self.lm_head.weight = self.transformer.wte.weight
        except Exception:
            pass

    def forward(self, input_ids, attention_mask=None, return_dict=False):
        # transformer outputs last_hidden_state (B, L, H)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state

        logits = self.lm_head(hidden)  # (B, L, V)
        q_values = self.q_head(hidden)  # (B, L, V)

        if return_dict:
            return {"logits": logits, "q_values": q_values, **outputs}
        return logits, q_values
