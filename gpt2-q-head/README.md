# GPT2 with Q-head example

This repo contains a minimal example showing how to add a vocabulary-sized Q head to a GPT-2 style model.

Files:
- `model.py`: `GPT2WithQ` adds a Q head over the vocabulary alongside the LM head.
- `dataset.py`: `TokenRewardDataset` and `collate_fn` for token sequences and per-token rewards.
- `train.py`: small training loop combining causal LM loss and a Q MSE loss on the taken next-token.
- `requirements.txt`: python dependencies.

Quick start (create virtualenv, install dependencies):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 3 --batch_size 8
```

Notes:
- The Q head predicts expected future return conditional on choosing a particular next token.
- The example supervises the Q value for the actually observed next token using a discounted-return target.
- For off-policy / counterfactual supervision you would need additional components (estimators, importance sampling, or model-based rollouts).
