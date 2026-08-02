# TiSASRec via RecBole CE (AC-TSR official fork)

Train **RecBole TiSASRec with full-catalog CE** on EmoRecAgent processed splits.
Vendored RecBole comes from [AIM-SE/AC-TSR](https://github.com/AIM-SE/AC-TSR) (paper code).

This is the AC-TSR Table 1 TiSASRec setup — **not** `baseline/TiSASRec.pytorch` (BCE)
and **not** EmoRecAgent Stage-1.

## Prerequisites

```bash
# ERA env (or equivalent with torch+cuda)
pip install -r baseline/RecBole-TiSASRec/requirements.txt
# Processed Yelp_AC must already exist:
#   python3 scripts/build_dataset.py --config configs/categories/Yelp_AC.yaml
```

## Quick start — Yelp_AC (AC-TSR recipe)

```bash
PYTHONPATH=src \
/home/ai/anaconda3/envs/ERA/bin/python \
  baseline/RecBole-TiSASRec/amazon/run_experiment.py \
  --config baseline/RecBole-TiSASRec/configs/paper_tisasrec_yelp_ac.yaml \
  --out results/Yelp_AC/recbole_tisasrec.json \
  --log-file logs/Yelp_AC/recbole_tisasrec.log
```

Shared settings (`configs/config_t.yaml` + `TiSASRec.yaml`):

| Parameter | Value |
|-----------|-------|
| Loss | CE (full catalog) |
| Optimizer | Adam, lr 1e-4, weight_decay 0 |
| Batch / epochs | 256 / 200 (early stop patience 10 on Recall@10) |
| Max len / time_span | 50 / 256 |
| Arch | n_layers=2, n_heads=2, hidden=64, inner=256 |
| Eval | full ranking, LOO (`LS: valid_and_test`) |

## Outputs

- Checkpoint under `baseline/RecBole-TiSASRec/checkpoints/`
- Metrics JSON: `results/Yelp_AC/recbole_tisasrec.json`
  - `means_per_user.recall@10/20`, `ndcg@10/20` (Hit@K copied to hr@K)
  - Raw RecBole valid/test blocks under `recbole`

## AC-TSR Table 1 targets (TiSASRec, Yelp)

Recall@10=0.0618 · Recall@20=0.0909 · NDCG@10=0.0387 · NDCG@20=0.0460

Paper best `{layers,heads,hidden,inner}` is grid-only; defaults above match RecBole / AC-TSR `TiSASRec.yaml`.
