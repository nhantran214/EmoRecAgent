"""Stage 2 InfoNCE training for alignment MLP."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.types import Interaction
from ..sequential.id_maps import build_id_maps_from_interactions
from .alignment_mlp import AlignmentMLP
from .infonce import infonce_loss
from .sequence_data import build_train_pairs, build_user_sequences
from .text_encoder import HashEncoder, SentenceTransformerEncoder, TextEncoderBackend
from .train_stage1 import load_train_valid_from_config
from .tu_cache import TuCacheRow, cache_key, load_tu_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlignTrainRow:
    user_local: int
    gold_local: int
    T_u: str


class AlignDataset(Dataset):
    def __init__(self, rows: list[AlignTrainRow]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> AlignTrainRow:
        return self._rows[idx]


def _build_align_rows(
    train: list[Interaction],
    tu_cache: dict[str, TuCacheRow],
    id_maps,
) -> list[AlignTrainRow]:
    """One row per train interaction with cached T_u and next-item target."""
    per_user: dict[str, list[Interaction]] = {}
    for it in train:
        per_user.setdefault(it.user_id, []).append(it)
    rows: list[AlignTrainRow] = []
    for uid, events in per_user.items():
        events = sorted(events, key=lambda x: (x.timestamp, x.item))
        u_local = id_maps.user_to_idx.get(uid)
        if u_local is None:
            continue
        for i in range(1, len(events)):
            prev = events[i - 1]
            cur = events[i]
            key = cache_key(uid, cur.timestamp)
            cached = tu_cache.get(key)
            if cached is None or not cached.T_u:
                continue
            gold = id_maps.item_to_idx.get(cur.item)
            if gold is None:
                continue
            rows.append(AlignTrainRow(u_local, gold, cached.T_u))
    return rows


@dataclass(frozen=True, slots=True)
class TrainStage2Result:
    checkpoint_path: Path
    best_tau: float
    best_loss: float


def train_stage2(
    *,
    train: list[Interaction],
    tu_cache_path: str | Path,
    e_i_matrix: torch.Tensor,
    hidden_dim: int,
    alignment_ckpt_path: str | Path,
    device: torch.device,
    text_encoder: TextEncoderBackend | None = None,
    tau_grid: tuple[float, ...] = (0.05, 0.07, 0.1),
    epochs: int = 10,
    batch_size: int = 64,
    num_random_neg: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42,
    activation: str = "elu",
) -> TrainStage2Result:
    random.seed(seed)
    torch.manual_seed(seed)
    id_maps = build_id_maps_from_interactions(train, [])
    tu_cache = load_tu_cache(tu_cache_path)
    rows = _build_align_rows(train, tu_cache, id_maps)
    if not rows:
        raise ValueError(f"no alignment rows from tu cache: {tu_cache_path}")

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    encoder = text_encoder or HashEncoder()
    use_hash_encoder = isinstance(encoder, HashEncoder)
    e_i = e_i_matrix.to(device)
    item_num = e_i.shape[0] - 1
    train_pairs = build_train_pairs(train, id_maps)
    user_seen: dict[int, set[int]] = {}
    for u, i in train_pairs:
        user_seen.setdefault(u, set()).add(i)

    mlp = AlignmentMLP(768, hidden_dim, activation=activation)  # type: ignore[arg-type]
    mlp.to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    best_tau = tau_grid[0]
    best_loss = float("inf")
    ckpt_path = Path(alignment_ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _epoch_loss(subset: list[AlignTrainRow], tau: float) -> float:
        mlp.eval()
        losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(subset), batch_size):
                batch = subset[start : start + batch_size]
                texts = [r.T_u for r in batch]
                t_u = encoder.encode(texts, device=device)
                p_u = mlp(t_u)
                pos = torch.stack(
                    [e_i[r.gold_local] for r in batch], dim=0
                )
                negs = []
                for r in batch:
                    forbidden = user_seen.get(r.user_local, set())
                    cand = [j for j in range(1, item_num + 1) if j not in forbidden]
                    sample = random.sample(
                        cand, min(num_random_neg, len(cand))
                    )
                    negs.append(torch.stack([e_i[j] for j in sample], dim=0))
                neg_t = torch.stack(negs, dim=0)
                losses.append(
                    float(infonce_loss(p_u, pos, neg_t, temperature=tau).cpu())
                )
        return sum(losses) / len(losses) if losses else 0.0

    for epoch in range(epochs):
        mlp.train()
        random.shuffle(train_rows)
        for start in range(0, len(train_rows), batch_size):
            batch = train_rows[start : start + batch_size]
            texts = [r.T_u for r in batch]
            t_u = encoder.encode(texts, device=device)
            p_u = mlp(t_u)
            pos = torch.stack([e_i[r.gold_local] for r in batch], dim=0)
            negs = []
            for r in batch:
                forbidden = user_seen.get(r.user_local, set())
                cand = [j for j in range(1, item_num + 1) if j not in forbidden]
                sample = random.sample(cand, min(num_random_neg, len(cand)))
                negs.append(torch.stack([e_i[j] for j in sample], dim=0))
            neg_t = torch.stack(negs, dim=0)
            loss = infonce_loss(p_u, pos, neg_t, temperature=best_tau)
            opt.zero_grad()
            loss.backward()
            opt.step()

        for tau in tau_grid:
            val_loss = _epoch_loss(val_rows, tau)
            if val_loss < best_loss:
                best_loss = val_loss
                best_tau = tau
                torch.save(
                    {
                        "model": mlp.state_dict(),
                        "meta": {
                            "tau": best_tau,
                            "activation": activation,
                            "hidden_dim": hidden_dim,
                            "use_hash_encoder": use_hash_encoder,
                        },
                    },
                    ckpt_path,
                )
        logger.info(
            "stage2 epoch=%s val_loss=%.4f best_tau=%.3f",
            epoch + 1,
            best_loss,
            best_tau,
        )

    return TrainStage2Result(
        checkpoint_path=ckpt_path, best_tau=best_tau, best_loss=best_loss
    )
