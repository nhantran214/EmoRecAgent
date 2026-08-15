"""Stage 2 InfoNCE training for alignment MLP."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.types import Interaction
from ..sequential.id_maps import build_id_maps_from_interactions
from .alignment_mlp import AlignmentMLP
from .infonce import infonce_loss
from .text_encoder import HashEncoder, TextEncoderBackend
from .tu_cache import TuCacheRow, cache_key, load_tu_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlignTrainRow:
    user_local: int
    gold_local: int
    T_u: str


def _flush() -> None:
    for handler in logger.handlers:
        handler.flush()
    root = logging.getLogger("emorecagent.tisasrec_align")
    for handler in root.handlers:
        handler.flush()


def _log(msg: str, *args: object) -> None:
    logger.info(msg, *args)
    _flush()


def _build_align_rows(
    train: list[Interaction],
    tu_cache: dict[str, TuCacheRow],
    id_maps,
    *,
    item_to_idx: dict[str, int] | None = None,
) -> list[AlignTrainRow]:
    """One row per train interaction with cached T_u and next-item target."""
    item_index = item_to_idx if item_to_idx is not None else id_maps.item_to_idx
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
            cur = events[i]
            key = cache_key(uid, cur.timestamp)
            cached = tu_cache.get(key)
            if cached is None or not cached.T_u:
                continue
            gold = item_index.get(cur.item)
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
    item_to_idx: dict[str, int] | None = None,
    log_every_batches: int | None = None,
) -> TrainStage2Result:
    random.seed(seed)
    torch.manual_seed(seed)
    _log(
        "stage2 start train_rows_raw=%s tu_cache=%s device=%s "
        "epochs=%s batch_size=%s tau_grid=%s",
        len(train),
        tu_cache_path,
        device,
        epochs,
        batch_size,
        list(tau_grid),
    )
    id_maps = build_id_maps_from_interactions(train, [])
    _log("loading tu_cache…")
    tu_cache = load_tu_cache(tu_cache_path)
    _log("tu_cache keys=%s; building alignment rows…", len(tu_cache))
    rows = _build_align_rows(
        train, tu_cache, id_maps, item_to_idx=item_to_idx
    )
    if not rows:
        raise ValueError(f"no alignment rows from tu cache: {tu_cache_path}")

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    n_batches = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    every = (
        log_every_batches
        if log_every_batches is not None and log_every_batches > 0
        else max(1, n_batches // 10)
    )
    _log(
        "alignment rows=%s train=%s val=%s batches/epoch=%s log_every=%s",
        len(rows),
        len(train_rows),
        len(val_rows),
        n_batches,
        every,
    )

    encoder = text_encoder or HashEncoder()
    use_hash_encoder = isinstance(encoder, HashEncoder)
    e_i = e_i_matrix.to(device)
    item_num = e_i.shape[0] - 1
    embed_dim = int(e_i.shape[-1])
    if hidden_dim != embed_dim:
        _log(
            "hidden_dim=%s != E_I dim=%s — using E_I dim for AlignmentMLP output",
            hidden_dim,
            embed_dim,
        )
        hidden_dim = embed_dim
    # Negatives / seen-items use the same item index space as e_i rows.
    item_index = item_to_idx if item_to_idx is not None else id_maps.item_to_idx
    user_seen: dict[int, set[int]] = {}
    for it in train:
        u_local = id_maps.user_to_idx.get(it.user_id)
        i_local = item_index.get(it.item)
        if u_local is None or i_local is None:
            continue
        user_seen.setdefault(u_local, set()).add(i_local)

    text_dim = 768
    if use_hash_encoder and isinstance(encoder, HashEncoder):
        text_dim = int(encoder.dim)
    mlp = AlignmentMLP(text_dim, hidden_dim, activation=activation)  # type: ignore[arg-type]
    mlp.to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    _log(
        "model ready encoder=%s text_dim=%s out_dim=%s items=%s activation=%s",
        "hash" if use_hash_encoder else "sentence_transformer",
        text_dim,
        hidden_dim,
        item_num,
        activation,
    )

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
        epoch_t0 = time.perf_counter()
        running = 0.0
        n_seen = 0
        _log("epoch %s/%s train start (%s batches)", epoch + 1, epochs, n_batches)
        for step, start in enumerate(range(0, len(train_rows), batch_size), start=1):
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
            loss_f = float(loss.detach().cpu())
            running += loss_f
            n_seen += 1
            if step == 1 or step == n_batches or step % every == 0:
                elapsed = time.perf_counter() - epoch_t0
                rate = step / max(elapsed, 1e-9)
                eta = (n_batches - step) / max(rate, 1e-9)
                _log(
                    "epoch %s/%s step %s/%s (%.0f%%) loss=%.4f avg=%.4f "
                    "rate=%.2f batch/s eta=%.0fs",
                    epoch + 1,
                    epochs,
                    step,
                    n_batches,
                    100.0 * step / n_batches,
                    loss_f,
                    running / n_seen,
                    rate,
                    eta,
                )

        _log("epoch %s/%s validating tau_grid=%s…", epoch + 1, epochs, list(tau_grid))
        for tau in tau_grid:
            val_loss = _epoch_loss(val_rows, tau)
            _log("epoch %s/%s val tau=%.3f loss=%.4f", epoch + 1, epochs, tau, val_loss)
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
                _log("checkpoint saved best_tau=%.3f loss=%.4f -> %s", best_tau, best_loss, ckpt_path)
        _log(
            "stage2 epoch=%s/%s done elapsed=%.0fs val_loss=%.4f best_tau=%.3f",
            epoch + 1,
            epochs,
            time.perf_counter() - epoch_t0,
            best_loss,
            best_tau,
        )

    return TrainStage2Result(
        checkpoint_path=ckpt_path, best_tau=best_tau, best_loss=best_loss
    )
