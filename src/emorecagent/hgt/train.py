"""BPR training loop for HGT link prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .embeddings import EmbeddingStore
from .graph_data import HgtGraphBundle
from .model import GNN, Matcher
from .temporal import assert_rte_edge_time


@dataclass(frozen=True, slots=True)
class TrainResult:
    best_valid_mrr: float
    epochs_run: int
    checkpoint_path: Path
    embeddings_dir: Path



def _evaluate_mrr(
    node_emb: torch.Tensor,
    pairs: list[tuple[int, int]],
    item_off: int,
    n_items: int,
) -> float:
    if not pairs:
        return 0.0
    ranks: list[float] = []
    item_mat = node_emb[item_off : item_off + n_items].detach().cpu().numpy()
    user_mat = node_emb.detach().cpu().numpy()
    for u_local, i_local in pairs[:256]:
        scores = item_mat @ user_mat[u_local]
        order = np.argsort(-scores)
        pos = np.where(order == i_local)[0]
        ranks.append(1.0 / (int(pos[0]) + 1) if len(pos) else 0.0)
    return float(np.mean(ranks))


def train_hgt(
    bundle: HgtGraphBundle,
    *,
    checkpoint_path: str | Path,
    embeddings_dir: str | Path,
    n_hid: int = 256,
    n_heads: int = 8,
    n_layers: int = 2,
    dropout: float = 0.2,
    use_RTE: bool = True,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 1024,
    neg_samples: int = 1,
    early_stop_patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
) -> TrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert_rte_edge_time(bundle.edge_time)

    dev = torch.device(device)
    node_feat = torch.tensor(bundle.node_feature, dtype=torch.float32, device=dev)
    node_type = torch.tensor(bundle.node_type, dtype=torch.long, device=dev)
    edge_index = torch.tensor(bundle.edge_index, dtype=torch.long, device=dev)
    edge_type = torch.tensor(bundle.edge_type, dtype=torch.long, device=dev)
    edge_time = torch.tensor(bundle.edge_time, dtype=torch.long, device=dev)

    in_dim = int(bundle.node_feature.shape[1])
    gnn = GNN(
        in_dim=in_dim,
        n_hid=n_hid,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        use_RTE=use_RTE,
    ).to(dev)
    matcher = Matcher(n_hid).to(dev)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(matcher.parameters()), lr=lr)

    train_pairs = list(bundle.train_pairs)
    n_items = bundle.n_items
    item_off = bundle.item_offset()

    best_mrr = -1.0
    patience = 0
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(embeddings_dir)

    for epoch in range(epochs):
        gnn.train()
        matcher.train()
        matcher.cache = None
        random.shuffle(train_pairs)
        total_loss = 0.0
        n_steps = 0
        node_emb = gnn(node_feat, node_type, edge_time, edge_index, edge_type)

        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start : start + batch_size]
            if not batch:
                continue
            u_locals = [u for u, _ in batch]
            pos_locals = [i for _, i in batch]
            neg_locals = [random.randrange(n_items) for _ in batch]

            u_nodes = torch.tensor(u_locals, dtype=torch.long, device=dev)
            pos_nodes = torch.tensor(
                [item_off + i for i in pos_locals], dtype=torch.long, device=dev
            )
            neg_nodes = torch.tensor(
                [item_off + i for i in neg_locals], dtype=torch.long, device=dev
            )

            u_emb = node_emb[u_nodes]
            pos_emb = node_emb[pos_nodes]
            neg_emb = node_emb[neg_nodes]
            pos_score = matcher(u_emb, pos_emb, pair=True)
            neg_score = matcher(u_emb, neg_emb, pair=True)
            loss = -F.logsigmoid(pos_score - neg_score).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_steps += 1

        gnn.eval()
        matcher.eval()
        with torch.no_grad():
            node_emb = gnn(node_feat, node_type, edge_time, edge_index, edge_type)
        valid_mrr = _evaluate_mrr(
            node_emb,
            bundle.valid_pairs,
            bundle.item_offset(),
            n_items,
        )
        if valid_mrr > best_mrr:
            best_mrr = valid_mrr
            patience = 0
            torch.save(
                {
                    "gnn": gnn.state_dict(),
                    "matcher": matcher.state_dict(),
                    "meta": {
                        "n_hid": n_hid,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "in_dim": in_dim,
                        "dropout": dropout,
                        "use_RTE": use_RTE,
                        "best_valid_mrr": valid_mrr,
                        "epoch": epoch,
                    },
                },
                ckpt_path,
            )
            _export_embeddings(bundle, node_emb.detach().cpu().numpy(), emb_dir, epoch, valid_mrr)
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    if not ckpt_path.exists():
        with torch.no_grad():
            node_emb = gnn(node_feat, node_type, edge_time, edge_index, edge_type)
        torch.save(
            {
                "gnn": gnn.state_dict(),
                "matcher": matcher.state_dict(),
                "meta": {
                    "n_hid": n_hid,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "in_dim": in_dim,
                    "dropout": dropout,
                    "use_RTE": use_RTE,
                    "best_valid_mrr": best_mrr,
                    "epoch": epochs - 1,
                },
            },
            ckpt_path,
        )
        _export_embeddings(bundle, node_emb.detach().cpu().numpy(), emb_dir, epochs - 1, best_mrr)

    return TrainResult(
        best_valid_mrr=best_mrr,
        epochs_run=epoch + 1,
        checkpoint_path=ckpt_path,
        embeddings_dir=emb_dir,
    )


def _export_embeddings(
    bundle: HgtGraphBundle,
    node_emb: np.ndarray,
    emb_dir: Path,
    epoch: int,
    valid_mrr: float,
) -> None:
    u_off = bundle.user_offset()
    i_off = bundle.item_offset()
    a_off = bundle.aspect_offset()
    store = EmbeddingStore(
        user_ids=bundle.user_ids,
        item_ids=bundle.item_ids,
        aspect_ids=bundle.aspect_ids,
        user_embeddings=node_emb[u_off : u_off + bundle.n_users].astype(np.float32),
        item_embeddings=node_emb[i_off : i_off + bundle.n_items].astype(np.float32),
        aspect_embeddings=node_emb[a_off : a_off + len(bundle.aspect_ids)].astype(np.float32),
        meta={"epoch": epoch, "valid_mrr": valid_mrr},
    )
    store.save(emb_dir)


def load_checkpoint_meta(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("meta", {}))
