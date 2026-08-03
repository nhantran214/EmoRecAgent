"""CPU smoke test for HGT BPR training."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from emorecagent.hgt.graph_data import HgtGraphBundle
from emorecagent.hgt.train import train_hgt


def _toy_bundle() -> HgtGraphBundle:
    n_u, n_i, n_a = 2, 3, 1
    n_nodes = n_u + n_i + n_a
    feats = np.random.RandomState(0).randn(n_nodes, 8).astype(np.float32)
    node_type = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    edge_index = np.array(
        [[0, 1, 0, 1], [2, 3, 3, 4]],
        dtype=np.int64,
    )
    edge_type = np.array([0, 0, 0, 0], dtype=np.int64)
    edge_time = np.array([1, 2, 3, 4], dtype=np.int64)
    return HgtGraphBundle(
        node_feature=feats,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_time=edge_time,
        user_ids=["u0", "u1"],
        item_ids=["i0", "i1", "i2"],
        aspect_ids=["scent"],
        train_pairs=[(0, 0), (1, 1), (0, 2), (1, 2)],
        valid_pairs=[(0, 1)],
        aspect_vocab={"aspects": ["scent"], "other_id": 0},
        meta={"feature_dim": 8},
    )


def test_train_one_epoch_exports_checkpoint(tmp_path):
    bundle = _toy_bundle()
    result = train_hgt(
        bundle,
        checkpoint_path=tmp_path / "ckpt.pt",
        embeddings_dir=tmp_path / "emb",
        n_hid=16,
        n_heads=4,
        n_layers=1,
        epochs=1,
        batch_size=2,
        early_stop_patience=1,
        device="cpu",
        seed=0,
    )
    assert result.checkpoint_path.exists()
    assert (tmp_path / "emb" / "user_embeddings.npy").exists()
