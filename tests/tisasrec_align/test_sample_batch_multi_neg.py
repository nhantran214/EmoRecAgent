"""Tests for multi-negative sample_batch."""

from __future__ import annotations

from emorecagent.tisasrec_align.sequence_data import sample_batch


def test_sample_batch_multi_neg_shape():
    user_train = {
        1: [(1, 1), (2, 2), (3, 3)],
        2: [(4, 1), (5, 2), (6, 3)],
    }
    _users, seqs, time_mats, poss, negs = sample_batch(
        user_train,
        item_num=6,
        maxlen=5,
        time_span=4,
        batch_size=4,
        num_negatives=3,
    )
    assert seqs.shape[0] == 4
    assert negs.shape == (4, 5, 3)
    assert (negs >= 0).all()


def test_sample_batch_single_neg_shape():
    user_train = {1: [(1, 1), (2, 2), (3, 3)]}
    _users, _seqs, _time_mats, _poss, negs = sample_batch(
        user_train,
        item_num=6,
        maxlen=5,
        time_span=4,
        batch_size=2,
        num_negatives=1,
    )
    assert negs.shape == (2, 5)
