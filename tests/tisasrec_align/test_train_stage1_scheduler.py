"""Tests for LR scheduler stale reset during Stage 1 training."""

from __future__ import annotations

import torch

from emorecagent.tisasrec_align.train_stage1 import apply_lr_scheduler_step


def test_lr_reduction_resets_stale():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    stale = 3
    for metric in [0.1, 0.1, 0.1]:
        stale, reduced, lr = apply_lr_scheduler_step(scheduler, opt, metric, stale)
        if reduced:
            break
    assert reduced is True
    assert stale == 0
    assert lr < 0.01


def test_no_scheduler_leaves_stale():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    stale, reduced, lr = apply_lr_scheduler_step(None, opt, 0.5, 4)
    assert reduced is False
    assert stale == 4
    assert lr == 0.01
