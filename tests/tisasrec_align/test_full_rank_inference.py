"""Inference-mode tensor must not break alignment MLP scoring."""

from __future__ import annotations

import torch

from emorecagent.tisasrec_align.alignment_mlp import AlignmentMLP


def test_alignment_mlp_accepts_inference_mode_input_under_no_grad():
    mlp = AlignmentMLP(768, 64)
    mlp.eval()
    with torch.inference_mode():
        t_u = torch.randn(1, 768)
    with torch.no_grad():
        out = mlp(t_u)
    assert out.shape == (1, 64)
