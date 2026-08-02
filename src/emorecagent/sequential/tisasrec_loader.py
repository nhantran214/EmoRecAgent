"""Load vendored TiSASRec weights from baseline/TiSASRec.pytorch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .id_maps import IdMaps
from .seq_utils import TiSASRecArgs

_BASELINE_ROOT = Path(__file__).resolve().parents[3] / "baseline" / "TiSASRec.pytorch"


def _ensure_baseline_on_path() -> None:
    root = str(_BASELINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_tisasrec_model(
    checkpoint_path: Path,
    id_maps: IdMaps,
    model_cfg: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[Any, TiSASRecArgs]:
    """Instantiate TiSASRec and load state dict from checkpoint."""
    _ensure_baseline_on_path()
    from model import TiSASRec  # noqa: WPS433

    dev = torch.device(device)
    args = TiSASRecArgs(model_cfg, str(dev))
    usernum = len(id_maps.user_to_idx)
    itemnum = len(id_maps.item_to_idx)
    model = TiSASRec(usernum, itemnum, itemnum, args).to(dev)
    state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, args
