"""Load Stage 1 TiSASRec + Stage 2 alignment checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch

from ..sequential.id_maps import IdMaps
from .alignment_mlp import AlignmentMLP
from .model import TiSASRecArgs, TiSASRecModel


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {resolved}\n"
            "For fusion mode, run:\n"
            "  make precompute-tu-emorecagent SPLIT=train NO_LLM=1\n"
            "  make train-align-emorecagent USE_HASH_ENCODER=1\n"
            "  make precompute-tu-emorecagent SPLIT=test NO_LLM=1"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class AlignBundle:
    tisasrec: TiSASRecModel
    alignment_mlp: AlignmentMLP | None
    item_ids: list[str]
    e_i_matrix: torch.Tensor
    args: TiSASRecArgs
    tau: float
    text_encoder_dim: int = 768


def load_stage1(
    checkpoint_path: str | Path,
    e_i_matrix_path: str | Path,
    device: torch.device,
) -> tuple[TiSASRecModel, list[str], torch.Tensor, TiSASRecArgs]:
    ckpt = _require_file(checkpoint_path, "Stage 1 checkpoint")
    e_i_path = _require_file(e_i_matrix_path, "E_I matrix")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload["meta"])
    # Drop unknown / legacy keys so older checkpoints still load.
    known = {f.name for f in fields(TiSASRecArgs)}
    raw_args = {k: v for k, v in dict(meta["args"]).items() if k in known}
    args = TiSASRecArgs(**raw_args)
    item_num = int(meta["item_num"])
    model = TiSASRecModel(item_num, args)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    e_i = torch.load(e_i_path, map_location=device, weights_only=True)
    item_ids = list(meta["item_ids"])
    return model, item_ids, e_i, args


def load_stage1_id_maps(checkpoint_path: str | Path) -> IdMaps:
    ckpt = _require_file(checkpoint_path, "Stage 1 checkpoint")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = dict(dict(payload["meta"]).get("id_maps") or {})
    if not raw:
        raise ValueError(f"checkpoint missing id_maps meta: {ckpt}")
    return IdMaps(
        user_to_idx={str(k): int(v) for k, v in raw["user_to_idx"].items()},
        item_to_idx={str(k): int(v) for k, v in raw["item_to_idx"].items()},
    )


def load_alignment_mlp(
    path: str | Path,
    *,
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
    activation: str = "elu",
) -> tuple[AlignmentMLP, float]:
    ckpt = _require_file(path, "Stage 2 alignment checkpoint")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    tau = float(meta.get("tau", 0.07))
    act = str(meta.get("activation", activation))
    mlp = AlignmentMLP(input_dim, hidden_dim, activation=act)  # type: ignore[arg-type]
    mlp.load_state_dict(payload["model"])
    mlp.to(device)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad = False
    return mlp, tau


def load_align_bundle(
    *,
    stage1_ckpt: str | Path,
    e_i_path: str | Path,
    device: torch.device,
    text_encoder_dim: int = 768,
    alignment_ckpt: str | Path | None = None,
) -> AlignBundle:
    tisasrec, item_ids, e_i, args = load_stage1(stage1_ckpt, e_i_path, device)
    mlp: AlignmentMLP | None = None
    tau = 0.07
    if alignment_ckpt is not None:
        mlp, tau = load_alignment_mlp(
            alignment_ckpt,
            input_dim=text_encoder_dim,
            hidden_dim=args.hidden_units,
            device=device,
        )
    return AlignBundle(
        tisasrec=tisasrec,
        alignment_mlp=mlp,
        item_ids=item_ids,
        e_i_matrix=e_i,
        args=args,
        tau=tau,
        text_encoder_dim=text_encoder_dim,
    )
