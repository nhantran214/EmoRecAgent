#!/usr/bin/env python3
"""Manual smoke test: structured schemas against a live TGI server."""

from __future__ import annotations

import argparse

from emorecagent.config import load_config
from emorecagent.llm.client import LLMClient
from emorecagent.llm.schemas import (
    HybridAbsaVerdict,
    ReasoningRankingVerdict,
    ReflectionVerdict,
    TripleSet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test TGI structured outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = LLMClient.from_config(cfg)
    schemas = [
        ("TripleSet", TripleSet, "Extract one aspect triple from: great battery life."),
        (
            "HybridAbsaVerdict",
            HybridAbsaVerdict,
            'Validate triples for review: "Smells nice but pricey."',
        ),
        (
            "ReasoningRankingVerdict",
            ReasoningRankingVerdict,
            "Rank item ids [1,2,3] for user who likes lightweight moisturizers.",
        ),
        (
            "ReflectionVerdict",
            ReflectionVerdict,
            "Judge whether ranking [3,1,2] fits user history of skincare items.",
        ),
    ]
    for name, schema, prompt in schemas:
        out = client.invoke_structured(prompt, schema)
        print(f"{name}: ok -> {type(out).__name__}")


if __name__ == "__main__":
    main()
