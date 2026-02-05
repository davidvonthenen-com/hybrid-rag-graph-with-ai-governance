#!/usr/bin/env python3
"""Entrypoint for the Hybrid RAG OpenAI-compatible REST agent service."""

from __future__ import annotations

from typing import Sequence

from query import parse_args, run_service


def main(argv: Sequence[str] | None = None) -> None:
    """Start the Hybrid RAG agent service."""

    args = parse_args(argv)
    args.service = True
    run_service(args)


if __name__ == "__main__":
    main()
