#!/usr/bin/env python3
"""OpenAI-compatible client for the hybrid RAG REST endpoint."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from openai import OpenAI

from common.config import load_settings
from common.logging import get_logger

LOGGER = get_logger(__name__)


def _resolve_base_url() -> str:
    """Resolve the OpenAI-compatible base URL for the RAG server."""

    env_url = os.getenv("RAG_SERVER_URL")
    if env_url:
        return env_url.rstrip("/")

    settings = load_settings()
    host = settings.server_host
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{settings.server_port}/v1"


def _resolve_api_key() -> str:
    """Resolve the API key for the RAG server client."""

    return os.getenv("RAG_SERVER_API_KEY", "local-rag")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the RAG client."""

    parser = argparse.ArgumentParser(description="Query the hybrid RAG OpenAI-compatible endpoint.")
    parser.add_argument("--question", required=True, help="Question to ask the RAG server.")
    parser.add_argument("--model", default=None, help="Override the model name in the request payload.")
    parser.add_argument("--base-url", default=None, help="Override the API base URL for the RAG server.")
    parser.add_argument("--api-key", default=None, help="Override the API key for the RAG server.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Send a chat completion request to the RAG server and print the answer."""

    args = parse_args(argv)
    base_url = args.base_url or _resolve_base_url()
    api_key = args.api_key or _resolve_api_key()
    settings = load_settings()
    model = args.model or settings.llm_server_model

    LOGGER.info("Sending request to %s", base_url)
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.question}],
    )

    content = response.choices[0].message.content if response.choices else ""
    print(content)


if __name__ == "__main__":
    main()
