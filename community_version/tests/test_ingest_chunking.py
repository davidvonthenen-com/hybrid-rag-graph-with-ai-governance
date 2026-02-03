from pathlib import Path
import sys

import pytest

COMMUNITY_VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMUNITY_VERSION_ROOT))

from ingest import build_graph_chunk_texts  # noqa: E402


def test_build_graph_chunk_texts_fixed() -> None:
    paragraphs = ["abcdefghij"]

    chunks = build_graph_chunk_texts(
        paragraphs,
        strategy="fixed",
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks == ["abcd", "defg", "ghij"]


def test_build_graph_chunk_texts_paragraph() -> None:
    paragraphs = ["abc", "def"]

    chunks = build_graph_chunk_texts(
        paragraphs,
        strategy="paragraph",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks == ["abc", "def"]


def test_build_graph_chunk_texts_invalid_strategy() -> None:
    with pytest.raises(ValueError):
        build_graph_chunk_texts(
            ["abc"],
            strategy="unknown",
            chunk_size=10,
            chunk_overlap=2,
        )
