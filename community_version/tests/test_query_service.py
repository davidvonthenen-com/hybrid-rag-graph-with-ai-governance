"""Tests for the Hybrid RAG service helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

COMMUNITY_VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMUNITY_VERSION_ROOT))

import query  # noqa: E402


def test_normalize_messages_valid() -> None:
    payload = {"messages": [{"role": "user", "content": "Hello"}]}

    messages = query._normalize_messages(payload)

    assert messages == [{"role": "user", "content": "Hello"}]


def test_normalize_messages_invalid() -> None:
    with pytest.raises(ValueError):
        query._normalize_messages({"messages": []})


def test_extract_question_from_messages_uses_last_user() -> None:
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Second"},
    ]

    question = query._extract_question_from_messages(messages)

    assert question == "Second"


def test_extract_question_from_messages_falls_back_to_last_message() -> None:
    messages = [
        {"role": "assistant", "content": "System note"},
        {"role": "assistant", "content": "Last message"},
    ]

    question = query._extract_question_from_messages(messages)

    assert question == "Last message"


def test_build_request_args_overrides() -> None:
    base_args = query.parse_args([])

    payload = {"temperature": 0.5, "top_k": 12, "vec_filter": "none"}
    merged = query._build_request_args(base_args, payload)

    assert merged.temperature == 0.5
    assert merged.top_k == 12
    assert merged.vec_filter == "none"
