"""Configuration loading utilities for the RAG service."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv



def _get_str(name: str, default_val: str) -> str:
    """Return an environment variable or ``default_val`` if unset."""

    value = os.getenv(name)
    return default_val if value is None else value


def _get_int(name: str, default_val: int) -> int:
    """Safely coerce an environment variable to ``int`` with fallback."""

    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return int(value)
    except ValueError:
        return default_val


def _get_bool(name: str, default_val: bool) -> bool:
    """Return a boolean flag parsed from common truthy strings."""

    value = os.getenv(name)
    if value is None:
        return default_val
    return value.lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default_val: float) -> float:
    """Safely coerce an environment variable to ``float`` with fallback."""

    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return float(value)
    except ValueError:
        return default_val


def _default_llama_n_gpu_layers() -> int:
    """Return the default GPU layer offload value for the current platform."""

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    return Settings.llama_n_gpu_layers


@dataclass(frozen=True)
class Settings:
    """Runtime configuration parameters for the application."""

    # ---------------------------------------------------------------------
    # Neo4j (graph grounding)
    # ---------------------------------------------------------------------

    # LONG store (vetted / read-mostly)
    neo4j_long_uri: str = "bolt://127.0.0.1:7688"
    neo4j_long_user: str = "neo4j"
    neo4j_long_password: str = "neo4jneo4j1"
    neo4j_long_database: str = "neo4j"

    # HOT store (unvetted / user data). Default to the same instance.
    neo4j_hot_uri: str = "bolt://127.0.0.1:7689"
    neo4j_hot_user: str = "neo4j"
    neo4j_hot_password: str = "neo4jneo4j2"
    neo4j_hot_database: str = "neo4j"

    # OpenSearch (vector store)
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_vector_index: str = "bbc-vector-chunks"
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = False

    # Retrieval / ranking parameters
    # search_size: int = 10                      # candidate docs per store
    # ranking_alpha: float = 0.5                 # per-store OpenSearch score threshold
    search_preference: str = "governance-audit-v1"
    os_explain: bool = False
    os_profile: bool = False

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # use thenlper/gte-small or Qwen/Qwen3-Embedding-0.6B

    # Llama.cpp
    # llama_model_path: str = "neural-chat-7b-v3-3.Q4_K_M.gguf"
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    # llama_ctx: int = 32768                  # "neural-chat = 32768, Qwen = 65536/101000
    llama_ctx: int = 65536                  # "neural-chat = 32768, Qwen = 65536/101000
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20             # -1 offloads all layers when GPU backend is available
    llama_n_batch: int = 256                 # prompt processing batch
    llama_n_ubatch: Optional[int] = 256      # physical micro-batch; None to let llama.cpp choose
    llama_low_vram: bool = True              # reduce Metal VRAM usage

    # External LLM (OpenAI-compatible endpoint). Used when USE_EXTERNAL_AI=true.
    llm_server_url: str = "http://127.0.0.1:8001/v1"
    llm_server_api_key: str = "local-llm"
    llm_server_model: str = "local-llm"
    external_base_url: str = "https://inference.do-ai.run/v1/chat/completions"
    external_model: str = "llama3-8b-instruct"

    # Named entity recognition service
    ner_url: str = "http://127.0.0.1:8000/ner"
    ner_timeout_secs: float = 5.0

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000


def load_settings() -> Settings:
    """Load settings from environment variables (with .env support)."""
    load_dotenv()

    external_ai = os.getenv("USE_EXTERNAL_AI", "false").lower() in ("1", "true", "yes", "on")

    llm_server_url=os.getenv("LLM_SERVER_URL", Settings.llm_server_url)
    if external_ai:
        llm_server_url = _get_str("EXTERNAL_LLM_URL", Settings.external_base_url)
    llm_server_api_key=os.getenv("LLM_SERVER_API_KEY", Settings.external_base_url)
    if external_ai:
        llm_server_api_key = _get_str("EXTERNAL_LLM_API_KEY", "")
    llm_server_model=os.getenv("LLM_SERVER_MODEL",  Settings.llm_server_model)
    if external_ai:
        llm_server_model=os.getenv("EXTERNAL_LLM_MODEL", Settings.external_model)
    llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx)
    if external_ai:
        llama_ctx=os.getenv("EXTERNAL_LLM_MAX_TOKENS", 131072)

    return Settings(
        # Neo4j (graph grounding)
        neo4j_long_uri=os.getenv("NEO4J_LONG_URI", Settings.neo4j_long_uri),
        neo4j_long_user=os.getenv("NEO4J_LONG_USER", Settings.neo4j_long_user),
        neo4j_long_password=os.getenv("NEO4J_LONG_PASSWORD", Settings.neo4j_long_password),
        neo4j_long_database=os.getenv("NEO4J_LONG_DATABASE", Settings.neo4j_long_database),

        neo4j_hot_uri=os.getenv("NEO4J_HOT_URI", Settings.neo4j_hot_uri),
        neo4j_hot_user=os.getenv("NEO4J_HOT_USER", Settings.neo4j_hot_user),
        neo4j_hot_password=os.getenv("NEO4J_HOT_PASSWORD", Settings.neo4j_hot_password),
        neo4j_hot_database=os.getenv("NEO4J_HOT_DATABASE", Settings.neo4j_hot_database),

        # Vector OpenSearch
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_vector_index=os.getenv("OPENSEARCH_VECTOR_INDEX", Settings.opensearch_vector_index),
        opensearch_user=os.getenv("OPENSEARCH_USER", Settings.opensearch_user),
        opensearch_password=os.getenv("OPENSEARCH_PASS", Settings.opensearch_password),
        opensearch_ssl=_get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl),

        # Embeddings
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),

        # LLaMA
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=llama_ctx,
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", _default_llama_n_gpu_layers()),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),

        # External LLM (OpenAI-compatible endpoint)
        llm_server_url=llm_server_url,
        llm_server_api_key=llm_server_api_key,
        llm_server_model=llm_server_model,

        # Retrieval / ranking
        # search_size=_get_int("SEARCH_SIZE", Settings.search_size),
        # ranking_alpha=_get_float("RANKING_ALPHA", Settings.ranking_alpha),
        search_preference=os.getenv(
            "PREFERENCE_TOKEN", Settings.search_preference
        ),
        os_explain=_get_bool("OS_EXPLAIN", Settings.os_explain),
        os_profile=_get_bool("OS_PROFILE", Settings.os_profile),

        # NER
        ner_url=os.getenv("NER_URL", Settings.ner_url),
        ner_timeout_secs=_get_float("NER_TIMEOUT_SECS", Settings.ner_timeout_secs),

        # Server
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
    )


__all__ = ["Settings", "load_settings"]
