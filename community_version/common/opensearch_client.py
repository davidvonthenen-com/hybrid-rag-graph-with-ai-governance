"""OpenSearch client utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError

from .config import Settings, load_settings, _get_str, _get_bool
from .logging import get_logger


LOGGER = get_logger(__name__)


VECTOR_FIELD = _get_str("VECTOR_FIELD", "embedding")
VECTOR_USE_RESCORE = _get_bool("VECTOR_USE_RESCORE", False)


# ---------------------------------------------------------------------------
# Low-level client construction
# ---------------------------------------------------------------------------

class MyOpenSearch(OpenSearch):
    """Custom OpenSearch client subclass for future extensions."""

    settings: Settings

    def __init__(self, *args: Any, settings: Settings, **kwargs: Any) -> None:
        """Initialize the custom OpenSearch client."""
        super().__init__(*args, **kwargs)
        self.settings = settings


def _build_client(
    host: str,
    port: int,
    *,
    user: Optional[str] = None,
    password: Optional[str] = None,
    ssl: bool = False,
    settings: Settings,
) -> MyOpenSearch:
    """Internal helper to construct an OpenSearch client."""
    http_auth = (user, password) if user and password else None
    scheme = "https" if ssl else "http"

    return MyOpenSearch(
        hosts=[{"host": host, "port": port, "scheme": scheme}],
        http_compress=True,
        http_auth=http_auth,
        use_ssl=ssl,
        verify_certs=ssl,
        ssl_assert_hostname=False if not ssl else None,
        ssl_show_warn=ssl,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
        settings=settings,
    )


def create_vector_client(settings: Optional[Settings] = None) -> Tuple[MyOpenSearch, str]:
    """Create an OpenSearch client for the primary (vector) store."""
    if not settings:
        settings = load_settings()
    LOGGER.info(
        "Connecting to OpenSearch (vector) at %s:%s (index=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_vector_index
    )
    client = _build_client(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user or None,
        password=settings.opensearch_password or None,
        ssl=bool(settings.opensearch_ssl),
        settings=settings,
    )
    return client, settings.opensearch_vector_index


# ---------------------------------------------------------------------------
# Vector search utilities
# ---------------------------------------------------------------------------

def knn_search_one(
    label: str,
    client: MyOpenSearch,
    index_name: str,
    query_vector: List[float],
    *,
    k: int,
    field: str = VECTOR_FIELD,
    num_candidates: int | None = None,
) -> Dict[str, Any]:
    """
    Run a kNN query against a vector index and return an OpenSearch-like response dict
    with the same observability metadata keys used by `search_one`.
    """
    body: Dict[str, Any] = {
        "size": k,
        "_source": True,  # or restrict to ["path","category","chunk_index","chunk_count","text"]
        "query": {
            "knn": {
                field: {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
        "track_total_hits": True,
    }

    # Optional: Lucene kNN can use oversampling via rescore in some setups.
    if VECTOR_USE_RESCORE and num_candidates and num_candidates > k:
        oversample_factor = float(num_candidates) / float(max(k, 1))
        body["query"]["knn"][field]["rescore"] = {"oversample_factor": oversample_factor}

    try:
        res = client.search(index=index_name, body=body, preference=client.settings.search_preference, request_timeout=10)
    except TransportError as e:
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_error": f"{e.__class__.__name__}: {getattr(e, 'error', str(e))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        }

    res["_store_label"] = label
    res["_index_used"] = index_name
    # Keep a tiny bit of provenance without dumping the full vector to logs.
    res["_query_vector_dim"] = len(query_vector)
    res["_k"] = k
    return res


# ---------------------------------------------------------------------------
# Lexical search + ranking helpers
# ---------------------------------------------------------------------------


HIGHLIGHT = {
    "fields": {"content": {}},
    "fragment_size": 160,
    "number_of_fragments": 2,
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
}

SOURCE_FIELDS = [
    "filepath",
    "category",
    "content",
    "explicit_terms",
    "explicit_terms_text",
    "ingested_at_ms",
    "doc_version",
]


def build_query_opensearch_ranking(question: str, entities: List[str]) -> Dict[str, Any]:
    """
    Build an OpenSearch query for lexical ranking using entities when present.

    Strategy
    --------
    * If no entities:
        - dis_max over a single match on the full question against `content`.
    * If entities exist:
        - dis_max with two bool branches:
          (A) STRICT / AND-style (boost 30.0)
              - terms_set on explicit_terms requiring ALL
              - match on explicit_terms_text with operator='and'
              - multi_match on content/category^0.5 with operator='and'
          (B) OR-style (boost 10.0)
              - terms on explicit_terms
              - match on explicit_terms_text
              - multi_match on content/category^0.5
    """
    if not entities:
        return {
            "dis_max": {
                "tie_breaker": 0.0,
                "queries": [
                    {"match": {"content": {"query": question}}},
                ],
            }
        }

    joined = " ".join(entities)

    strict_bool: Dict[str, Any] = {
        "bool": {
            "should": [
                {
                    "terms_set": {
                        "explicit_terms": {
                            "terms": entities,
                            "minimum_should_match_script": {"source": "params.num_terms"},
                        }
                    }
                },
                {
                    "match": {
                        "explicit_terms_text": {
                            "query": joined,
                            "operator": "and",
                        }
                    }
                },
                {
                    "multi_match": {
                        "query": joined,
                        "fields": ["content^1.0", "category^0.5"],
                        "operator": "and",
                    }
                },
            ],
            "minimum_should_match": 1,
            "boost": 30.0,
        }
    }

    or_bool: Dict[str, Any] = {
        "bool": {
            "should": [
                {"terms": {"explicit_terms": entities}},
                {"match": {"explicit_terms_text": joined}},
                {"multi_match": {"query": joined, "fields": ["content^1.0", "category^0.5"]}},
            ],
            "minimum_should_match": 1,
            "boost": 10.0,
        }
    }

    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [strict_bool, or_bool],
        }
    }


def build_query_external_ranking(question: str, entities: List[str]) -> Dict[str, Any]:
    """
    Build an OpenSearch query optimized for external BM25 re-ranking.

    This mirrors `build_query_opensearch_ranking` but omits the strict AND
    branch and focuses on recall for a downstream ranker.
    """
    if not entities:
        return {
            "dis_max": {
                "tie_breaker": 0.0,
                "queries": [
                    {"match": {"content": {"query": question}}},
                ],
            }
        }

    joined = " ".join(entities)

    or_bool: Dict[str, Any] = {
        "bool": {
            "should": [
                {"terms": {"explicit_terms": entities}},
                {"match": {"explicit_terms_text": joined}},
                {"multi_match": {"query": joined, "fields": ["content^1.0", "category^0.5"]}},
            ],
            "minimum_should_match": 1,
            "boost": 10.0,
        }
    }

    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [or_bool],
        }
    }


def search_one(
    label: str,
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
    settings: Settings,
) -> Dict[str, Any]:
    """Execute a single lexical search with observability toggles."""
    body: Dict[str, Any] = {
        "query": query,
        "_source": SOURCE_FIELDS,
        "highlight": HIGHLIGHT,
        "size": settings.search_size,
        "explain": settings.os_explain,
        "profile": settings.os_profile,
        "track_total_hits": True,
    }
    try:
        res = client.search(
            index=index_name,
            body=body,
            preference=settings.search_preference,
            request_timeout=10,
        )
    except TransportError as exc:
        # Soft-fail: return empty result + diagnostic
        LOGGER.warning(
            "OpenSearch search_one error on %s/%s: %s",
            label,
            index_name,
            getattr(exc, "error", str(exc)),
        )
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_query": query,
            "_error": f"{exc.__class__.__name__}: {getattr(exc, 'error', str(exc))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        }

    res["_store_label"] = label
    res["_index_used"] = index_name
    res["_query"] = query
    return res


def rank_hits(res: Dict[str, Any], *, alpha: float) -> List[Dict[str, Any]]:
    """
    Threshold OpenSearch hits per store.

    Policy: keep any hit whose score >= alpha * top1, evaluated per-store
    to avoid cross-cluster score mixing.
    """
    hits = res.get("hits", {}).get("hits", []) or []
    if not hits:
        return []
    top1 = hits[0]["_score"]
    keep: List[Dict[str, Any]] = [h for h in hits if h["_score"] >= alpha * top1]
    for h in keep:
        h["_store_label"] = res.get("_store_label", "?")
        h["_index_used"] = res.get("_index_used", "?")
    return keep


def combine_hits(
    hits_a: List[Dict[str, Any]],
    hits_b: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine two per-store lists without pretending cross-store scores are comparable.

    Policy: interleave A and B (stable) while preserving each list's order.
    """
    combined: List[Dict[str, Any]] = []
    ia = ib = 0
    while len(combined) < top_k and (ia < len(hits_a) or ib < len(hits_b)):
        if ia < len(hits_a):
            combined.append(hits_a[ia])
            ia += 1
        if len(combined) >= top_k:
            break
        if ib < len(hits_b):
            combined.append(hits_b[ib])
            ib += 1
    return combined


def render_observability_summary(res: Dict[str, Any]) -> str:
    """Return a concise text summary for logging or debugging."""
    err = res.get("_error")
    store = res.get("_store_label", "?")
    idx = res.get("_index_used", "?")
    total = res.get("hits", {}).get("total", {}).get("value", 0)
    if err:
        return f"[SUMMARY] STORE={store} INDEX={idx} ERROR: {err}"
    return f"[SUMMARY] STORE={store} INDEX={idx} TOTAL={total}"


def render_matches(hits: List[Dict[str, Any]]) -> str:
    """Pretty-print matches and highlights for debugging."""
    lines: List[str] = []
    lines.append("\n================= MATCH EXPLANATIONS =================")
    if not hits:
        lines.append("(no documents kept from either store)")
        lines.append("======================================================\n")
        return "\n".join(lines)

    for i, h in enumerate(hits, 1):
        store = h.get("_store_label", "?")
        idx = h.get("_index_used", "?")
        fp = h.get("_source", {}).get("filepath", "<unknown>")
        score = h.get("_score")
        lines.append(f"\n[{i}] STORE={store} INDEX={idx} SCORE={score:.4f}")
        lines.append(f"     DOC={fp}")
        if "highlight" in h and "content" in h["highlight"]:
            frag = h["highlight"]["content"][0]
            lines.append(f"     highlight: {frag}")
        lines.append("")
    lines.append("======================================================\n")
    return "\n".join(lines)


__all__ = [
    "create_vector_client",
    "create_long_client",
    "create_hot_client",
    "knn_search_one",
    "build_query_opensearch_ranking",
    "build_query_external_ranking",
    "search_one",
    "rank_hits",
    "combine_hits",
    "render_observability_summary",
    "render_matches",
]
