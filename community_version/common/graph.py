"""Neo4j-based graph grounding retrieval.

This module replaces the BM25 truth-grounding channel with a Knowledge Graph.

Schema (labels / relationships)
------------------------------
We model:

* (:Document {store, path, category, ...})
* (:Chunk    {store, doc_path, chunk_index, chunk_count, text, ...})
* (:Entity   {name})

Edges:
* (Document)-[:HAS_CHUNK]->(Chunk)
* (Chunk)-[:MENTIONS]->(Entity)

The retrieval policy is explicit and auditable:
* Anchors: rank documents by #distinct query entities matched (then #chunks).
* Evidence: rank chunks by #distinct query entities matched (then length).
* Optional neighbor expansion adds ±N adjacent chunks per matched chunk.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from neo4j.exceptions import Neo4jError

from .logging import get_logger
from .models import RetrievalHit
from .neo4j_client import MyNeo4j


LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Schema helpers
# --------------------------------------------------------------------------------------


def ensure_graph_schema(
    client: MyNeo4j,
    *,
    create_fulltext: bool = True,
    observability: bool = False,
) -> None:
    """Ensure constraints / indexes exist.

    The function is idempotent for constraints / b-tree indexes via
    `IF NOT EXISTS`. Fulltext indexes do not support `IF NOT EXISTS` across
    all Neo4j versions, so we try to detect first and otherwise best-effort.
    """

    constraints = [
        # Entity names are global.
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        # Partition documents by store.
        "CREATE CONSTRAINT doc_store_path_unique IF NOT EXISTS FOR (d:Document) REQUIRE (d.store, d.path) IS UNIQUE",
        # Each chunk is uniquely identified by (store, doc_path, chunk_index).
        "CREATE CONSTRAINT chunk_store_doc_idx_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.store, c.doc_path, c.chunk_index) IS UNIQUE",
        # Useful lookup index for neighbor expansion.
        "CREATE INDEX chunk_store_doc_path IF NOT EXISTS FOR (c:Chunk) ON (c.store, c.doc_path)",
    ]

    for cypher in constraints:
        _run_schema(client, cypher, observability=observability)

    if not create_fulltext:
        return

    # Optional fulltext index for keyword fallback.
    # NOTE: Neo4j supports fulltext in Community, but the procedure / SHOW syntax
    # varies slightly by version.
    name = "chunkText"
    if _fulltext_index_exists(client, name):
        return

    if not _fulltext_create_available(client):
        LOGGER.info(
            "Fulltext index creation procedure not available; skipping fulltext index setup (store=%s)",
            client.store_label,
        )
        return

    try:
        cypher = (
            "CALL db.index.fulltext.createNodeIndex($name, ['Chunk'], ['text','explicit_terms_text'])"
        )
        _run_schema(client, cypher, params={"name": name}, observability=observability)
    except Exception:
        # Not fatal. Retrieval still works via entity matches.
        if observability:
            print(f"[GRAPH_SCHEMA] fulltext index '{name}' not created (non-fatal)")


def _run_schema(
    client: MyNeo4j,
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    observability: bool = False,
) -> None:
    if observability:
        print(
            "\n[GRAPH_SCHEMA_CYPHER]\n"
            + json.dumps({"cypher": cypher, "params": params or {}}, ensure_ascii=False, indent=2, sort_keys=True)
        )

    try:
        client.run(cypher, params or {}, readonly=False)
    except Neo4jError as exc:
        # Many schema operations require admin rights. We don't hard fail.
        if observability:
            print(
                f"[GRAPH_SCHEMA] schema operation failed (store={client.store_label}): {getattr(exc, 'message', str(exc))}"
            )


def _fulltext_index_exists(client: MyNeo4j, name: str) -> bool:
    # Prefer SHOW INDEXES (Neo4j 4+).
    try:
        recs = client.run(
            "SHOW INDEXES YIELD name, type WHERE name = $name AND type = 'FULLTEXT' RETURN name",
            {"name": name},
            readonly=True,
        )
        return bool(recs)
    except Exception:
        pass

    # Fallback: CALL db.indexes() for older versions.
    try:
        recs = client.run(
            "CALL db.indexes() YIELD name, type WHERE name = $name AND type = 'FULLTEXT' RETURN name",
            {"name": name},
            readonly=True,
        )
        return bool(recs)
    except Exception:
        return False


def _fulltext_create_available(client: MyNeo4j) -> bool:
    """Return True when the fulltext create procedure is available."""

    try:
        recs = client.run(
            "CALL dbms.procedures() YIELD name WHERE name = $name RETURN name",
            {"name": "db.index.fulltext.createNodeIndex"},
            readonly=True,
        )
        return bool(recs)
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


def _store_key(client: MyNeo4j) -> str:
    return (client.store_label or "").strip().lower() or "long"


def graph_retrieve_doc_anchors(
    graph_client: MyNeo4j,
    *,
    question: str,
    entities: List[str],
    k: int,
    observability: bool,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Return top-K document paths as anchors.

    We rank documents by number of distinct query entities that appear anywhere
    in the document (via chunk mentions), then by number of matched chunks.
    """

    store = _store_key(graph_client)

    if not entities:
        # Without entities, graph-only retrieval is ambiguous. Keep anchors empty
        # and let the orchestration decide whether to fall back.
        return [], {"cypher": "", "params": {}}, []

    cypher = (
        "MATCH (d:Document {store: $store})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
        "WHERE e.name IN $entities "
        "WITH d, collect(DISTINCT e.name) AS matched_entities, count(DISTINCT c) AS matched_chunks "
        "RETURN d.path AS path, d.category AS category, "
        "       size(matched_entities) AS entity_matches, "
        "       matched_chunks AS chunk_matches, "
        "       matched_entities AS matched_entities "
        "ORDER BY entity_matches DESC, chunk_matches DESC, path ASC "
        "LIMIT $k"
    )

    params = {"store": store, "entities": [e.strip().lower() for e in entities if e.strip()], "k": int(k)}
    query_obj = {"cypher": cypher, "params": params}

    if observability:
        print("\n[GRAPH_DOC_ANCHOR_QUERY]\n" + json.dumps(query_obj, ensure_ascii=False, indent=2, sort_keys=True))

    records: List[Dict[str, Any]] = []
    try:
        records = graph_client.run(cypher, params, readonly=True)
    except Exception:
        records = []

    paths: List[str] = []
    for r in records:
        p = r.get("path")
        if p:
            paths.append(str(p))

    # unique preserve order
    seen: set[str] = set()
    uniq: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    return uniq, query_obj, records


def graph_retrieve_chunks(
    graph_client: MyNeo4j,
    *,
    question: str,
    entities: List[str],
    k: int,
    anchor_paths: Optional[List[str]],
    neighbor_window: int,
    observability: bool,
) -> Tuple[List[RetrievalHit], Dict[str, Any], List[Dict[str, Any]]]:
    """Retrieve grounding chunks from the Knowledge Graph."""

    store = _store_key(graph_client)

    if not entities:
        return [], {"cypher": "", "params": {}}, []

    cypher = (
        "MATCH (d:Document {store: $store})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
        "WHERE e.name IN $entities "
        "  AND ($anchor_paths IS NULL OR d.path IN $anchor_paths) "
        "WITH d, c, collect(DISTINCT e.name) AS matched_entities "
        "WITH d, c, matched_entities, size(matched_entities) AS entity_matches "
        "RETURN d.path AS path, d.category AS category, "
        "       c.chunk_index AS chunk_index, c.chunk_count AS chunk_count, "
        "       c.text AS text, "
        "       matched_entities AS matched_entities, "
        "       entity_matches AS entity_matches "
        "ORDER BY entity_matches DESC, size(c.text) DESC, path ASC, chunk_index ASC "
        "LIMIT $k"
    )

    params = {
        "store": store,
        "entities": [e.strip().lower() for e in entities if e.strip()],
        "anchor_paths": list(anchor_paths) if anchor_paths else None,
        "k": int(k),
    }
    query_obj = {"cypher": cypher, "params": params}

    if observability:
        print("\n[GRAPH_CHUNK_QUERY]\n" + json.dumps(query_obj, ensure_ascii=False, indent=2, sort_keys=True))

    raw: List[Dict[str, Any]] = []
    try:
        raw = graph_client.run(cypher, params, readonly=True)
    except Exception:
        raw = []

    hits: List[RetrievalHit] = []
    for i, r in enumerate(raw, start=1):
        path = str(r.get("path") or "")
        chunk_index = r.get("chunk_index")
        try:
            chunk_index_int = int(chunk_index) if chunk_index is not None else None
        except Exception:
            chunk_index_int = None

        chunk_count = r.get("chunk_count")
        try:
            chunk_count_int = int(chunk_count) if chunk_count is not None else None
        except Exception:
            chunk_count_int = None

        matched_entities = r.get("matched_entities") or []
        if not isinstance(matched_entities, list):
            matched_entities = []

        entity_matches = r.get("entity_matches")
        try:
            entity_matches_int = int(entity_matches) if entity_matches is not None else None
        except Exception:
            entity_matches_int = None

        chunk_id = path
        if chunk_index_int is not None:
            chunk_id = f"{path}::chunk-{chunk_index_int:03d}"

        hits.append(
            RetrievalHit(
                channel="graph_chunk",
                handle=f"G{i}",
                index=f"neo4j:{graph_client.database}",
                os_id=chunk_id,
                score=float(entity_matches_int or 0),
                store=graph_client.store_label,
                path=path,
                category=str(r.get("category") or ""),
                chunk_index=chunk_index_int,
                chunk_count=chunk_count_int,
                text=(str(r.get("text") or "").strip()),
                explicit_terms=[str(x) for x in matched_entities if str(x).strip()],
                entity_overlap=entity_matches_int,
                meta={
                    "matched_entities": matched_entities,
                    "entity_matches": entity_matches_int,
                },
            )
        )

    if neighbor_window > 0 and hits:
        hits = _expand_graph_neighbors(
            graph_client,
            seed_hits=hits,
            window=neighbor_window,
            observability=observability,
        )
        # Re-handle
        hits = [RetrievalHit(**{**h.to_jsonable(), "handle": f"G{i+1}"}) for i, h in enumerate(hits)]

    return hits, query_obj, raw


def _expand_graph_neighbors(
    graph_client: MyNeo4j,
    *,
    seed_hits: List[RetrievalHit],
    window: int,
    observability: bool,
) -> List[RetrievalHit]:
    store = _store_key(graph_client)

    doc_positions: Dict[str, List[int]] = {}
    seed_score: Dict[str, float] = {}
    seed_entities: Dict[str, List[str]] = {}
    for h in seed_hits:
        seed_score[h.os_id] = float(h.score or 0.0)
        if h.explicit_terms:
            seed_entities[h.os_id] = list(h.explicit_terms)
        if h.chunk_index is None:
            continue
        doc_positions.setdefault(h.path, []).append(int(h.chunk_index))

    expanded: List[RetrievalHit] = []

    for path, positions in doc_positions.items():
        idxs: set[int] = set()
        for pos in positions:
            for off in range(-window, window + 1):
                if pos + off >= 0:
                    idxs.add(pos + off)

        if not idxs:
            continue

        cypher = (
            "MATCH (c:Chunk {store: $store, doc_path: $path}) "
            "WHERE c.chunk_index IN $idxs "
            "RETURN c.doc_path AS path, c.category AS category, "
            "       c.chunk_index AS chunk_index, c.chunk_count AS chunk_count, "
            "       c.text AS text"
        )
        params = {"store": store, "path": path, "idxs": sorted(idxs)}

        if observability:
            print(
                "\n[GRAPH_NEIGHBOR_QUERY]\n"
                + json.dumps({"cypher": cypher, "params": params}, ensure_ascii=False, indent=2, sort_keys=True)
            )

        try:
            recs = graph_client.run(cypher, params, readonly=True)
        except Exception:
            recs = []

        for r in recs:
            ci = r.get("chunk_index")
            try:
                ci_int = int(ci) if ci is not None else None
            except Exception:
                ci_int = None

            cc = r.get("chunk_count")
            try:
                cc_int = int(cc) if cc is not None else None
            except Exception:
                cc_int = None

            chunk_id = f"{path}::chunk-{ci_int:03d}" if ci_int is not None else path
            expanded.append(
                RetrievalHit(
                    channel="graph_chunk" if chunk_id in seed_score else "graph_neighbor",
                    handle="G0",  # temporary, reassigned later
                    index=f"neo4j:{graph_client.database}",
                    os_id=chunk_id,
                    score=float(seed_score.get(chunk_id, 0.0)),
                    store=graph_client.store_label,
                    path=str(r.get("path") or path),
                    category=str(r.get("category") or ""),
                    chunk_index=ci_int,
                    chunk_count=cc_int,
                    text=str(r.get("text") or "").strip(),
                    explicit_terms=seed_entities.get(chunk_id),
                    entity_overlap=None,
                )
            )

    # de-dup (path, chunk_index) preserve order
    seen: set[Tuple[str, Optional[int], str]] = set()
    uniq: List[RetrievalHit] = []
    for h in expanded:
        key = (h.store or "", h.path, h.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    if observability:
        print(f"\n[GRAPH_NEIGHBOR_EXPANSION] window={window} expanded_chunks={len(uniq)}")

    return uniq


__all__ = [
    "ensure_graph_schema",
    "graph_retrieve_doc_anchors",
    "graph_retrieve_chunks",
]
