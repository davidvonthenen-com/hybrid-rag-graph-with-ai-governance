#!/usr/bin/env python3
"""Hybrid ingestion for the BBC dataset (Graph + Vector).

This replaces the BM25 paragraph-chunk truth grounding with a Neo4j Knowledge Graph
grounding index while keeping vector embedding ingestion exactly the same.

Ingests:
* Neo4j (graph grounding)
    - (:Document {store, path, category, ...})
    - (:Chunk {store, doc_path, chunk_index, chunk_count, text, ...})
    - (:Entity {name})
    - (Document)-[:HAS_CHUNK]->(Chunk)
    - (Chunk)-[:MENTIONS]->(Entity)

* OpenSearch (vector semantic context)
    - paragraph sliding-window chunks into a kNN vector index
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm
from opensearchpy import OpenSearch

from common.embeddings import EmbeddingModel, to_list
from common.graph import ensure_graph_schema
from common.logging import get_logger
from common.named_entity import post_ner, normalize_entities
from common.neo4j_client import create_graph_hot_client, create_graph_long_client, MyNeo4j
from common.opensearch_client import create_vector_client


LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ingestion job."""

    parser = argparse.ArgumentParser(
        description="Hybrid ingestion: Neo4j graph grounding + OpenSearch vector chunks (BBC dataset)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bbc",
        help="Path to BBC dataset root (category subdirs)",
    )
    parser.add_argument(
        "--graph-store",
        choices=["long", "hot", "both"],
        default="long",
        help="Which graph store(s) to ingest into.",
    )
    parser.add_argument(
        "--no-graph-fulltext",
        action="store_true",
        default=False,
        help="Disable attempting to create a Neo4j fulltext index for keyword fallback.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Vector batch size (paragraphs)",
    )
    parser.add_argument(
        "--graph-chunking",
        choices=["fixed", "paragraph"],
        default="fixed",
        help="Graph chunking strategy: fixed-size window (default) or paragraph-based.",
    )
    parser.add_argument(
        "--graph-chunk-size",
        type=int,
        default=1024,
        help="Maximum characters per graph chunk when using fixed-size chunking.",
    )
    parser.add_argument(
        "--graph-chunk-overlap",
        type=int,
        default=256,
        help="Character overlap between consecutive graph chunks when using fixed-size chunking.",
    )
    parser.add_argument(
        "--vector-chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per vector chunk",
    )
    parser.add_argument(
        "--vector-chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive vector chunks",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs separated by blank lines."""

    paragraphs: List[str] = []
    current: List[str] = []

    for line in text.splitlines():
        if line.strip():
            current.append(line)
            continue

        if current:
            paragraphs.append("\n".join(current).strip())
            current = []

    if current:
        paragraphs.append("\n".join(current).strip())

    return paragraphs or ([text.strip()] if text.strip() else [])


def extract_entities(text: str) -> List[str]:
    """Run NER and return normalized entity strings."""

    ner_result = post_ner(text)
    return normalize_entities(ner_result)


# ---------------------------------------------------------------------------
# Vector index management (unchanged from BM25 version)
# ---------------------------------------------------------------------------


def ensure_vector_index(client: OpenSearch, index_name: str, dim: int) -> None:
    """Ensure that the target index exists with the correct mapping."""
    if client.indices.exists(index=index_name):
        LOGGER.info("OpenSearch index '%s' already exists", index_name)
        return

    body = {
        "settings": {
            "index": {
                # Enable k-NN index structures
                "knn": True,
                # Lucene engine ignores ef_search and derives it from k,
                # but keeping this setting is harmless if you switch engines later.
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        # You can add "parameters": {"m": 16, "ef_construction": 128} here if desired.
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------


def iter_bbc_files(data_dir: Path):
    """Yield (category, file_path, text) for each BBC article."""

    for category_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        for fp in sorted(category_dir.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            yield category, fp, text


@dataclass
class IngestStats:
    """Counters for the ingestion run."""

    docs: int = 0
    graph_chunks: int = 0
    vector_chunks: int = 0


def _build_graph_chunks(
    chunk_texts: Iterable[str],
    *,
    chunk_count: int,
) -> List[Dict[str, object]]:
    """Create graph chunk payloads with entity extraction.

    Args:
        chunk_texts: Iterable of chunk strings.
        chunk_count: Total number of chunks for the document.

    Returns:
        Chunk payloads with entity metadata for graph ingestion.
    """

    out: List[Dict[str, object]] = []
    for idx, chunk in enumerate(chunk_texts):
        cleaned = chunk.strip()
        if not cleaned:
            continue
        ents = extract_entities(cleaned)
        out.append(
            {
                "chunk_index": int(idx),
                "chunk_count": int(chunk_count),
                "text": cleaned,
                "entities": ents,
                "entities_text": " ".join(ents) if ents else "",
            }
        )
    return out


def build_vector_chunks(
    paragraphs: Iterable[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Create sliding-window text chunks for vector embeddings.

    Args:
        paragraphs: Paragraphs from the source document. Blank entries are ignored.
        chunk_size: Maximum size (in characters) of each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of chunk strings sized for dense embedding models.
    """

    cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not cleaned_paragraphs:
        return []

    combined_text = "\n\n".join(cleaned_paragraphs)
    chunks: List[str] = []

    start = 0
    text_length = len(combined_text)
    step = max(chunk_size - chunk_overlap, 1)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = combined_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += step

    return chunks


def build_graph_chunk_texts(
    paragraphs: Iterable[str],
    *,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Create graph chunk text based on the selected strategy.

    Args:
        paragraphs: Paragraphs from the source document.
        strategy: Chunking strategy ("fixed" or "paragraph").
        chunk_size: Maximum size (in characters) of each chunk when using fixed chunking.
        chunk_overlap: Number of characters to overlap between consecutive chunks when using fixed chunking.

    Returns:
        A list of chunk strings to ingest into the graph.

    Raises:
        ValueError: If an unsupported chunking strategy is provided.
    """

    cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not cleaned_paragraphs:
        return []

    if strategy == "fixed":
        return build_vector_chunks(
            cleaned_paragraphs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if strategy == "paragraph":
        return cleaned_paragraphs

    raise ValueError(f"Unsupported graph chunking strategy: {strategy}")


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------

def doc_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


_CYPHER_INGEST_DOC = """
MERGE (d:Document {store: $store, path: $path})
SET d.category = $category,
    d.content = $content,
    d.explicit_terms = $doc_entities,
    d.explicit_terms_text = $doc_entities_text,
    d.ingested_at_ms = $now_ms,
    d.doc_version = $now_ms
WITH d
UNWIND $chunks AS ch
MERGE (c:Chunk {store: $store, doc_path: $path, chunk_index: ch.chunk_index})
SET c.chunk_count = ch.chunk_count,
    c.text = ch.text,
    c.category = $category,
    c.explicit_terms = ch.entities,
    c.explicit_terms_text = ch.entities_text,
    c.ingested_at_ms = $now_ms,
    c.doc_version = $now_ms
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, ch
UNWIND coalesce(ch.entities, []) AS ent
MERGE (e:Entity {name: ent})
MERGE (c)-[:MENTIONS]->(e)
""".strip()


def _ingest_document_to_graph(
    graph_client: MyNeo4j,
    *,
    store: str,
    category: str,
    rel_path: str,
    text: str,
    graph_chunk_texts: List[str],
    now_ms: int,
) -> int:
    """Ingest a single document into Neo4j and return #chunks written."""

    doc_entities = extract_entities(text)
    chunks = _build_graph_chunks(graph_chunk_texts, chunk_count=len(graph_chunk_texts))

    params = {
        "store": store,
        "path": rel_path,
        "category": category,
        "content": text,
        "doc_entities": doc_entities,
        "doc_entities_text": " ".join(doc_entities) if doc_entities else "",
        "chunks": chunks,
        "now_ms": int(now_ms),
    }

    graph_client.run(_CYPHER_INGEST_DOC, params, readonly=False)
    return len(chunks)


def ingest_hybrid(
    data_dir: Path,
    *,
    graph_store: str,
    graph_fulltext: bool,
    batch_size: int,
    graph_chunking: str,
    graph_chunk_size: int,
    graph_chunk_overlap: int,
    vector_chunk_size: int,
    vector_chunk_overlap: int,
) -> None:
    """Run ingestion for graph + vector."""

    # 1) Connect Graph cluster (LONG)
    graph_hot = create_graph_hot_client()
    graph_long = create_graph_long_client()

    targets: List[tuple[MyNeo4j, str]] = []
    if graph_store in ("long", "both"):
        targets.append((graph_long, "long"))
    if graph_store in ("hot", "both"):
        targets.append((graph_hot, "hot"))

    for cli, store_key in targets:
        ensure_graph_schema(cli, create_fulltext=graph_fulltext, observability=False)
        LOGGER.info("Graph schema ensured for %s (store=%s db=%s)", cli.store_label, store_key, cli.database)

    # 2) Connect vector store and ensure index exists with right dimension
    vec_client, _ = create_vector_client()
    embedder = EmbeddingModel()
    ensure_vector_index(vec_client, vec_client.settings.opensearch_vector_index, embedder.dimension)

    now_ms = int(time.time() * 1000)
    stats = IngestStats()

    progress = tqdm(desc="Hybrid ingest (docs)", unit="doc")
    try:
        for category, fp, text in iter_bbc_files(data_dir):
            rel_path = fp.relative_to(data_dir).as_posix()

            paragraphs = split_into_paragraphs(text)
            graph_chunk_texts = build_graph_chunk_texts(
                paragraphs,
                strategy=graph_chunking,
                chunk_size=graph_chunk_size,
                chunk_overlap=graph_chunk_overlap,
            )

            # 1) Graph ingestion (truth grounding)
            for cli, store_key in targets:
                try:
                    written = _ingest_document_to_graph(
                        cli,
                        store=store_key,
                        category=category,
                        rel_path=rel_path,
                        text=text,
                        graph_chunk_texts=graph_chunk_texts,
                        now_ms=now_ms,
                    )
                    stats.graph_chunks += int(written)
                except Exception as exc:
                    LOGGER.warning(
                        "Graph ingest failed for %s store=%s path=%s: %s",
                        cli.store_label,
                        store_key,
                        rel_path,
                        f"{type(exc).__name__}: {exc}",
                    )

            # 2) Vector ingestion (semantic context)
            vector_text_chunks = build_vector_chunks(
                paragraphs,
                chunk_size=vector_chunk_size,
                chunk_overlap=vector_chunk_overlap,
            )

            vec_chunks: List[Dict[str, object]] = [
                {
                    "category": category,
                    "rel_path": rel_path,
                    "chunk_index": idx,
                    "chunk_count": len(vector_text_chunks),
                    "text": chunk_text,
                }
                for idx, chunk_text in enumerate(vector_text_chunks)
            ]

            for i in range(0, len(vec_chunks), batch_size):
                batch = vec_chunks[i : i + batch_size]
                if not batch:
                    continue
                texts = [c["text"] for c in batch]
                embeddings = embedder.encode(texts)
                for chunk_meta, embedding in zip(batch, embeddings):
                    emb_vec = to_list(embedding)
                    chunk_id = f"{chunk_meta['rel_path']}::chunk-{int(chunk_meta['chunk_index']):03d}"
                    vec_id = doc_sha1(chunk_id)
                    body = {
                        "path": chunk_meta["rel_path"],
                        "category": chunk_meta["category"],
                        "chunk_index": chunk_meta["chunk_index"],
                        "chunk_count": chunk_meta["chunk_count"],
                        "text": chunk_meta["text"],
                        "embedding": emb_vec,
                    }
                    vec_client.index(
                        index=vec_client.settings.opensearch_vector_index,
                        id=vec_id,
                        body=body,
                        refresh=False,
                    )
                    stats.vector_chunks += 1

            stats.docs += 1
            progress.update(1)
    finally:
        progress.close()

    vec_client.indices.refresh(index=vec_client.settings.opensearch_vector_index)

    for cli, _store_key in targets:
        cli.close()

    LOGGER.info(
        "Hybrid ingest complete: %d docs, %d graph chunks, %d vector chunks into vector_index='%s'",
        stats.docs,
        stats.graph_chunks,
        stats.vector_chunks,
        vec_client.settings.opensearch_vector_index,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    ingest_hybrid(
        data_dir=data_dir,
        graph_store=str(args.graph_store),
        graph_fulltext=not bool(args.no_graph_fulltext),
        batch_size=int(args.batch_size),
        graph_chunking=str(args.graph_chunking),
        graph_chunk_size=int(args.graph_chunk_size),
        graph_chunk_overlap=int(args.graph_chunk_overlap),
        vector_chunk_size=int(args.vector_chunk_size),
        vector_chunk_overlap=int(args.vector_chunk_overlap),
    )


if __name__ == "__main__":
    main()
