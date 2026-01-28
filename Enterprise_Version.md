# Hybrid RAG: Enterprise Version

Retrieval-Augmented Generation (RAG) has become a critical pattern for grounding Large Language Model (LLM) responses in real-world data, improving both accuracy and reliability. Yet, conventional RAG implementations often default to vector-only databases, which come with drawbacks: weaker precision for exact facts, opaque ranking logic, and challenges with regulatory compliance.

![Enterprise Hybrid RAG](./images/enterprise_deployment.png)

This enterprise architecture uses a **Hybrid RAG** design that blends:

- **Graph-based truth grounding** in **Neo4j** (explicit entities, relationships, evidence chunks, and provenance)
- **Vector embeddings for semantic context** in **OpenSearch** (kNN over chunk embeddings)

In other words: the **graph** answers "what are the grounded facts and where did they come from?", while the **vectors** answer "what else is contextually relevant that improves clarity and recall?". Together, the system produces retrievals that are observable, reproducible, and audit-ready without sacrificing nuance for ambiguous or paraphrased queries.

## Key Benefits of Hybrid RAG (Using Graph + Vector)

* **Transparency and Explainability**: Graph grounding is traceable through explicit entity matches, subgraph retrieval, and evidence chunks. You can show exactly which entities and which chunks anchored the answer.
* **Semantic Coverage without Drift**: Vector context augments graph evidence, expanding recall while keeping factual anchors intact.
* **Determinism and Auditability**: Graph retrieval is driven by explicit signals (entity overlap, anchored documents, neighborhood expansion) with stable provenance metadata.
* **Governance and Compliance**: Observable retrieval paths and lineage-friendly metadata simplify regulatory adherence and policy enforcement.
* **Bias and Risk Mitigation**: Entity extraction and curated graph evidence make signal selection explicit and reviewable, reducing silent failures common in embedding-only systems.

## Dual-Memory Architecture

The enterprise Hybrid RAG agent uses a **HOT (unstable) + Long-Term store** design. To meet enterprise SLAs and governance boundaries, we map these logical tiers to specific NetApp storage capabilities that guarantee isolation and performance.

This architecture uses **two databases**:

- **Neo4j** for the Knowledge Graph (truth grounding)
- **OpenSearch** for vector retrieval (contextual awareness)

| Memory Type | Infrastructure | NetApp Augmentation | Data Stored | Purpose |
| --- | --- | --- | --- | --- |
| **Long-term Memory** | **Neo4j (LT)** + OpenSearch vector index | **MetroCluster** (HA)<br><br>**FabricPool** (Tiering)<br><br>**SnapLock** (WORM) | Curated, validated evidence: documents, chunks, entities, relationships; plus embeddings for semantic context | Authoritative knowledge base; compliance-ready |
| **HOT (unstable)** | **Neo4j (HOT)** + OpenSearch vector index (shared or filtered) | **FlexCache**<br><br>**Storage QoS**<br><br>**FlexClone** | User-specific, unvetted evidence chunks and entities (plus optional vector context) | Governance boundary, per-user isolation, retention variations |

## Business Impact

Deploying Graph-grounded Hybrid RAG in the enterprise yields clear strategic advantages:

* **Operational clarity** via parallel LT+HOT retrievals and a merged view that respects graph truth grounding while enriching with semantic context.
* **Improved compliance and risk control** thanks to explainable evidence retrieval and complete data lineage, backed by immutable snapshot technology.
* **Scalability and resilience** by leveraging enterprise storage practices and predictable retrieval behavior.

By grounding AI systems in observable **graph evidence retrieval** and supplementing results with **semantic vector context**, enterprises can increase trustworthiness, compliance, and operational clarity while meeting real-world performance requirements.

This guide provides an enterprise-oriented reference for implementing Graph + Vector Hybrid RAG architectures, enabling organizations to build faster, clearer, and fully governable AI solutions.

# 2. Ingesting Your Data Into Long-Term Memory

> **Same core pipeline, enterprise-grade surroundings.**
> This section mirrors the community version but emphasizes enterprise priorities such as audit trails, schema governance, and storage economics.

## Why We Start with Clean Knowledge

Long-term memory is the system's **source of truth**. Anything that lands here must be:

1. **Authoritative**: derived from validated, trusted documents.
2. **Traceable**: every evidence chunk is tied to explicit provenance and metadata.
3. **Governance-ready**: aligned with organizational taxonomies, compliance policies, and audit requirements.

## Four-Step Ingestion Pipeline

| Stage | What Happens | Enterprise Add-Ons |
| --- | --- | --- |
| **1. Parse** | Raw content (text files, PDFs, tickets) is loaded into the pipeline. | **NetApp XCP** accelerates migration of massive legacy datasets (Hadoop/HDFS, NFS) into the ingest path. |
| **2. Slice** | Documents are chunked (paragraphs for graph evidence; sliding windows for vectors). | Preserve offsets and category metadata if you later enable alternative slicing strategies. |
| **3. Extract Entities** | An **external NER HTTP service** returns normalized entities (lowercased, deduped). | Maintain the NER service version separately; entities and terms are stored, not the model name. |
| **4. Persist (Graph + Vectors)** | Evidence is written to **Neo4j** (Documents/Chunks/Entities + edges), and vector chunks are embedded and indexed into **OpenSearch**. | Use **FlexClone** to fork Neo4j/OpenSearch volumes for testing new extraction or embedding strategies without impacting production. |

The ingestion process ensures idempotency at the document level via a stable identifier derived from the source path or URI. In enterprise settings, you can add a **batch ID** around each run to trace or roll back entire ingests.

### Implementation Considerations

* **Fast Migration with XCP:** If your source data lives in legacy HDFS or disparate NFS silos, use **NetApp XCP** to consolidate it into the ingest path. XCP offers high-throughput copy and verification, ensuring that the "Source of Truth" in your RAG matches the source systems bit-for-bit.
* **Graph schema stability:** Keep merge keys stable (`doc_id`, `chunk_id`, canonicalized entity names). This is what makes graph retrieval reproducible.
* **Fail-soft NER behavior:** Capture NER errors and store empty entity lists without halting the run. Broken NER should degrade retrieval quality, not take production down.
* **Safe upgrades with FlexClone:** When testing new chunking policies, entity normalization rules, or embedding models, clone the underlying volumes for Neo4j and/or OpenSearch and test against the clone. This keeps production isolated and audit-friendly.
* Provenance metadata should remain explicit and minimal: `filepath`/`URI`, `ingested_at_ms`, and `doc_version`. Track NER and embedding model versions at the service layer for audit and reproducibility.
* **HOT → LT promotion policy:** the only time data moves from HOT back into LT is when there is (1) enough positive reinforcement to warrant promotion **or** (2) a trusted human-in-the-loop has verified the data.

## Additional Notes

* *Version everything.* Use `doc_version` and your VCS/infra configs to record schema and pipeline changes you might rerun later.

With clean, well-labeled evidence in long-term memory, every downstream query inherits stable, auditable provenance. Next, we cover how to maintain **HOT (unstable)** as a user-specific tier for governance boundaries, TTL-based lifecycle, and operational isolation.

# 4. Promotion of Long-Term Memory into **HOT (unstable)**

> **Goal:** Maintain a governed, low-retention working set near the application while preserving provenance and compliance. This tiering exists primarily for **governance, isolation, and policy asymmetry**, not because LT is inherently slow.

## Why Promote?

* **Retention variations control.** HOT absorbs high write rates, user uploads, and unstable facts without impacting LT governance posture.
* **Policy asymmetry.** LT stays conservative (curated, WORM-like behavior); HOT can be permissive with TTL-based lifecycle and user-scoped evidence.
* **Operational locality.** Keep user-specific context colocated with serving paths while leaving LT untouched. Throughput improves by isolating churn, not by mutating authoritative stores.

## Enterprise Twist vs. Community Guide

| Stage | Community Edition | Enterprise Edition |
| --- | --- | --- |
| **Detect entities** | Extract entities in-process | **External NER HTTP service** produces normalized entities (lowercased, de-duplicated). |
| **Select evidence** | Graph retrieval over LONG | Evidence selection can be constrained by entity sets, document anchors, and policy filters. |
| **Transfer data** | Script writes user evidence to HOT | HOT **materializes** user-relevant evidence into **Neo4j (HOT)** with stable IDs and lifecycle stamps. |
| **TTL management** | Simple cron delete | **Eviction job** removes HOT evidence past TTL with rate limiting and capped batches. |
| **Performance** | Standard shared resources | **Storage QoS** guarantees HOT tier latency is never starved by background ingest or maintenance. |

## Promotion Flow (Enterprise)

1. **User-specific context is selected out-of-band** from uploads, per-user ingest, or operator input using the external NER service.
2. **Promotion filter** targets normalized entities and optional time windows (for example using `ingested_at_ms`) when pulling shared evidence into a user scope.
3. **HOT graph materialization** copies only the relevant subgraph into the HOT Neo4j store and stamps `hot_promoted_at` for lifecycle control.
4. **Serving**: the application queries **LT and HOT graphs in parallel** (Neo4j) for grounded evidence and queries **OpenSearch** for vector context. Results are merged into:
   - an **evidence channel** (graph chunks and provenance)
   - a **support channel** (vector-retrieved contextual chunks)
5. **Eviction job** removes HOT nodes/edges/chunks whose `hot_promoted_at` exceeds the configured TTL.

> **Reverse path (HOT → LT):** Promotion occurs only when (1) sufficient positive reinforcement warrants it **or** (2) a trusted human-in-the-loop verifies the data.

## Operational Knobs

| Variable | Typical Value | Purpose |
| --- | --- | --- |
| `HOT_TTL_MINUTES` | `30` | Eviction threshold for `hot_promoted_at`. |
| `GRAPH_LONG_URI` | `bolt://…` | Neo4j LT connection for authoritative evidence. |
| `GRAPH_HOT_URI` | `bolt://…` | Neo4j HOT connection for user-scoped evidence. |
| `VECTOR_INDEX` | `bbc-vec-chunks` | OpenSearch vector index for semantic support. |
| `GRAPH_DOC_K` | `10` | Document anchors selected from the graph (optional). |
| `GRAPH_CHUNK_K` | `25` | Evidence chunk budget from graph retrieval. |
| `VEC_K` | `12` | Vector retrieval budget for contextual support. |
| `NEIGHBOR_WINDOW` | `1` | Expand ±N chunks around matched evidence for continuity. |

### Replication: the Enterprise Upgrade Path

Default is **out-of-band HOT materialization + TTL eviction**.

If you need continuous movement at scale:

1. **Storage-level replication** (SnapMirror) can replicate Neo4j and OpenSearch volumes for DR and governance.
2. **OpenSearch snapshot and lifecycle policies** can enforce retention and recovery for vector indices.
3. **Event-driven promotions** (Kafka CDC, workflow triggers) can materialize HOT evidence when upstream systems change.

### Where to Run **HOT**

Your HOT backing choice governs performance and operational flexibility:

| Option | Speed | Caveats | Best for |
| --- | --- | --- | --- |
| **Ephemeral HOT graph** | Fast, minimal overhead | Volatile on restart; limited retention | Demos, PoCs |
| **Local NVMe-backed HOT** | Low latency, high write throughput | Data tied to node; failover requires orchestration | Fixed clusters |
| **ONTAP FlexCache-backed HOT** | Predictable reads at scale | Requires ONTAP; improves portability and rescheduling | Production Kubernetes or multi-site setups |

**Why FlexCache Helps Enterprises**

* **Elastic capacity** beyond RAM without pipeline redesigns.
* **Portability** cache volumes can follow pods across nodes/AZs.
* **Governance** SnapMirror and thin provisioning aid audit and cost control.

In short: **HOT materialization + TTL eviction** gives speed-through-isolation and determinism today; adding **replication + policy automation + FlexCache** layers in resilience and governance when scale and ops require it.

## 4. Implementation Guide

For a reference, please check out the following: [enterprise_version/README.md](./enterprise_version/README.md)

# 5. Conclusion

Graph-grounded Hybrid RAG turns retrieval-augmented generation into a transparent, governed architecture. By grounding retrieval in explicit **graph evidence** (Neo4j) and enriching responses with **vector similarity** (OpenSearch), you get answers that are:

* **Governed performance.** Parallel queries to LT and HOT keep UX consistent; tiers exist primarily for governance, isolation, and policy asymmetry.
* **Clearer.** Evidence is traceable through entities, chunks, and provenance, with semantic context treated as supportive rather than authoritative.
* **Safer.** Grounded evidence reduces hallucinations while vector context improves recall for paraphrases and long-tail phrasing.
* **Compliant.** Built-in provenance metadata (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) makes regulatory alignment and retention policies straightforward.

The enterprise path centers on **user-specific HOT graph materialization** executed out-of-band (not in the request path), plus a **TTL eviction job** keyed on `hot_promoted_at`. Serving queries LT and HOT graphs in parallel, merges grounded evidence with vector context, and supplies the combined result to the LLM. **HOT → LT promotion occurs only** when sufficient reinforcement or human verification says "yes."

## NetApp Enterprise Enhancements: The Resilience Layer

While the logic layer handles RAG, NetApp technologies solidify the physical data layer into an enterprise-grade platform:

* **High Availability with Zero RPO**: For authoritative **Long-Term** evidence stores (Neo4j + OpenSearch), use **NetApp MetroCluster** to maintain continuity even under site failure.
* **Noisy Neighbor Isolation via QoS**: RAG workloads are bursty. Ingest jobs can saturate I/O and starve HOT. Use **NetApp Storage QoS** to guarantee HOT tier latency while controlling background maintenance throughput.
* **Instant ML Ops with FlexClone**: Testing new entity normalization rules, graph schemas, or embedding models usually means risky migrations. **NetApp FlexClone** lets you fork Neo4j/OpenSearch volumes instantly for A/B testing without impacting production.
* **Compliance Snapshots**: Use **SnapCenter** to take immutable, application-consistent snapshots of both graph and vector stores. This enables point-in-time "what did the agent know then?" audits.
* **Accelerated Ingest with XCP**: When populating LT from legacy HDFS or vast NFS shares, **NetApp XCP** provides the high-performance migration needed to hydrate the pipeline efficiently.

## Next Steps

1. **Clone the repo.** The reference code and docs live at `github.com/NetApp/hybrid-rag-graph-with-ai-governance`. Try it locally with Docker Compose.
2. **Stand up the stores.** Deploy Neo4j (LT + HOT) and OpenSearch, then configure credentials and indices.
3. **Feed it live data.** Point ingest at your corpus (tickets, PDFs, KBs) and run the NER service.
4. **Tune retrieval budgets.** Adjust graph anchor budgets, graph chunk budgets, neighbor expansion, and vector `k` until evidence stays compact and accurate.
5. **Operationalize HOT.** Decide what belongs in HOT, how long it lives, and what "promotion-ready" means in your environment.
6. **Share lessons.** File issues, submit pull requests, or publish a case study. Enterprise-grade governance improves when patterns are shared.

Graph-based Hybrid RAG is not a science project. It is running code with traceable evidence and an operational posture that auditors can tolerate.
