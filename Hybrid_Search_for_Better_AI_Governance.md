# **Hybrid Search for Better AI Governance**

## **1. Executive Summary**

Most RAG implementations lean on vectors alone or on generic "hybrid" search that blends dense vectors with lexical scoring. Hybrid improves recall and robustness, and common search stacks can combine lexical and vectors to boost retrieval quality. That said, even "hybrid" often blurs *why* something surfaced. Semantic similarity scores still aren't human-interpretable, and "keyword match" alone doesn't capture the relationships that matter in enterprise knowledge (who did what, to whom, where, and under which constraints).

This repository implements a **Hybrid RAG** that is intentionally **two-channel**:

* **Knowledge Graph (GraphRAG) as the grounding channel** for factual evidence and auditability.
* **Vector kNN as the semantic support channel** to add contextual language and phrasing without introducing new facts.

A domain-aware [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) service runs at ingest and query time.

* At ingest, entities are normalized and stored explicitly alongside content and provenance metadata.
* At query time, extracted entities drive **graph retrieval** first (deterministic, inspectable), and then optionally constrain **vector retrieval** to reduce semantic drift.

From an AI Governance standpoint, this design is superior on four fronts:

* **Transparency and Explainability**: Factual claims are grounded in graph-retrieved evidence first, with explicit references to which evidence chunks contributed to the answer. Vector evidence is used only to clarify, not to introduce new facts.
* **Accountability and Responsibility**: Retrieval steps are reproducible and loggable. Evidence carries stable identifiers plus provenance fields. The query runner can emit full observability traces and audit records that capture queries, retrieved chunks, and final answers.
* **Data Governance and Regulatory Compliance**: The dual-store layout is explicit. **HOT (unstable)** holds user-specific or experimental material; **LT** retains vetted knowledge. This makes retention and access policies enforceable and keeps audited content separate from unverified input.
* **Risk Management and Safety**: Answers are grounded in explicit evidence selection (entity overlap and graph anchoring). Vector context is constrained to graph anchors when possible, making noise easier to detect and limiting hallucinations.

**Bottom line:**
Pure vectors maximize fuzzy recall. Hybrid (vector + lexical) balances fuzziness with keywords. **This repo goes further by separating roles and grounding in structure.** The graph provides evidence; vectors improve phrasing... so answers stay explainable, reproducible, and governable without sacrificing retrieval quality.

> **IMPORTANT NOTE:** This implementation uses a Knowledge Graph for explicit grounding and a dedicated vector index for semantic context. Searches query **LT and HOT in parallel** for graph evidence, then use **vector kNN** for semantic context, typically filtered to graph-anchored documents. The LLM first drafts a grounded answer from graph evidence and then (optionally) refines language using vector context without adding new facts. This preserves determinism and auditability while improving readability.

## **2. Document-Based RAG Architecture**

### **High-Level Architecture Description**

At a high level, the Hybrid RAG architecture consists of three main components:

1. **Large Language Model (LLM)**: Generates responses from retrieved context plus the user's question, and is constrained to that context.

2. **Knowledge Stores**: Two distinct storage roles exist:

   * **Graph Evidence Stores (LT + HOT)**:
     * **Long-Term (LT)** holds durable, vetted knowledge in a graph form: documents, chunks, entities, and explicit relationships.
     * **HOT (unstable)** holds volatile, user-specific, or experimental graph evidence.
     * Both stores preserve provenance metadata (source identifiers, ingestion timestamps, versions).
     * Retrieval uses explicit, inspectable policies (entity overlap, document anchoring, optional neighbor expansion).

   * **Vector Index (Semantic Support)**:
     * Holds sliding-window text chunks embedded for semantic kNN retrieval.
     * Used to improve recall for paraphrases and add supporting context.
     * Typically filtered to graph-anchored documents to prevent semantic drift.

3. **Integration Layer (Middleware)**:
   * Connects the LLM, the NER service, the graph stores, and the vector index.
   * For each question:
     1) extracts normalized entities via NER,
     2) retrieves graph evidence (LT + HOT in parallel),
     3) optionally retrieves vector context (preferably anchored),
     4) prepares LLM prompts that keep graph grounding authoritative and vectors as support.

![Generic Hybrid RAG Implementation](./images/reinforcement_learning.png)

In this implementation, the graph is not an abstract "knowledge representation theory project." It is a **retrieval substrate**: explicit nodes, explicit edges, explicit ranking rules, and explicit evidence chunks that can be cited. Vectors augment retrieval by improving semantic recall, but factual grounding remains graph-first.

Overall, the design marries an LLM's generation with transparent retrieval. You tune behavior by deciding what lives in LT vs HOT, how NER enrichments are stored, and how vector retrieval is constrained. Next, we outline how the two stores work together to strengthen governance.

### **HOT vs. Long-Term Roles**

The architecture separates HOT and LT to optimize governance, provenance, and operational hygiene:

* **HOT (unstable)**: A store for **volatile graph evidence**: user-generated, experimental, or unverified content. HOT is optimized for write churn and policy-controlled lifecycle.
* **Long-Term (LT)**: The durable, vetted repository. Evidence is ingested with NER enrichments and provenance metadata. **Promotion from HOT → LT occurs only when** (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

### **Benefits of Hybrid RAG**

Adopting this Hybrid RAG architecture provides several distinct advantages:

* **Structured Knowledge Representation**: Entities and provenance metadata give structure to unstructured text and enable precise, auditable filters and explanations.
* **Deterministic Retrieval**: Graph evidence is retrieved from LT and HOT in parallel using explicit policies (entity overlap, anchors, neighborhood expansion).
* **Reduced Hallucinations, Improved Accuracy**: Answers are grounded in graph evidence; vectors are used only for clarifications or phrasing support.
* **Transparency and Traceability**: Observability tooling can emit full retrieval policies, per-store results, and final answers for audit purposes.
* **Open-Source Flexibility**: Built with common open tooling (graph DB + vector indexing + Python middleware); customizable without vendor lock-in.

In summary, this Hybrid RAG approach combines explicit grounding with semantic augmentation to deliver governance-friendly answers. The next sections show how these choices increase explainability and how the system behaves in practice.

### **Enhancing Transparency and Explainability**

Transparency is built in and observable end-to-end:

* **Documented Evidence**: Every answer links back to specific evidence chunks. Graph evidence is always the authoritative source for factual claims.
* **Metadata Annotations**: NER outputs are stored explicitly (entity lists and/or mention edges), making retrieval explainable in human terms.
* **Explicit Retrieval Logic**: The integration layer issues structured graph retrieval steps (anchors + evidence chunks), not opaque similarity-only ranking.
* **Audit Trails**: Provenance fields and store stamps provide a clear trail from question → entities → per-store retrieval → answer. HOT → LT promotion events are discrete, reviewable steps.

Reasoning is externalized: we can map query → retrieved evidence → answer without relying on opaque similarity scores. This is useful for regulated domains where reviewers must see and verify the chain of custody.

### **Visualizing the Architecture (Referencing Diagram)**

To conceptualize this, picture two evidence stores and an orchestrator:

* The **Orchestrator** receives a question, calls the **NER service**, and **queries LT and HOT graph stores in parallel**.
* It optionally runs **vector kNN** against the embedding index, filtering to graph-anchored documents where possible.
* The **LLM** receives graph evidence as the authoritative grounding context, and vector context as optional semantic support.
* **Governance policy**: HOT → LT promotion happens **only** with sufficient positive reinforcement or explicit human verification.

Unlike vector-only RAG, this dual-store, graph-first design protects provenance and limits blast radius while keeping semantic augmentation constrained and auditable.

## **3. HOT (unstable) Store**

### **Overview of HOT**

HOT in this graph-grounded RAG system is **not** a chat transcript. It is an **evidence store** for volatile, user-specific, or experimental facts represented as:

* documents and chunks
* extracted entities
* explicit relationships linking chunks to entities (and documents to chunks)

Relevance for retrieval is driven by NER-extracted entities from the user's question. The integration layer uses those entities to run **explicit graph retrieval steps** and **queries LT and HOT in parallel**.

HOT is optimized for speed and policy control. Keeping HOT small means graph queries return relevant evidence quickly and are easy to inspect. LT remains the source of truth.

### **Implementation Details**

Implementing HOT centers on how evidence is **written**, **queried**, and **pruned**:

* **Graph schema and constraints**: Enforce stable identifiers for documents and chunks, and uniqueness for entity names. Stable keys make ingests idempotent and retrieval reproducible.
* **NER enrichment**: NER runs as a separate service. At ingest time, entities are attached to evidence (stored as explicit lists and/or entity mention edges). At question time, NER extracts query entities and drives grounding retrieval.
* **Parallel retrieval**: The orchestrator runs the same retrieval policy against LT and HOT, then merges results deterministically (LT-preference is recommended when conflicts arise).
* **Lifecycle management**: HOT should be pruned using explicit policy (TTL, eviction windows, or operator actions). Graph pruning is typically implemented via scheduled Cypher deletes based on timestamps/versions or per-tenant boundaries.
* **HOT → LT promotion policy**: Promotion **from HOT to LT** happens **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data.

### **Performance Considerations and Optimization**

HOT must respond quickly under load:

* **Keep it small**: Small HOT graphs produce faster match/collect steps and lower memory pressure.
* **Index the right things**: Index entity names and document/chunk keys. Optional full-text indexes can support explainable fallback when entity extraction is sparse.
* **Query shape**: Prefer entity overlap scoring (distinct matched entities), optionally anchored by top documents, and optionally expanded with neighbor windows for local context continuity.
* **Isolation**: HOT can run with different performance posture (fewer durability guarantees, higher write churn tolerance) than LT.
* **Maintenance**: Run pruning on a schedule and cap per-run deletions to avoid spiky load.

Applied together, these optimizations keep HOT responsive while maintaining governance-friendly evidence.

### **Benefits of HOT**

A HOT layer improves both operations and governance:

* **Low-Latency Serving:** Small working sets return entity-relevant evidence quickly.
* **Deterministic Hygiene:** Lifecycle policy keeps volatile evidence from lingering.
* **Explainable Context:** Entity-driven retrieval makes it clear **why** evidence was selected.
* **Governance by Design:** LT remains authoritative; HOT enables rapid iteration without corrupting audited content. HOT → LT promotion remains explicitly gated.

## **4. Long-Term Memory**

### **Overview of Long-Term Memory**

Long-term memory is the persistent knowledge foundation of the Hybrid RAG architecture. This is where the system's accumulated information, expected to remain relevant over time, is stored. Unlike **HOT (unstable)**, long-term memory contains data that doesn't expire on a timer... it stays until updated or removed deliberately.

Some characteristics of long-term memory:

* **It is comprehensive:** The store covers a wide range of documents: manuals, knowledge articles, policies, historical records, and operational runbooks.
* **It is structured for retrieval:** LT stores evidence as graph objects (documents/chunks/entities/edges) with deterministic keys and provenance metadata. This supports reproducible retrieval and auditable evidence chains.
* **It ensures consistency and accuracy:** The LT store is curated via a controlled ingest path that enriches with external NER and assigns stable identifiers plus versioning.
* **It provides historical context:** LT holds enduring documents and facts, not conversational state.
* **It scales technically:** Graph stores scale via indexing, partitioning strategies, clustering, and careful query design. The evidence model stays minimal to reduce operational burden.
* **It evolves with time:** LT can be updated via re-ingest or controlled mutations; version fields and timestamps support governance and replay.

In essence, long-term memory acts as the AI's body of record. It complements HOT by providing stability, provenance, and breadth.

### **Integration with HOT**

The interaction between long-term and HOT is what gives the system its power:

* **During Query Processing:** The orchestrator extracts entities from the user's question (using NER), runs graph retrieval against **LT and HOT in parallel**, and optionally runs vector kNN retrieval for semantic support (preferably constrained to graph anchors).
* **Promotion from HOT → LT** occurs **only** when (1) there is **enough positive reinforcement** of the data **or** (2) a **trusted human-in-the-loop** has verified the data. Long-term remains authoritative.
* **Data Consistency:** Conflicts resolve to LT as the source of truth (recommended posture).
* **Multi-Store Search:** Results can be merged deterministically across stores with a clear precedence policy.

### **Performance and Scalability Considerations**

Long-term memory contains most of the data, so scale and steadiness matter:

* **Scalability:** Use indexing and clustering where needed. Keep the schema stable and evidence model minimal.
* **Ingestion throughput:** Batch ingests and idempotent merges preserve reproducibility.
* **Resource management:** Size memory and I/O for predictable read latency and safe ingest concurrency.
* **Backup and recovery:** Snapshots/replication protect LT and enable point-in-time audits ("what did the system know then?").
* **Monitoring and optimization:** Track query latency, memory usage, and index performance; tune constraints and query patterns based on workload.
* **Security and multitenancy:** Enforce role-based access and tenant boundaries as required.

Treat the LT graph store like a production knowledge system: stable keys, predictable query patterns, and disciplined change control.

## **5. AI Governance Improvements**

Effective AI governance means ensuring that AI systems operate in a manner that is transparent, fair, accountable, and safe, while adhering to relevant laws and ethical standards. The Hybrid RAG architecture described above offers concrete improvements in each of these areas by design.

### **Transparency and Explainability**

The system links each answer back to specific evidence chunks retrieved from the graph stores. Retrieval is driven by explicit entity overlap rules and optional anchoring, and LT and HOT are queried in parallel. Vector context is added only as support and is constrained to graph anchors when possible. This preserves explainability while still benefiting from semantic coverage.

### **Fairness and Bias Mitigation**

Fairness starts with curation of the LT corpus and visibility into retrieval drivers. Because entity extraction and graph retrieval steps are explicit, teams can audit which entities drive evidence selection, detect over-reliance on particular categories/sources, and adjust data or policies accordingly.

### **Accountability and Responsibility**

Every critical step is loggable in plain terms:

* extracted entities from NER
* retrieval queries/parameters used for graph anchoring and evidence selection
* per-store results (LT vs HOT)
* final selected evidence and citations

These artifacts provide a replayable transaction log for governance review.

### **Data Governance**

Lifecycle control is built in. LT is durable and versioned; HOT is volatile and governed by pruning policies. Stable identifiers and provenance fields make schema validation and data hygiene operational. **HOT → LT promotion occurs only** when there is sufficient positive reinforcement or a trusted human reviewer verifies the data.

### **Regulatory Compliance and Standards**

This design supports the controls regulators care about:

* evidence traceability
* reproducibility
* precise deletion (for correction/erasure workflows)
* separation of vetted vs unvetted stores
* point-in-time auditing via backups/snapshots of LT

### **Risk Management and Safety**

Graph grounding reduces hallucinations because factual claims must map back to explicit evidence. Vector context improves recall and readability but remains additive. HOT policies curb stale or unverified evidence from persisting beyond its governance window. When issues occur, saved observability artifacts accelerate root-cause analysis.

Hybrid Search for AI governance takes the mystery out of retrieval. Two stores exist primarily for **governance boundaries and retention policy control**, not for magic performance gains... so trust and operational discipline improve together.

## **6. Target Audience and Use Cases**

Hybrid RAG with graph grounding is a flexible architecture that serves multiple stakeholders. Below we outline the primary audiences and concrete use cases aligned with this **dual-store, graph-grounded, vector-augmented** design.

### **Open-Source Engineers**

Builders who value transparency, composability, and zero vendor lock-in.

* **Why it matters**
  Everything is inspectable: deterministic keys, explicit entity extraction, explicit evidence selection, and clear separation of grounding vs support.
* **Use case**
  A programming assistant ingests manuals and API docs into LT. At question time, entities drive graph evidence retrieval; vectors improve semantic coverage for paraphrases. HOT can hold user uploads and experiments without corrupting LT.

### **Enterprise Architects**

Leaders who must integrate AI into existing estates with guardrails for scale, security, and compliance.

* **Why it matters**
  Graph evidence makes provenance and relationship reasoning explicit. Dual-store separation supports policy asymmetry (strict LT, flexible HOT).
* **Use case**
  A policy assistant for a regulated firm grounds answers in LT graph evidence and uses vectors only as support. HOT holds short-lived user artifacts and quarantined content pending review.

### **NetApp Infrastructure Customers**

Teams standardizing on ONTAP who want storage-level reliability with application-level transparency.

* **Why it matters**
  The architecture maps cleanly to storage controls: durable LT snapshots/replication, fast HOT working sets, explicit promotion gates.
* **Use case**
  A support assistant with LT protected by snapshots and replication, HOT tuned for churn, and retrieval logs usable for compliance review.

### **Cross-Industry Applicability**

The pattern stays the same; only the corpus and NER tuning change:

* **Healthcare**: clinical guidelines grounded in vetted LT evidence; HOT quarantines newly uploaded notes pending review.
* **Retail**: product specs and policies grounded in LT; vectors help with fuzzy phrasing and long-tail queries.
* **Legal**: statutes and rulings grounded in LT evidence chains; retrieval remains explainable and replayable.

Across audiences, the benefits are consistent: **explainability, determinism, and operational control**. The dual-store design proves where answers came from, separates durable truth from volatile experiments, and turns policy enforcement into configuration.

## **7. Implementation Guide**

Please see:

- [Community / Open Source Version](./OSS_Community_Version.md)  
  For a reference implementation, please check out: [community_version/README.md](./community_version/README.md)

- [Enterprise Version](./Enterprise_Version.md)  
  For a reference implementation, please check out: [enterprise_version/README.md](./enterprise_version/README.md)

## **8. Conclusion**

This Hybrid RAG architecture moves AI retrieval from opaque heuristics to **observable, governable evidence selection**.

Pairing an LLM with **graph-grounded retrieval** and **vector semantic augmentation** blends generation with verifiable evidence. Queries run against **both Long-Term (LT) and HOT (unstable)** in parallel, graph evidence is gathered from both stores, and vector context is added in a controlled, anchored way... maintaining reliability, transparency, and compliance that vector-only stacks struggle to provide.

Knowledge is treated as a first-class asset:

* **LT** is the vetted, durable evidence store with stable identifiers and provenance.
* **HOT (unstable)** is an operational evidence tier governed by lifecycle policy.
* **HOT → LT promotion happens only** when there is sufficient **positive reinforcement** of the data **or** a **trusted human-in-the-loop** has verified it.

Transparency is built in. Answers are grounded in retrievable evidence with explicit citation blocks and auditable retrieval policies. Observability controls make the path from **question → entities → per-store evidence → answer** explainable and reproducible for reviewers and auditors.

Finally, this architecture aligns with enterprise governance. The split between stores exists for **governance boundaries, retention control, and policy asymmetry**. Using explicit entities, deterministic evidence selection, and controlled lifecycle policies, the system meets accountability and regulatory needs without slowing delivery. Built on mature open tooling, it's practical, scalable, and cost-effective.

Hybrid RAG proves powerful AI can be both **capable and accountable**.
