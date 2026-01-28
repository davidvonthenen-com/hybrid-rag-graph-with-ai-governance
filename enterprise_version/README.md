# Enterprise Deployment: Graph-Based Hybrid RAG with NetApp Storage Integrations

This document outlines the **infrastructure deployment strategy** for the Graph-based Hybrid RAG system. It assumes the use of the application logic found in the [Community Version](../community_version/README.md) and deploys it with NetApp enterprise data services for **high availability**, **QoS isolation**, and **governance controls**.

The core difference vs. the Enterprise deployment is the data plane:

- **Truth grounding (Graph):** Neo4j (Knowledge Graphs)
- **Contextual awareness (Vector):** OpenSearch (kNN / vector index)

By decoupling the compute (Neo4j/OpenSearch/LLM) from the storage layer, we achieve strict Service Level Agreements (SLAs), distinct hardware QoS for "HOT" vs. "Long-Term" memory, and automated compliance governance without changing the core application code.

## Architecture: The Storage Overlay

While the application logically sees **two graph endpoints** (Neo4j LONG + Neo4j HOT) and a **vector endpoint** (OpenSearch), the physical storage layer is architected to meet specific access patterns using NetApp ONTAP technologies.

| Logical Tier | Application Role | Physical Storage Configuration | NetApp Technology |
| --- | --- | --- | --- |
| **Long-Term Memory** | Authoritative Source of Truth; Read-Heavy; Compliance Required. | **Graph DB (Neo4j LONG):** High-throughput NFS.<br><br>**Vector DB (OpenSearch):** Durable shard storage with tiering.<br><br>**Protection:** Zero-RPO replication + WORM options. | **MetroCluster** (HA)<br><br>**FabricPool** (Tiering)<br><br>**SnapLock** (WORM) |
| **HOT Memory** | Unstable/Working Set; Mixed Read-Write; Low Latency. | **Graph DB (Neo4j HOT):** In-memory speed with WAN caching.<br><br>**Isolation:** IOPS guarantees for user-driven updates/queries. | **FlexCache**<br><br>**Storage QoS**<br><br>**FlexClone** |

> **Important framing:** HOT vs. LONG is a policy boundary for the **graph truth store**. The vector store can be implemented as a single OpenSearch deployment (recommended for operational simplicity) with filtering/metadata separation, or as distinct indices/tiers depending on your SLA and tenant isolation requirements.

## 1. Long-Term Memory Configuration (Authoritative Store)

The Long-Term memory contains the vetted Knowledge Base. As this dataset grows to petabyte scale, keeping it entirely on high-performance SSDs is cost-prohibitive, yet it must remain instantly accessible for retrievals.

In the Graph-based Hybrid RAG deployment, Long-Term memory primarily means:

- **Neo4j LONG** holds curated **Document/Chunk/Entity** evidence used for truth grounding.
- **OpenSearch** holds the **vector chunk index** used for semantic support.

### Automated Cost Optimization (Auto-tiering)

We utilize **NetApp Auto-tiering** for the storage backing the Long-Term tier:

- **Neo4j LONG database files** (store files, transaction logs, backups) live on a durable NFS-backed volume.
- **OpenSearch vector shards** for the vector index benefit from tiering as older segments become cold.

Key behaviors mirror the community version approach, but now apply to the graph + vector pair:

- **Cold Data Movement:** Data blocks not accessed for a configurable period are automatically moved to lower-cost object storage tiers.
- **Transparent Retrieval:** The application remains unaware of data movement; if the agent needs an old fact or vector segment, it is recalled seamlessly.
- **Intelligent Caching:** Storage detects large sequential reads (segment merging, backups, compaction) and avoids evicting active working sets.

### Ingestion Pipeline (NetApp XCP)

For the initial population of Long-Term memory from legacy data lakes (Hadoop/HDFS or massive NFS shares), standard copy tools are insufficient.

We utilize **NetApp XCP** to:

- **Move source documents** from legacy HDFS/NFS silos into the RAG source volume(s) that the ingestion pipeline reads.
- **Verify integrity** bit-for-bit so the Long-Term knowledge base can be traced back to the authoritative source.

> Graph-based truth grounding gets its governance strength from provenance. If your source ingest is sloppy, your audit story becomes interpretive fiction.

## 2. HOT Memory Configuration (The Working Set)

The HOT memory handles user-specific context, new unverified facts, and high-frequency queries. It requires extremely low latency and isolation from the heavy I/O of Long-Term ingest processes.

In the Graph-based Hybrid RAG deployment, HOT memory primarily means:

- **Neo4j HOT** holds unvetted **Document/Chunk/Entity** evidence used for truth grounding of new facts.
- The **vector store** remains available for semantic support; HOT-vs-LONG separation can be expressed as metadata filters, per-index partitioning, or a separate OpenSearch tier if required.

### Global Locality (FlexCache)

In distributed inference clusters (e.g., Kubernetes nodes across zones), we back the **Neo4j HOT** store with a **NetApp FlexCache** volume:

- **Burst Performance:** Brings the active working set close to inference compute.
- **Reduce Latency:** Frequently accessed HOT evidence stays local, improving end-user responsiveness.

### Noisy Neighbor Isolation (Storage QoS)

Ingest jobs and rebuild operations can saturate bandwidth. To protect user experience on the HOT tier:

- **Min/Max IOPS:** Apply **Storage QoS** to HOT volumes to guarantee minimum throughput for HOT queries and updates.
- **Policy asymmetry:** HOT stays fast and writable; LONG stays durable and compliant.

### Migration & Re-indexing (Hot Tier Bypass)

When re-indexing vectors or rebuilding graph state at scale, bulk admin operations can flood caches and evict real working sets.

Operational pattern:

- Route bulk graph rebuild writes and vector re-index writes to volumes/tiering policies that avoid polluting HOT cache locality.
- Use clone-based workflows (FlexClone) for test/dev rebuilds so production HOT remains stable.

## 3. Data Protection and Governance

### Zero Data Loss (MetroCluster)

For the Long-Term memory, we utilize **NetApp MetroCluster**. This provides synchronous replication, ensuring that even in the event of a total site failure, the authoritative Knowledge Base is preserved with **Zero RPO**, allowing the agent to resume operations immediately.

In this Graph-based architecture, MetroCluster primarily protects:

- **Neo4j LONG** database volumes (graph truth grounding source of record)
- **OpenSearch vector index** volumes (semantic support index)

### Compliance Snapshots (SnapCenter & SnapLock)

- **SnapCenter:** Used to take application-consistent snapshots of:
  - **Neo4j LONG** database volumes (graph truth grounding state)
  - **OpenSearch** vector indices (exact semantic context available at a point in time)
- **SnapLock:** For strict regulatory environments, SnapLock enforces WORM (Write Once, Read Many) protection on Long-Term volumes, proving the knowledge base has not been tampered with.

The outcome is an immutable audit trail of exactly what the AI "knew" at a specific time, across both truth grounding (graph) and semantic support (vectors).

## 4. Deployment Configuration

The following Docker configuration demonstrates how to map the **Neo4j** and **OpenSearch** containers to the specific NetApp volumes described above. This replaces the standard local volume mapping from the Community Version.

```bash
# create a docker network so that opensearch instances and dashboards can access each other
docker network create opensearch-net

# ---------------------------------------------------------
# VECTOR DATABASE (OpenSearch)
# Backed by NetApp Volume with Tiering & Snapshots
# ---------------------------------------------------------
docker run -d \
  --name opensearch-vector \
    --network opensearch-net \
  -p 9201:9200 -p 9601:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  # Map vector shard data to the NetApp-backed volume
  -v "/mnt/netapp_vector_vol/data:/usr/share/opensearch/data" \
  # Map snapshots to a SnapLock compliance volume (optional)
  -v "/mnt/netapp_snaplock_vol/snapshots:/mnt/snapshots" \
  opensearchproject/opensearch:3.2.0

# Dashboards (Standard configuration)
docker run -d \
  --name opensearch-vector-dashboards \
  --network opensearch-net \
  -p 5601:5601 \
  -e 'OPENSEARCH_HOSTS=["http://opensearch-vector:9200"]' \
  -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
  opensearchproject/opensearch-dashboards:3.2.0


# create a docker network so that neo4j instances and dashboards can access each other
docker network create graph-net

# ---------------------------------------------------------
# LONG-TERM MEMORY (Truth Grounding - Neo4j LONG)
# Backed by NetApp Volume with MetroCluster & WORM options
#
# API: localhost:7688
# Admin Panel and API: http://127.0.0.1:7475
# Username: neo4j
# Password: neo4jneo4j1
# ---------------------------------------------------------
docker run -d \
    --name neo4j-long-term \
    --network graph-net \
    -p 7475:7474  -p 7688:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j1 \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_server_http_advertised__address="localhost:7475" \
    -e NEO4J_server_bolt_advertised__address="localhost:7688" \
    -v "/mnt/netapp_longterm_graph_vol/data":/data \
    -v "/mnt/netapp_longterm_graph_vol/logs":/logs \
    -v "$HOME/neo4j-long/import":/import \
    -v "$HOME/neo4j-long/plugins":/plugins \
    neo4j:5.26.16


# ---------------------------------------------------------
# HOT MEMORY (Truth Grounding - Neo4j HOT)
# Backed by FlexCache for low-latency locality + QoS isolation
#
# API: localhost:7689
# Admin Panel and API: http://127.0.0.1:7476
# Username: neo4j
# Password: neo4jneo4j2
# ---------------------------------------------------------
docker run -d \
    --name neo4j-short-term \
    --network graph-net \
    -p 7476:7474 -p 7689:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j2 \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_server_http_advertised__address="localhost:7476" \
    -e NEO4J_server_bolt_advertised__address="localhost:7689" \
    -v "/mnt/netapp_flexcache_hot_graph/data":/data \
    -v "/mnt/netapp_flexcache_hot_graph/logs":/logs \
    -v "$HOME/neo4j-short/import":/import \
    -v "$HOME/neo4j-short/plugins":/plugins \
    neo4j:5.26.16
```

The container images are standard. The enterprise value is in the storage mapping:

- LONG-term volumes are protected by MetroCluster and governed by SnapLock/SnapCenter.
- HOT volumes are accelerated via FlexCache and protected from noisy neighbors via Storage QoS.

That's the whole trick: the Graph-based Hybrid RAG code stays the same, but the storage substrate stops behaving like a developer laptop and starts behaving like an enterprise platform.

