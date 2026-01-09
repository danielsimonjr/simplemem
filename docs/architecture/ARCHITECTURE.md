# SimpleMem Architecture Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Core Architecture](#core-architecture)
4. [Three-Stage Pipeline](#three-stage-pipeline)
5. [Data Models](#data-models)
6. [Component Deep Dive](#component-deep-dive)
7. [Storage Layer](#storage-layer)
8. [LLM Integration](#llm-integration)
9. [Parallel Processing](#parallel-processing)
10. [Configuration System](#configuration-system)
11. [Evaluation Framework](#evaluation-framework)
12. [Design Decisions](#design-decisions)

---

## Executive Summary

SimpleMem is a research implementation of an efficient long-term memory system for LLM agents, based on the paper "SimpleMem: Efficient Lifelong Memory for LLM Agents" (arXiv:2601.02553). The system addresses the fundamental challenge of maintaining efficient, accurate memory across extended conversations while minimizing token usage and computational overhead.

### Key Innovations

1. **Semantic Lossless Compression**: Transforms raw dialogues into self-contained atomic facts with resolved coreferences and absolute timestamps
2. **Multi-View Indexing**: Indexes memories across semantic (dense vectors), lexical (BM25 keywords), and symbolic (metadata) layers
3. **Adaptive Query-Aware Retrieval**: Uses complexity-aware pruning and planning-based multi-query decomposition
4. **Write-Time Disambiguation**: Eliminates downstream reasoning overhead by resolving ambiguity at memory creation time

### Performance Characteristics

- **43.24% F1 Score** on LoCoMo-10 benchmark (vs 34.20% for Mem0)
- **480.9s total processing time** (12.5x faster than A-Mem)
- **~550 tokens** per query (30x fewer than full-context methods)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SimpleMem System                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         SimpleMemSystem                               │   │
│  │                          (main.py)                                    │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │   │
│  │  │ LLMClient   │  │ EmbeddingModel   │  │     VectorStore        │   │   │
│  │  │ (utils/)    │  │ (utils/)         │  │     (database/)        │   │   │
│  │  └──────┬──────┘  └────────┬─────────┘  └───────────┬────────────┘   │   │
│  │         │                  │                        │                │   │
│  │         v                  v                        v                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Core Pipeline                             │    │   │
│  │  │  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │    │   │
│  │  │  │ MemoryBuilder │→ │ HybridRetriever │→ │AnswerGenerator│  │    │   │
│  │  │  │   (Stage 1)   │  │    (Stage 3)    │  │   (Stage 3)   │  │    │   │
│  │  │  └───────────────┘  └─────────────────┘  └───────────────┘  │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                    │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │   │
│  │  │  Dialogue   │  │   MemoryEntry    │  │      LanceDB           │   │   │
│  │  │  (models/)  │  │    (models/)     │  │  (vector storage)      │   │   │
│  │  └─────────────┘  └──────────────────┘  └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Organization

```
simplemem/
├── main.py                 # SimpleMemSystem - main orchestrator
├── config.py.example       # Configuration template
├── core/                   # Core pipeline components
│   ├── __init__.py
│   ├── memory_builder.py   # Stage 1: Semantic Structured Compression
│   ├── hybrid_retriever.py # Stage 3: Adaptive Query-Aware Retrieval
│   └── answer_generator.py # Stage 3: Reconstructive Synthesis
├── database/
│   └── vector_store.py     # LanceDB-backed multi-view indexing
├── models/
│   ├── __init__.py
│   └── memory_entry.py     # MemoryEntry and Dialogue data classes
├── utils/
│   ├── __init__.py
│   ├── llm_client.py       # OpenAI-compatible LLM client
│   └── embedding.py        # SentenceTransformers embedding model
├── test_locomo10.py        # LoCoMo-10 benchmark evaluation
└── test_ref/               # Reference test utilities
    ├── load_dataset.py     # Dataset parsing
    ├── test_advanced.py    # Advanced testing (external memory systems)
    └── utils.py            # Evaluation metrics
```

---

## Core Architecture

### Dependency Graph

```
                    ┌─────────────────────┐
                    │   SimpleMemSystem   │
                    │      (main.py)      │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           v                   v                   v
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │MemoryBuilder│    │HybridRetriever│    │AnswerGen   │
    └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
           │                  │                   │
           └──────────┬───────┴───────────────────┘
                      │
                      v
              ┌──────────────┐
              │  VectorStore │
              │  (LanceDB)   │
              └──────┬───────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         v           v           v
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │LLMClient│ │Embedding│ │  config │
    └─────────┘ └─────────┘ └─────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Methods |
|-----------|----------------|-------------|
| `SimpleMemSystem` | Orchestration, API facade | `add_dialogue()`, `ask()`, `finalize()` |
| `MemoryBuilder` | Dialogue → Atomic entries | `add_dialogues()`, `process_window()` |
| `HybridRetriever` | Multi-path retrieval | `retrieve()`, `_retrieve_with_planning()` |
| `AnswerGenerator` | Context → Answer synthesis | `generate_answer()` |
| `VectorStore` | Storage and indexing | `add_entries()`, `semantic_search()` |
| `LLMClient` | LLM API communication | `chat_completion()`, `extract_json()` |
| `EmbeddingModel` | Vector embeddings | `encode()`, `encode_query()` |

---

## Three-Stage Pipeline

### Overview

The SimpleMem pipeline implements a three-stage process based on the paper's Semantic Lossless Compression framework:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WRITE PATH (Memory Building)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Dialogues                                                               │
│       │                                                                      │
│       v                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              STAGE 1: Semantic Structured Compression                │    │
│  │                                                                      │    │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │    │
│  │   │ Φ_extract   │ → │  Φ_coref    │ → │      Φ_time          │    │    │
│  │   │ (extraction)│    │(coreference)│    │(temporal anchoring) │    │    │
│  │   └─────────────┘    └─────────────┘    └─────────────────────┘    │    │
│  │                                                                      │    │
│  │   De-linearization: F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       v                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              STAGE 2: Structured Multi-View Indexing                 │    │
│  │                                                                      │    │
│  │   M(m_k) = { v_k (semantic), h_k (lexical), R_k (symbolic) }        │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │    │
│  │   │ Semantic     │  │ Lexical      │  │ Symbolic             │     │    │
│  │   │ Dense Vector │  │ BM25 Keywords│  │ Metadata Constraints │     │    │
│  │   │ (1024-d)     │  │              │  │ (time, persons, etc.)│     │    │
│  │   └──────────────┘  └──────────────┘  └──────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       v                                                                      │
│  LanceDB Storage                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        READ PATH (Query Processing)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Question                                                               │
│       │                                                                      │
│       v                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              STAGE 3: Adaptive Query-Aware Retrieval                 │    │
│  │                                                                      │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Query Complexity Estimation (C_q)                            │   │    │
│  │   │   → Analyze information requirements                         │   │    │
│  │   │   → Determine optimal retrieval depth k_dyn                  │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                          │                                          │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Planning-Based Multi-Query Generation                        │   │    │
│  │   │   → Generate targeted sub-queries                            │   │    │
│  │   │   → Execute parallel semantic search                         │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                          │                                          │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Reflection-Based Additional Retrieval (Optional)             │   │    │
│  │   │   → Check answer adequacy                                    │   │    │
│  │   │   → Generate additional queries if insufficient              │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                          │                                          │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Reconstructive Synthesis                                     │   │    │
│  │   │   C_final = ⊕_{m ∈ Top-k_dyn(S)} [t_m: Content(m)]          │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       v                                                                      │
│  Answer                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Semantic Structured Compression

**Location**: `core/memory_builder.py`

**Purpose**: Transform raw, ambiguous dialogue streams into atomic entries—self-contained facts with resolved coreferences and absolute timestamps.

**Key Transformation** (De-linearization F_θ):
```
F_θ(W_t) = Φ_time ∘ Φ_coref ∘ Φ_extract(W_t)
```

Where:
- **Φ_extract**: Extract information from dialogue window
- **Φ_coref**: Force coreference resolution (eliminate pronouns: "he", "she", "it")
- **Φ_time**: Temporal anchoring (convert "tomorrow" to absolute dates)

**Example**:
```
Input:  [2025-11-15T14:30:00] Alice: "He'll meet Bob tomorrow at 2pm"
Output: "Alice will meet Bob at Starbucks on 2025-11-16T14:00:00"
```

**Processing Flow**:

```
Dialogues → add_dialogue() → dialogue_buffer
                                    │
                    when len(buffer) >= window_size
                                    │
                                    v
                          process_window()
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
            Sequential Mode              Parallel Mode
                     │                             │
                     v                             v
           _generate_memory_entries()    _process_windows_parallel()
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    v
                            LLM Extraction
                      (JSON-structured output)
                                    │
                                    v
                         List[MemoryEntry]
                                    │
                                    v
                      VectorStore.add_entries()
```

**LLM Prompt Structure**:

The memory builder uses a carefully designed prompt that:
1. Enforces complete coverage of all information
2. Prohibits pronouns and relative time expressions
3. Requires ISO 8601 timestamps
4. Extracts structured metadata (keywords, persons, entities, location, topic)

### Stage 2: Structured Multi-View Indexing

**Location**: `database/vector_store.py`

**Purpose**: Index memories across three orthogonal dimensions for robust retrieval.

**Indexing Layers**:

| Layer | Type | Storage | Purpose |
|-------|------|---------|---------|
| **Semantic** | Dense | 1024-d vectors | Conceptual similarity |
| **Lexical** | Sparse | Keywords list | Exact term matching |
| **Symbolic** | Metadata | Structured fields | Hard filtering constraints |

**Mathematical Representation**:
```
M(m_k) = { v_k, h_k, R_k }

where:
  v_k = E_dense(S_k)           # Dense embedding
  h_k = Sparse(S_k)            # Keyword extraction
  R_k = {(key, val)}           # Metadata key-value pairs
```

**Schema (LanceDB)**:
```python
schema = pa.schema([
    pa.field("entry_id", pa.string()),
    pa.field("lossless_restatement", pa.string()),  # Semantic layer base
    pa.field("keywords", pa.list_(pa.string())),     # Lexical layer
    pa.field("timestamp", pa.string()),              # Symbolic layer
    pa.field("location", pa.string()),               # Symbolic layer
    pa.field("persons", pa.list_(pa.string())),      # Symbolic layer
    pa.field("entities", pa.list_(pa.string())),     # Symbolic layer
    pa.field("topic", pa.string()),                  # Symbolic layer
    pa.field("vector", pa.list_(pa.float32(), 1024)) # Dense embedding
])
```

### Stage 3: Adaptive Query-Aware Retrieval

**Location**: `core/hybrid_retriever.py`, `core/answer_generator.py`

**Purpose**: Retrieve optimal context with complexity-aware depth modulation.

**Retrieval Flow**:

```
Query → _analyze_information_requirements()
            │
            │  Query Complexity C_q
            v
        _generate_targeted_queries()
            │
            │  k_dyn = k_base · (1 + δ · C_q)
            v
    ┌───────┴───────┐
    │ Parallel      │
    │ Semantic      │
    │ Search        │
    └───────┬───────┘
            │
            v
    _merge_and_deduplicate_entries()
            │
    ┌───────┴───────┐
    │  Optional     │
    │  Reflection   │
    │  Loop         │
    └───────┬───────┘
            │
            v
    AnswerGenerator.generate_answer()
            │
            v
        Answer
```

**Planning Process**:

1. **Information Requirement Analysis**:
   - Classify question type (factual, temporal, relational, explanatory)
   - Identify key entities and relationships
   - Estimate minimal queries needed

2. **Targeted Query Generation**:
   - Generate 1-4 focused queries
   - Each targets specific information requirements
   - Avoid redundant or overlapping queries

3. **Reflection Loop** (Optional):
   - Check if current results satisfy the query
   - Generate additional targeted queries if insufficient
   - Maximum 2 reflection rounds by default

---

## Data Models

### MemoryEntry (Atomic Entry)

**Location**: `models/memory_entry.py`

**Structure**:

```python
class MemoryEntry(BaseModel):
    # Unique identifier
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # [Semantic Layer] - Base for dense embedding
    lossless_restatement: str  # Self-contained fact with resolved references

    # [Lexical Layer] - BM25-style matching
    keywords: List[str]        # Core keywords for exact matching

    # [Symbolic Layer] - Metadata constraints
    timestamp: Optional[str]   # ISO 8601 format
    location: Optional[str]    # Natural language location
    persons: List[str]         # Extracted person names
    entities: List[str]        # Companies, products, etc.
    topic: Optional[str]       # Topic phrase
```

**Example Instance**:
```python
MemoryEntry(
    entry_id="550e8400-e29b-41d4-a716-446655440000",
    lossless_restatement="Alice discussed the marketing strategy for new product XYZ with Bob at Starbucks in Shanghai on November 15, 2025 at 14:30.",
    keywords=["Alice", "Bob", "product XYZ", "marketing strategy", "discussion"],
    timestamp="2025-11-15T14:30:00",
    location="Starbucks, Shanghai",
    persons=["Alice", "Bob"],
    entities=["product XYZ"],
    topic="Product marketing strategy discussion"
)
```

### Dialogue

**Location**: `models/memory_entry.py`

**Structure**:
```python
class Dialogue(BaseModel):
    dialogue_id: int           # Sequential identifier
    speaker: str               # Speaker name
    content: str               # Dialogue text
    timestamp: Optional[str]   # ISO 8601 format (optional)
```

### Data Flow Diagram

```
┌────────────┐
│  Dialogue  │
│ (raw input)│
└─────┬──────┘
      │ add_dialogue()
      v
┌────────────────────────────────────────────────────────┐
│                  dialogue_buffer                        │
│  [Dialogue, Dialogue, Dialogue, ...]                   │
└─────────────────────┬──────────────────────────────────┘
                      │ process_window() when buffer >= window_size
                      v
┌────────────────────────────────────────────────────────┐
│                   LLM Extraction                        │
│  JSON prompt → structured memory entries               │
└─────────────────────┬──────────────────────────────────┘
                      │
                      v
┌────────────────────────────────────────────────────────┐
│              List[MemoryEntry]                          │
│  (atomic, self-contained facts)                        │
└─────────────────────┬──────────────────────────────────┘
                      │ add_entries()
                      v
┌────────────────────────────────────────────────────────┐
│                   VectorStore                           │
│  - Generate embeddings                                  │
│  - Store in LanceDB                                     │
│  - Index across semantic/lexical/symbolic layers       │
└────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### SimpleMemSystem

**Location**: `main.py`

**Role**: Main facade and orchestrator for the entire system.

**Initialization**:
```python
SimpleMemSystem(
    api_key: Optional[str] = None,           # OpenAI API key
    model: Optional[str] = None,             # LLM model name
    base_url: Optional[str] = None,          # Custom API endpoint
    db_path: Optional[str] = None,           # LanceDB storage path
    table_name: Optional[str] = None,        # Memory table name
    clear_db: bool = False,                  # Clear existing data
    enable_thinking: Optional[bool] = None,  # Deep thinking mode
    use_streaming: Optional[bool] = None,    # Streaming responses
    enable_planning: Optional[bool] = None,  # Multi-query planning
    enable_reflection: Optional[bool] = None, # Reflection loops
    max_reflection_rounds: Optional[int] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None
)
```

**Key Methods**:

| Method | Purpose | Stage |
|--------|---------|-------|
| `add_dialogue(speaker, content, timestamp)` | Add single dialogue | Stage 1 |
| `add_dialogues(dialogues)` | Batch add dialogues | Stage 1 |
| `finalize()` | Process remaining buffer | Stage 1 |
| `ask(question)` | Query and get answer | Stage 3 |
| `get_all_memories()` | Debug: retrieve all entries | - |
| `print_memories()` | Debug: print all entries | - |

### MemoryBuilder

**Location**: `core/memory_builder.py`

**Key Attributes**:
- `window_size`: Number of dialogues per processing batch (default: 10)
- `dialogue_buffer`: Accumulates dialogues until window_size reached
- `previous_entries`: Context for avoiding duplication
- `processed_count`: Total dialogues processed

**Processing Modes**:

1. **Sequential Mode** (default for small batches):
   - Process windows one at a time
   - Lower overhead, simpler execution

2. **Parallel Mode** (for large batches):
   - Uses `ThreadPoolExecutor`
   - Configurable `max_parallel_workers`
   - Batch LLM calls for all windows simultaneously

**LLM Retry Mechanism**:
- 3 attempts per extraction
- Handles JSON parsing failures gracefully

### HybridRetriever

**Location**: `core/hybrid_retriever.py`

**Retrieval Strategies**:

| Strategy | Method | Description |
|----------|--------|-------------|
| Semantic | `_semantic_search()` | Dense vector similarity (cosine) |
| Lexical | `_keyword_search()` | BM25-style keyword matching |
| Structured | `_structured_search()` | Metadata filtering (time, persons, etc.) |

**Planning-Based Retrieval Flow**:

```python
def _retrieve_with_planning(self, query, enable_reflection):
    # Step 1: Analyze query complexity
    information_plan = self._analyze_information_requirements(query)

    # Step 2: Generate targeted sub-queries
    search_queries = self._generate_targeted_queries(query, information_plan)

    # Step 3: Execute parallel searches
    if self.enable_parallel_retrieval and len(search_queries) > 1:
        all_results = self._execute_parallel_searches(search_queries)
    else:
        all_results = [self._semantic_search(q) for q in search_queries]

    # Step 4: Merge and deduplicate
    merged_results = self._merge_and_deduplicate_entries(all_results)

    # Step 5: Optional reflection
    if should_use_reflection:
        merged_results = self._retrieve_with_intelligent_reflection(
            query, merged_results, information_plan
        )

    return merged_results
```

**Reflection Loop**:

```
┌─────────────────────────────────────────────────────────┐
│                    Reflection Loop                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  for round in range(max_reflection_rounds):             │
│      │                                                   │
│      v                                                   │
│  ┌────────────────────────────────────────────┐         │
│  │ _analyze_information_completeness()        │         │
│  │   → "complete" | "incomplete" | "no_results"│         │
│  └──────────────────┬─────────────────────────┘         │
│                     │                                    │
│          ┌─────────┴─────────┐                          │
│          │                   │                          │
│     "complete"          "incomplete"                    │
│          │                   │                          │
│          v                   v                          │
│       break         _generate_missing_info_queries()    │
│                              │                          │
│                              v                          │
│                     Execute additional searches          │
│                              │                          │
│                              v                          │
│                     Merge with current results          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### AnswerGenerator

**Location**: `core/answer_generator.py`

**Role**: Synthesize final answer from retrieved atomic contexts.

**Answer Generation Prompt Structure**:
```
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
[Context 1] Content: ... Time: ... Location: ...
[Context 2] Content: ... Persons: ...
...

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase)
3. Answer must be based ONLY on the provided context
4. Dates formatted as 'DD Month YYYY'
5. Return JSON format: {"reasoning": "...", "answer": "..."}
```

### VectorStore

**Location**: `database/vector_store.py`

**Backend**: LanceDB (columnar vector database)

**Search Methods**:

1. **Semantic Search** (`semantic_search`):
   ```python
   def semantic_search(self, query: str, top_k: int = 5):
       query_vector = self.embedding_model.encode_single(query, is_query=True)
       results = self.table.search(query_vector.tolist()).limit(top_k).to_list()
       return [self._row_to_entry(r) for r in results]
   ```

2. **Keyword Search** (`keyword_search`):
   - Scores entries based on keyword overlap
   - +2 points for keyword list match
   - +1 point for text content match

3. **Structured Search** (`structured_search`):
   - Filters by: persons, timestamp_range, location, entities
   - Returns entries matching ALL specified constraints

---

## Storage Layer

### LanceDB Integration

**Configuration**:
- `LANCEDB_PATH`: Default `"./lancedb_data"`
- `MEMORY_TABLE_NAME`: Default `"memory_entries"`

**Table Operations**:

| Operation | Method | Description |
|-----------|--------|-------------|
| Initialize | `_init_table()` | Create schema if not exists |
| Add | `add_entries(entries)` | Batch insert with embeddings |
| Search | `semantic_search()` | Vector similarity search |
| Clear | `clear()` | Drop and recreate table |
| Get All | `get_all_entries()` | Retrieve entire table |

**Schema Details**:

```
┌──────────────────────────────────────────────────────────┐
│                    memory_entries table                   │
├───────────────────────┬──────────────────────────────────┤
│ Column                │ Type                             │
├───────────────────────┼──────────────────────────────────┤
│ entry_id              │ string (UUID)                    │
│ lossless_restatement  │ string                           │
│ keywords              │ list<string>                     │
│ timestamp             │ string (ISO 8601)                │
│ location              │ string                           │
│ persons               │ list<string>                     │
│ entities              │ list<string>                     │
│ topic                 │ string                           │
│ vector                │ list<float32>[1024]              │
└───────────────────────┴──────────────────────────────────┘
```

---

## LLM Integration

### LLMClient

**Location**: `utils/llm_client.py`

**Supported Backends**:
- OpenAI API (default)
- Qwen/Alibaba DashScope
- Azure OpenAI
- Any OpenAI-compatible endpoint

**Features**:

1. **Streaming Support**:
   ```python
   if self.use_streaming:
       return self._handle_streaming_response(**kwargs)
   ```

2. **Deep Thinking Mode** (Qwen-specific):
   ```python
   if is_qwen_api:
       if self.use_streaming and self.enable_thinking:
           kwargs["extra_body"] = {"enable_thinking": True}
   ```

3. **Retry Mechanism**:
   - 3 attempts with exponential backoff (1s, 2s, 4s)
   - Handles transient API failures

4. **JSON Extraction**:
   - Handles multiple formats: pure JSON, ```json blocks, embedded JSON
   - Cleans trailing commas, removes comments
   - Finds balanced JSON objects/arrays

### EmbeddingModel

**Location**: `utils/embedding.py`

**Supported Models**:

| Model Type | Example | Dimension |
|------------|---------|-----------|
| Qwen3 Embedding | `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| SentenceTransformers | `all-MiniLM-L6-v2` | 384 |

**Query vs Document Encoding**:

Qwen3 models support query-specific prompts for asymmetric retrieval:
```python
def encode_query(self, queries: List[str]) -> np.ndarray:
    return self.encode(queries, is_query=True)  # Uses "query" prompt

def encode_documents(self, documents: List[str]) -> np.ndarray:
    return self.encode(documents, is_query=False)  # No prompt
```

**Optimization**:
- Flash Attention 2 support (when available)
- Left padding for batch efficiency
- Automatic fallback to standard mode

---

## Parallel Processing

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Parallel Processing Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Memory Building (MemoryBuilder)                                             │
│  ────────────────────────────────                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ add_dialogues_parallel()                                             │    │
│  │                                                                      │    │
│  │   dialogues → split into windows → ThreadPoolExecutor               │    │
│  │                                                                      │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │    │
│  │   │  Window 1   │  │  Window 2   │  │  Window N   │                │    │
│  │   │   Worker    │  │   Worker    │  │   Worker    │                │    │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │    │
│  │          │                │                │                        │    │
│  │          └────────────────┴────────────────┘                        │    │
│  │                           │                                          │    │
│  │                           v                                          │    │
│  │                  Collect all entries                                 │    │
│  │                           │                                          │    │
│  │                           v                                          │    │
│  │               VectorStore.add_entries() (batch)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Query Retrieval (HybridRetriever)                                          │
│  ─────────────────────────────────                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ _execute_parallel_searches()                                         │    │
│  │                                                                      │    │
│  │   queries → ThreadPoolExecutor                                       │    │
│  │                                                                      │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │    │
│  │   │  Query 1    │  │  Query 2    │  │  Query N    │                │    │
│  │   │   Search    │  │   Search    │  │   Search    │                │    │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │    │
│  │          │                │                │                        │    │
│  │          └────────────────┴────────────────┘                        │    │
│  │                           │                                          │    │
│  │                           v                                          │    │
│  │               Merge & deduplicate results                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_parallel_processing` | True | Enable parallel memory building |
| `max_parallel_workers` | 3 | Workers for memory building |
| `enable_parallel_retrieval` | True | Enable parallel query execution |
| `max_retrieval_workers` | 3 | Workers for retrieval |

### Fallback Behavior

Both parallel processing systems include automatic fallback to sequential mode:
```python
try:
    # Parallel execution
    with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
        ...
except Exception as e:
    print(f"Parallel execution failed: {e}. Falling back to sequential...")
    # Sequential fallback
```

---

## Configuration System

### Configuration File

**Template**: `config.py.example`

```python
# LLM Configuration
OPENAI_API_KEY = "your-api-key-here"
OPENAI_BASE_URL = None  # or custom endpoint
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Advanced LLM Features
ENABLE_THINKING = False
USE_STREAMING = False
USE_JSON_FORMAT = True

# Memory Building Parameters
WINDOW_SIZE = 10
OVERLAP_SIZE = 2

# Retrieval Parameters
SEMANTIC_TOP_K = 5
KEYWORD_TOP_K = 3
STRUCTURED_TOP_K = 10

# Database Configuration
LANCEDB_PATH = "./lancedb_data"
MEMORY_TABLE_NAME = "memory_entries"
```

### Parameter Precedence

```
Constructor Argument → config.py → Default Value
        (highest)                    (lowest)
```

Example:
```python
# If enable_parallel_processing=None passed to constructor,
# use config.ENABLE_PARALLEL_PROCESSING if defined,
# otherwise use True as default
self.enable_parallel_processing = (
    enable_parallel_processing
    if enable_parallel_processing is not None
    else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
)
```

---

## Evaluation Framework

### LoCoMo-10 Benchmark

**Location**: `test_locomo10.py`

**Dataset Structure**:
```
LoCoMoSample
├── sample_id: str
├── qa: List[QA]                    # Question-answer pairs
│   ├── question: str
│   ├── answer: str
│   ├── evidence: List[str]
│   ├── category: int (1-5)
│   └── adversarial_answer: str     # For category 5
├── conversation: Conversation
│   ├── speaker_a: str
│   ├── speaker_b: str
│   └── sessions: Dict[int, Session]
│       └── turns: List[Turn]
├── event_summary: EventSummary
├── observation: Observation
└── session_summary: Dict[str, str]
```

**Question Categories**:

| Category | Type | Description |
|----------|------|-------------|
| 1 | SingleHop | Direct factual questions |
| 2 | Temporal | Time-related questions |
| 3 | MultiHop | Multi-step reasoning |
| 4 | Complex | Complex reasoning |
| 5 | Adversarial | Questions with no answer in context |

### Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `exact_match` | Exact string match | 0-1 |
| `f1` | Token-based F1 score | 0-1 |
| `rouge1_f`, `rouge2_f`, `rougeL_f` | ROUGE scores | 0-1 |
| `bleu1-4` | BLEU scores | 0-1 |
| `bert_f1` | BERTScore F1 | 0-1 |
| `meteor` | METEOR score | 0-1 |
| `sbert_similarity` | Sentence embedding similarity | 0-1 |
| `llm_judge_score` | LLM-as-judge evaluation | 0-1 |

### Testing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LoCoMoTester.run_test()                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  for each sample:                                                            │
│      │                                                                       │
│      ├── Clear system (vector_store.clear())                                │
│      │                                                                       │
│      ├── Convert to Dialogues (convert_to_dialogues)                        │
│      │                                                                       │
│      ├── Add to memory (system.add_dialogues)                               │
│      │                                                                       │
│      ├── Finalize (system.finalize)                                         │
│      │                                                                       │
│      └── for each QA in sample:                                             │
│              │                                                               │
│              ├── Retrieve context (hybrid_retriever.retrieve)               │
│              │                                                               │
│              ├── Generate answer                                             │
│              │   ├── Category 5: Binary choice (special handling)           │
│              │   └── Others: Standard generation                            │
│              │                                                               │
│              ├── Calculate metrics (calculate_metrics)                      │
│              │                                                               │
│              └── Record timing and results                                  │
│                                                                              │
│  Aggregate results (aggregate_metrics)                                       │
│                                                                              │
│  Save to JSON file                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Category 5 Handling

Adversarial questions (category 5) use special binary-choice answer generation:
```python
def generate_category5_answer(self, question, contexts, adversarial_answer):
    # Present two options in random order:
    # - "Not mentioned in the conversation"
    # - adversarial_answer (the wrong answer)

    # Ground truth for category 5 is always:
    # "Not mentioned in the conversation"
```

---

## Design Decisions

### 1. Write-Time Disambiguation

**Decision**: Resolve coreferences and temporal expressions during memory creation, not retrieval.

**Rationale**:
- Eliminates downstream reasoning overhead
- Atomic entries are self-contained and unambiguous
- Retrieval can focus on matching, not interpretation

**Trade-off**: Higher write-time cost, but significantly faster and more accurate reads.

### 2. Multi-View Indexing

**Decision**: Index across semantic, lexical, and symbolic dimensions simultaneously.

**Rationale**:
- Semantic: Captures conceptual similarity
- Lexical: Handles exact term matching (names, dates)
- Symbolic: Enables hard filtering constraints

**Trade-off**: Storage overhead, but enables flexible retrieval strategies.

### 3. Planning-Based Retrieval

**Decision**: Analyze query complexity and generate targeted sub-queries before retrieval.

**Rationale**:
- Simple queries → minimal retrieval
- Complex queries → comprehensive coverage
- Reduces wasted tokens on irrelevant context

**Trade-off**: Additional LLM calls for planning, but overall token reduction.

### 4. LanceDB as Vector Store

**Decision**: Use LanceDB instead of alternatives (FAISS, Pinecone, etc.)

**Rationale**:
- Columnar storage with native vector support
- Rich metadata filtering capabilities
- No external service dependencies
- Easy local development

### 5. Parallel Processing with ThreadPoolExecutor

**Decision**: Use Python's ThreadPoolExecutor for parallelism.

**Rationale**:
- LLM API calls are I/O-bound, not CPU-bound
- GIL doesn't affect network-bound operations
- Simpler than multiprocessing for this use case
- Automatic fallback to sequential mode

### 6. JSON-Structured LLM Output

**Decision**: Require JSON format for all LLM outputs.

**Rationale**:
- Predictable parsing
- Structured metadata extraction
- Easier error detection and retry

**Implementation**: Robust JSON extraction handles multiple formats and cleans common LLM output issues.

---

## Appendix: API Reference

### SimpleMemSystem

```python
class SimpleMemSystem:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None
    ) -> None: ...

    def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None
    ) -> None: ...

    def add_dialogues(self, dialogues: List[Dialogue]) -> None: ...

    def finalize(self) -> None: ...

    def ask(self, question: str) -> str: ...

    def get_all_memories(self) -> List[MemoryEntry]: ...

    def print_memories(self) -> None: ...
```

### MemoryEntry

```python
class MemoryEntry(BaseModel):
    entry_id: str
    lossless_restatement: str
    keywords: List[str]
    timestamp: Optional[str]
    location: Optional[str]
    persons: List[str]
    entities: List[str]
    topic: Optional[str]
```

### Dialogue

```python
class Dialogue(BaseModel):
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str]
```
