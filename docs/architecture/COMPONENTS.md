# SimpleMem Components Reference

This document provides a comprehensive reference for all components in the SimpleMem system, including detailed API specifications, internal implementation details, and usage examples.

---

## Table of Contents

1. [Component Overview](#component-overview)
2. [SimpleMemSystem](#simplememsystem)
3. [MemoryBuilder](#memorybuilder)
4. [HybridRetriever](#hybridretriever)
5. [AnswerGenerator](#answergenerator)
6. [VectorStore](#vectorstore)
7. [LLMClient](#llmclient)
8. [EmbeddingModel](#embeddingmodel)
9. [Data Models](#data-models)
10. [Configuration Reference](#configuration-reference)

---

## Component Overview

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SimpleMemSystem                                  │
│                         (Orchestrator)                                   │
│                                                                          │
│  ┌─────────────────────┬────────────────────┬─────────────────────┐    │
│  │                     │                    │                     │    │
│  │    MemoryBuilder    │  HybridRetriever   │   AnswerGenerator   │    │
│  │    (Write Path)     │  (Retrieval)       │   (Synthesis)       │    │
│  │                     │                    │                     │    │
│  └─────────┬───────────┴──────────┬─────────┴──────────┬──────────┘    │
│            │                      │                    │               │
│            └──────────────────────┼────────────────────┘               │
│                                   │                                     │
│                          ┌────────┴────────┐                           │
│                          │   VectorStore   │                           │
│                          │   (Storage)     │                           │
│                          └────────┬────────┘                           │
│                                   │                                     │
│            ┌──────────────────────┼──────────────────────┐             │
│            │                      │                      │             │
│     ┌──────┴──────┐        ┌──────┴──────┐        ┌──────┴──────┐     │
│     │  LLMClient  │        │EmbeddingModel│        │   LanceDB   │     │
│     │  (LLM API)  │        │  (Vectors)   │        │  (Storage)  │     │
│     └─────────────┘        └─────────────┘        └─────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibility Matrix

| Component | File | Primary Responsibility | Dependencies |
|-----------|------|----------------------|--------------|
| `SimpleMemSystem` | `main.py` | System orchestration, API facade | All components |
| `MemoryBuilder` | `core/memory_builder.py` | Dialogue → Atomic entries | LLMClient, VectorStore |
| `HybridRetriever` | `core/hybrid_retriever.py` | Multi-path retrieval, planning | LLMClient, VectorStore |
| `AnswerGenerator` | `core/answer_generator.py` | Context → Answer synthesis | LLMClient |
| `VectorStore` | `database/vector_store.py` | Storage, indexing, search | EmbeddingModel, LanceDB |
| `LLMClient` | `utils/llm_client.py` | LLM API communication | OpenAI SDK |
| `EmbeddingModel` | `utils/embedding.py` | Vector embeddings | SentenceTransformers |
| `MemoryEntry` | `models/memory_entry.py` | Atomic entry data model | Pydantic |
| `Dialogue` | `models/memory_entry.py` | Input dialogue data model | Pydantic |

---

## SimpleMemSystem

**Location**: `main.py`

**Purpose**: Main orchestrator and public API facade for the SimpleMem system.

### Class Definition

```python
class SimpleMemSystem:
    """
    SimpleMem Main System

    Three-stage pipeline based on Semantic Lossless Compression:
    1. Semantic Structured Compression: add_dialogue() -> MemoryBuilder -> VectorStore
    2. Structured Indexing and Recursive Consolidation: (background evolution - future work)
    3. Adaptive Query-Aware Retrieval: ask() -> HybridRetriever -> AnswerGenerator
    """
```

### Constructor

```python
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
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[str]` | `config.OPENAI_API_KEY` | OpenAI API key |
| `model` | `Optional[str]` | `config.LLM_MODEL` | LLM model name (e.g., "gpt-4o-mini") |
| `base_url` | `Optional[str]` | `config.OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint |
| `db_path` | `Optional[str]` | `config.LANCEDB_PATH` | LanceDB database path |
| `table_name` | `Optional[str]` | `config.MEMORY_TABLE_NAME` | Memory table name |
| `clear_db` | `bool` | `False` | Clear existing database on init |
| `enable_thinking` | `Optional[bool]` | `config.ENABLE_THINKING` | Enable deep thinking mode (Qwen) |
| `use_streaming` | `Optional[bool]` | `config.USE_STREAMING` | Enable streaming responses |
| `enable_planning` | `Optional[bool]` | `config.ENABLE_PLANNING` | Enable multi-query planning |
| `enable_reflection` | `Optional[bool]` | `config.ENABLE_REFLECTION` | Enable reflection-based retrieval |
| `max_reflection_rounds` | `Optional[int]` | `config.MAX_REFLECTION_ROUNDS` | Maximum reflection iterations |
| `enable_parallel_processing` | `Optional[bool]` | `True` | Enable parallel memory building |
| `max_parallel_workers` | `Optional[int]` | `3` | Max workers for memory building |
| `enable_parallel_retrieval` | `Optional[bool]` | `True` | Enable parallel query execution |
| `max_retrieval_workers` | `Optional[int]` | `3` | Max workers for retrieval |

### Internal Components

Created during initialization:

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm_client` | `LLMClient` | LLM API client instance |
| `embedding_model` | `EmbeddingModel` | Embedding model instance |
| `vector_store` | `VectorStore` | Storage and indexing |
| `memory_builder` | `MemoryBuilder` | Dialogue processing |
| `hybrid_retriever` | `HybridRetriever` | Query retrieval |
| `answer_generator` | `AnswerGenerator` | Answer synthesis |

### Methods

#### `add_dialogue`

```python
def add_dialogue(
    self,
    speaker: str,
    content: str,
    timestamp: Optional[str] = None
) -> None
```

Add a single dialogue turn.

| Parameter | Type | Description |
|-----------|------|-------------|
| `speaker` | `str` | Speaker name |
| `content` | `str` | Dialogue text content |
| `timestamp` | `Optional[str]` | ISO 8601 timestamp |

**Example**:
```python
system.add_dialogue(
    speaker="Alice",
    content="Let's meet at Starbucks tomorrow",
    timestamp="2025-11-15T14:30:00"
)
```

#### `add_dialogues`

```python
def add_dialogues(self, dialogues: List[Dialogue]) -> None
```

Batch add dialogues. Automatically selects parallel or sequential processing based on batch size.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dialogues` | `List[Dialogue]` | List of Dialogue objects |

**Example**:
```python
dialogues = [
    Dialogue(dialogue_id=1, speaker="Alice", content="Hello", timestamp="2025-11-15T14:30:00"),
    Dialogue(dialogue_id=2, speaker="Bob", content="Hi there", timestamp="2025-11-15T14:31:00"),
]
system.add_dialogues(dialogues)
```

#### `finalize`

```python
def finalize(self) -> None
```

Process any remaining dialogues in the buffer. Call after adding all dialogues.

#### `ask`

```python
def ask(self, question: str) -> str
```

Query the memory system and get an answer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | `str` | User question |

**Returns**: `str` - Generated answer

**Example**:
```python
answer = system.ask("When will Alice and Bob meet?")
# Returns: "16 November 2025 at 2:00 PM"
```

#### `get_all_memories`

```python
def get_all_memories(self) -> List[MemoryEntry]
```

Retrieve all stored memory entries. Used for debugging.

#### `print_memories`

```python
def print_memories(self) -> None
```

Print all memory entries to console in formatted output.

### Usage Example

```python
from main import SimpleMemSystem, create_system

# Using convenience function
system = create_system(
    clear_db=True,
    enable_planning=True,
    enable_parallel_processing=True
)

# Or direct instantiation
system = SimpleMemSystem(
    api_key="your-api-key",
    model="gpt-4o-mini",
    clear_db=True
)

# Add dialogues
system.add_dialogue("Alice", "Meeting at 2pm tomorrow", "2025-11-15T10:00:00")
system.add_dialogue("Bob", "I'll prepare the documents", "2025-11-15T10:01:00")

# Finalize processing
system.finalize()

# Query
answer = system.ask("What time is the meeting?")
```

---

## MemoryBuilder

**Location**: `core/memory_builder.py`

**Purpose**: Implements Stage 1 - Semantic Structured Compression. Transforms raw dialogues into atomic memory entries.

### Class Definition

```python
class MemoryBuilder:
    """
    Memory Builder - Stage 1: Semantic Structured Compression

    Paper Reference: Section 3.1 - Semantic Structured Compression

    Core Functions:
    1. Entropy-based filtering (implicit via window processing)
    2. De-linearization transformation F_θ: Dialogue → Atomic Entries
    3. Coreference resolution Φ_coref (no pronouns)
    4. Temporal anchoring Φ_time (absolute timestamps)
    5. Generate self-contained Atomic Entries {m_k}
    """
```

### Constructor

```python
def __init__(
    self,
    llm_client: LLMClient,
    vector_store: VectorStore,
    window_size: int = None,
    enable_parallel_processing: bool = True,
    max_parallel_workers: int = 3
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `LLMClient` | Required | LLM client for extraction |
| `vector_store` | `VectorStore` | Required | Storage for entries |
| `window_size` | `int` | `config.WINDOW_SIZE` | Dialogues per processing window |
| `enable_parallel_processing` | `bool` | `True` | Enable parallel window processing |
| `max_parallel_workers` | `int` | `3` | Maximum parallel workers |

### Internal State

| Attribute | Type | Description |
|-----------|------|-------------|
| `dialogue_buffer` | `List[Dialogue]` | Accumulated dialogues awaiting processing |
| `processed_count` | `int` | Total dialogues processed |
| `previous_entries` | `List[MemoryEntry]` | Last window's entries (context for deduplication) |

### Methods

#### `add_dialogue`

```python
def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None
```

Add single dialogue to buffer. Triggers processing when buffer reaches `window_size`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dialogue` | `Dialogue` | Required | Dialogue to add |
| `auto_process` | `bool` | `True` | Auto-process when buffer is full |

#### `add_dialogues`

```python
def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True) -> None
```

Batch add dialogues. Selects parallel processing for large batches (`> window_size * 2`).

#### `add_dialogues_parallel`

```python
def add_dialogues_parallel(self, dialogues: List[Dialogue]) -> None
```

Add dialogues with parallel processing. Groups dialogues into windows and processes concurrently.

**Processing Flow**:
1. Extend dialogue buffer with all dialogues
2. Split into windows of `window_size`
3. Process all windows (including remainder) in parallel via `ThreadPoolExecutor`
4. Batch store all entries to VectorStore
5. Fallback to sequential on failure

#### `process_window`

```python
def process_window(self) -> None
```

Process current window from buffer. Core processing logic.

**Steps**:
1. Extract window from buffer
2. Call `_generate_memory_entries()` for LLM extraction
3. Store entries to VectorStore
4. Update `previous_entries` for context

#### `process_remaining`

```python
def process_remaining(self) -> None
```

Process any remaining dialogues in buffer. Called by `finalize()`.

### Private Methods

#### `_generate_memory_entries`

```python
def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]
```

Core De-linearization transformation F_θ.

**Paper Reference**: Section 3.1 - Eq. (3)

```
F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract
```

**Process**:
1. Build dialogue text from input
2. Include previous entries as context (avoids duplication)
3. Build extraction prompt
4. Call LLM with retry mechanism (3 attempts)
5. Parse JSON response to MemoryEntry list

#### `_build_extraction_prompt`

```python
def _build_extraction_prompt(
    self,
    dialogue_text: str,
    dialogue_ids: List[int],
    context: str
) -> str
```

Build the LLM prompt for memory extraction.

**Prompt Requirements Enforced**:
- Complete coverage of all information
- Force disambiguation (no pronouns)
- Lossless restatement
- ISO 8601 timestamps
- Structured metadata extraction

#### `_parse_llm_response`

```python
def _parse_llm_response(
    self,
    response: str,
    dialogue_ids: List[int]
) -> List[MemoryEntry]
```

Parse LLM JSON response into MemoryEntry objects.

#### `_process_windows_parallel`

```python
def _process_windows_parallel(self, windows: List[List[Dialogue]]) -> None
```

Process multiple windows in parallel using `ThreadPoolExecutor`.

**Features**:
- Submits all windows as concurrent tasks
- Collects results as they complete
- Batch stores all entries
- Updates `previous_entries` with last 10 entries

#### `_generate_memory_entries_worker`

```python
def _generate_memory_entries_worker(
    self,
    window: List[Dialogue],
    dialogue_ids: List[int],
    window_num: int
) -> List[MemoryEntry]
```

Worker function for parallel processing. Called by executor for each window.

### Extraction Prompt Template

The prompt enforces the paper's requirements for atomic entries:

```
Your task is to extract all valuable information from the following dialogues
and convert them into structured memory entries.

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information
   in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they,
   this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete,
   independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format
   - location: Specific location name
   - persons: All person names mentioned
   - entities: Companies, products, organizations
   - topic: The topic of this information
```

---

## HybridRetriever

**Location**: `core/hybrid_retriever.py`

**Purpose**: Implements Stage 3 - Adaptive Query-Aware Retrieval with Pruning.

### Class Definition

```python
class HybridRetriever:
    """
    Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning

    Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning

    Core Components:
    1. Query-aware retrieval across three structured layers:
       - Semantic Layer: Dense vector similarity
       - Lexical Layer: Sparse keyword matching (BM25)
       - Symbolic Layer: Metadata filtering
    2. Hybrid Scoring Function S(q, m_k): aggregates multi-layer signals
    3. Complexity-Aware Pruning: dynamic depth based on C_q
    4. Planning-based multi-query decomposition for comprehensive retrieval
    """
```

### Constructor

```python
def __init__(
    self,
    llm_client: LLMClient,
    vector_store: VectorStore,
    semantic_top_k: int = None,
    keyword_top_k: int = None,
    structured_top_k: int = None,
    enable_planning: bool = True,
    enable_reflection: bool = True,
    max_reflection_rounds: int = 2,
    enable_parallel_retrieval: bool = True,
    max_retrieval_workers: int = 3
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `LLMClient` | Required | LLM client for planning/reflection |
| `vector_store` | `VectorStore` | Required | Storage for searches |
| `semantic_top_k` | `int` | `config.SEMANTIC_TOP_K` | Results for semantic search |
| `keyword_top_k` | `int` | `config.KEYWORD_TOP_K` | Results for keyword search |
| `structured_top_k` | `int` | `config.STRUCTURED_TOP_K` | Results for structured search |
| `enable_planning` | `bool` | `True` | Enable multi-query planning |
| `enable_reflection` | `bool` | `True` | Enable reflection loops |
| `max_reflection_rounds` | `int` | `2` | Maximum reflection iterations |
| `enable_parallel_retrieval` | `bool` | `True` | Parallel query execution |
| `max_retrieval_workers` | `int` | `3` | Workers for parallel retrieval |

### Methods

#### `retrieve`

```python
def retrieve(
    self,
    query: str,
    enable_reflection: Optional[bool] = None
) -> List[MemoryEntry]
```

Main retrieval entry point.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `enable_reflection` | `Optional[bool]` | `None` | Override reflection setting |

**Returns**: `List[MemoryEntry]` - Relevant memory entries

**Example**:
```python
# With reflection (default)
results = retriever.retrieve("When will Alice and Bob meet?")

# Without reflection (for adversarial questions)
results = retriever.retrieve("What is Alice's favorite food?", enable_reflection=False)
```

### Planning-Based Retrieval

#### `_retrieve_with_planning`

```python
def _retrieve_with_planning(
    self,
    query: str,
    enable_reflection: Optional[bool] = None
) -> List[MemoryEntry]
```

Execute retrieval with intelligent planning process.

**Steps**:
1. **Analyze information requirements** → `_analyze_information_requirements()`
2. **Generate targeted queries** → `_generate_targeted_queries()`
3. **Execute parallel searches** → `_execute_parallel_searches()`
4. **Merge and deduplicate** → `_merge_and_deduplicate_entries()`
5. **Optional reflection** → `_retrieve_with_intelligent_reflection()`

#### `_analyze_information_requirements`

```python
def _analyze_information_requirements(self, query: str) -> Dict[str, Any]
```

Query Complexity Estimation C_q (Paper Reference: Section 3.3 - Eq. (8)).

**Returns** JSON structure:
```json
{
  "question_type": "factual|temporal|relational|explanatory",
  "key_entities": ["entity1", "entity2"],
  "required_info": [
    {
      "info_type": "type description",
      "description": "specific information needed",
      "priority": "high|medium|low"
    }
  ],
  "relationships": ["relationship1", "relationship2"],
  "minimal_queries_needed": 2
}
```

#### `_generate_targeted_queries`

```python
def _generate_targeted_queries(
    self,
    original_query: str,
    information_plan: Dict[str, Any]
) -> List[str]
```

Generate minimal targeted queries based on information requirements.

**Features**:
- Always includes original query
- Limits to max 4 queries for efficiency
- Each query targets specific information requirement
- Avoids redundant/overlapping queries

### Search Methods

#### `_semantic_search`

```python
def _semantic_search(self, query: str) -> List[MemoryEntry]
```

Semantic Layer Retrieval using dense vector similarity.

**Paper Reference**: Section 3.3 - `λ₁ · cos(e_q, v_k)`

#### `_keyword_search`

```python
def _keyword_search(
    self,
    query: str,
    query_analysis: Dict[str, Any]
) -> List[MemoryEntry]
```

Lexical Layer Retrieval using BM25-style keyword matching.

**Paper Reference**: Section 3.3 - `λ₂ · BM25(q_lex, S_k)`

#### `_structured_search`

```python
def _structured_search(self, query_analysis: Dict[str, Any]) -> List[MemoryEntry]
```

Symbolic Layer Retrieval using metadata constraints.

**Paper Reference**: Section 3.3 - `γ · 𝕀(R_k ⊨ C_meta)`

### Reflection Loop

#### `_retrieve_with_intelligent_reflection`

```python
def _retrieve_with_intelligent_reflection(
    self,
    query: str,
    initial_results: List[MemoryEntry],
    information_plan: Dict[str, Any]
) -> List[MemoryEntry]
```

Execute intelligent reflection-based additional retrieval.

**Process** (per round):
1. Analyze information completeness → `_analyze_information_completeness()`
2. If incomplete, generate missing info queries → `_generate_missing_info_queries()`
3. Execute additional searches (parallel or sequential)
4. Merge with current results

#### `_analyze_information_completeness`

```python
def _analyze_information_completeness(
    self,
    query: str,
    current_results: List[MemoryEntry],
    information_plan: Dict[str, Any]
) -> str
```

Analyze if current results provide complete information.

**Returns**: `"complete"` | `"incomplete"` | `"no_results"`

#### `_check_answer_adequacy`

```python
def _check_answer_adequacy(
    self,
    query: str,
    contexts: List[MemoryEntry]
) -> str
```

Check if current contexts are sufficient to answer the query.

**Returns**: `"sufficient"` | `"insufficient"` | `"no_results"`

### Parallel Execution

#### `_execute_parallel_searches`

```python
def _execute_parallel_searches(self, search_queries: List[str]) -> List[MemoryEntry]
```

Execute multiple search queries in parallel using `ThreadPoolExecutor`.

**Features**:
- Automatic fallback to sequential on failure
- Logs progress for each query

### Utility Methods

#### `_analyze_query`

```python
def _analyze_query(self, query: str) -> Dict[str, Any]
```

Use LLM to analyze query intent and extract structured information.

**Returns**:
```json
{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1", "name2"],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1"]
}
```

#### `_parse_time_range`

```python
def _parse_time_range(self, time_expression: str) -> Optional[tuple]
```

Parse time expression to (start_time, end_time) tuple using `dateparser`.

#### `_merge_and_deduplicate_entries`

```python
def _merge_and_deduplicate_entries(self, entries: List[MemoryEntry]) -> List[MemoryEntry]
```

Merge and deduplicate memory entries by `entry_id`.

---

## AnswerGenerator

**Location**: `core/answer_generator.py`

**Purpose**: Synthesizes final answers from retrieved atomic contexts.

### Class Definition

```python
class AnswerGenerator:
    """
    Answer Generator - Reconstructive Synthesis from Atomic Contexts

    Paper Reference: Section 3.3 - Eq. (10)
    Synthesizes final answer from pruned, query-specific context:
    C_final = ⊕_{m ∈ Top-k_dyn(S)} [t_m: Content(m)]

    Features:
    1. Receive query and retrieved atomic entries
    2. Generate answers from disambiguated, self-contained facts
    3. Ensure accuracy through atomic context independence
    """
```

### Constructor

```python
def __init__(self, llm_client: LLMClient) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_client` | `LLMClient` | LLM client for answer generation |

### Methods

#### `generate_answer`

```python
def generate_answer(
    self,
    query: str,
    contexts: List[MemoryEntry]
) -> str
```

Generate concise answer from retrieved contexts.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | User question |
| `contexts` | `List[MemoryEntry]` | Retrieved memory entries |

**Returns**: `str` - Concise answer phrase

**Features**:
- Returns "No relevant information found" if no contexts
- Retry mechanism (3 attempts)
- JSON-structured output with reasoning

**Example**:
```python
answer = generator.generate_answer(
    "When will they meet?",
    [memory_entry1, memory_entry2]
)
# Returns: "16 November 2025 at 2:00 PM"
```

#### `_format_contexts`

```python
def _format_contexts(self, contexts: List[MemoryEntry]) -> str
```

Format contexts to readable text for LLM prompt.

**Output Format**:
```
[Context 1]
Content: Alice suggested meeting Bob at Starbucks on 2025-11-16...
Time: 2025-11-16T14:00:00
Location: Starbucks
Persons: Alice, Bob

[Context 2]
Content: ...
```

#### `_build_answer_prompt`

```python
def _build_answer_prompt(self, query: str, context_str: str) -> str
```

Build answer generation prompt.

**Prompt Requirements**:
1. Think through reasoning process
2. Provide CONCISE answer (short phrase)
3. Answer ONLY from provided context
4. Format dates as 'DD Month YYYY'
5. Return JSON format

**Expected Output**:
```json
{
  "reasoning": "The context explicitly states the meeting time...",
  "answer": "16 November 2025 at 2:00 PM"
}
```

---

## VectorStore

**Location**: `database/vector_store.py`

**Purpose**: Implements Structured Multi-View Indexing (Section 3.2) with LanceDB backend.

### Class Definition

```python
class VectorStore:
    """
    Structured Multi-View Indexing - Storage and retrieval for Atomic Entries

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity
    2. Lexical Layer: Sparse keyword vectors for precise term matching
    3. Symbolic Layer: Structured metadata for deterministic filtering
    """
```

### Constructor

```python
def __init__(
    self,
    db_path: str = None,
    embedding_model: EmbeddingModel = None,
    table_name: str = None
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `config.LANCEDB_PATH` | LanceDB storage path |
| `embedding_model` | `EmbeddingModel` | New instance | Embedding model |
| `table_name` | `str` | `config.MEMORY_TABLE_NAME` | Table name |

### Database Schema

```python
schema = pa.schema([
    pa.field("entry_id", pa.string()),              # UUID
    pa.field("lossless_restatement", pa.string()),  # Semantic layer base
    pa.field("keywords", pa.list_(pa.string())),    # Lexical layer
    pa.field("timestamp", pa.string()),             # Symbolic layer
    pa.field("location", pa.string()),              # Symbolic layer
    pa.field("persons", pa.list_(pa.string())),     # Symbolic layer
    pa.field("entities", pa.list_(pa.string())),    # Symbolic layer
    pa.field("topic", pa.string()),                 # Symbolic layer
    pa.field("vector", pa.list_(pa.float32(), dimension))  # Dense embedding
])
```

### Methods

#### `add_entries`

```python
def add_entries(self, entries: List[MemoryEntry]) -> None
```

Batch add memory entries with embeddings.

**Process**:
1. Generate embeddings for all `lossless_restatement` fields
2. Build data dictionaries
3. Add to LanceDB table

#### `semantic_search`

```python
def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]
```

Semantic Layer Search using dense vector similarity.

**Paper Reference**: Section 3.1 - `v_k = E_dense(S_k)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `top_k` | `int` | `5` | Maximum results |

**Process**:
1. Check for empty table
2. Generate query vector (with query prompt for Qwen3)
3. Execute vector search via LanceDB
4. Convert results to MemoryEntry objects

#### `keyword_search`

```python
def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]
```

Lexical Layer Search using keyword matching.

**Paper Reference**: Section 3.1 - `h_k = Sparse(S_k)`

**Scoring**:
- +2 points for keyword list match
- +1 point for text content match

**Process**:
1. Load all entries to pandas DataFrame
2. Score each entry against keywords
3. Sort by score descending
4. Return top_k results

#### `structured_search`

```python
def structured_search(
    self,
    persons: Optional[List[str]] = None,
    timestamp_range: Optional[tuple] = None,
    location: Optional[str] = None,
    entities: Optional[List[str]] = None,
    top_k: Optional[int] = None
) -> List[MemoryEntry]
```

Symbolic Layer Search using metadata filtering.

**Paper Reference**: Section 3.1 - `R_k = {(key, val)}`

| Parameter | Type | Description |
|-----------|------|-------------|
| `persons` | `Optional[List[str]]` | Filter by person names |
| `timestamp_range` | `Optional[tuple]` | (start, end) ISO 8601 |
| `location` | `Optional[str]` | Filter by location substring |
| `entities` | `Optional[List[str]]` | Filter by entities |
| `top_k` | `Optional[int]` | Maximum results |

**Filter Logic**: All provided filters are AND-combined.

#### `get_all_entries`

```python
def get_all_entries(self) -> List[MemoryEntry]
```

Retrieve all stored memory entries.

#### `clear`

```python
def clear(self) -> None
```

Drop and recreate the table.

### Private Methods

#### `_init_table`

```python
def _init_table(self) -> None
```

Initialize table schema. Creates new table or opens existing.

---

## LLMClient

**Location**: `utils/llm_client.py`

**Purpose**: Unified LLM API client with streaming, thinking mode, and robust JSON parsing.

### Class Definition

```python
class LLMClient:
    """
    Unified LLM client interface
    """
```

### Constructor

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    use_streaming: Optional[bool] = None
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[str]` | `config.OPENAI_API_KEY` | API key |
| `model` | `Optional[str]` | `config.LLM_MODEL` | Model name |
| `base_url` | `Optional[str]` | `config.OPENAI_BASE_URL` | Custom endpoint |
| `enable_thinking` | `Optional[bool]` | `config.ENABLE_THINKING` | Deep thinking mode |
| `use_streaming` | `Optional[bool]` | `config.USE_STREAMING` | Streaming responses |

### Supported Backends

- **OpenAI API** (default)
- **Qwen/DashScope** (via `dashscope.aliyuncs.com`)
- **Azure OpenAI**
- **Any OpenAI-compatible endpoint**

### Methods

#### `chat_completion`

```python
def chat_completion(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    response_format: Optional[Dict[str, str]] = None,
    max_retries: int = 3
) -> str
```

Standard chat completion with retry mechanism.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict[str, str]]` | Required | Chat messages |
| `temperature` | `float` | `0.2` | Generation temperature |
| `response_format` | `Optional[Dict]` | `None` | JSON mode: `{"type": "json_object"}` |
| `max_retries` | `int` | `3` | Retry attempts |

**Retry Mechanism**:
- Exponential backoff: 1s, 2s, 4s
- Raises last exception after all retries fail

**Qwen API Handling**:
```python
# For Qwen API (dashscope.aliyuncs.com):
# - Streaming + thinking: enable_thinking=True
# - Non-streaming: enable_thinking=False (required)
# - JSON format: enable_thinking=False (incompatible)
```

#### `_handle_streaming_response`

```python
def _handle_streaming_response(self, **kwargs) -> str
```

Handle streaming response and collect full content.

#### `extract_json`

```python
def extract_json(self, text: str) -> Any
```

Extract JSON from LLM response with robust parsing.

**Supported Formats**:
1. Pure JSON
2. ` ```json ... ``` ` block
3. ` ``` ... ``` ` generic code block
4. JSON embedded in text with common prefixes
5. Multiple JSON objects (returns first valid)

**Process**:
1. Remove common LLM prefixes ("Here's the JSON:", etc.)
2. Try direct `json.loads()`
3. Try extracting from ` ```json ``` ` block
4. Try extracting from generic code block
5. Try finding balanced JSON object/array
6. Try cleaning and parsing chunks

### Private Methods

#### `_clean_json_string`

```python
def _clean_json_string(self, json_str: str) -> str
```

Clean common issues in JSON strings:
- Remove trailing commas before `}` or `]`
- Remove comments (`//` and `/* */`)

#### `_extract_balanced_json`

```python
def _extract_balanced_json(self, text: str, start_char: str) -> Any
```

Extract balanced JSON object or array by tracking depth.

**Features**:
- Handles nested structures
- Ignores brackets inside strings
- Handles escape sequences

---

## EmbeddingModel

**Location**: `utils/embedding.py`

**Purpose**: Generate vector embeddings using SentenceTransformers with Qwen3 support.

### Class Definition

```python
class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
```

### Constructor

```python
def __init__(
    self,
    model_name: str = None,
    use_optimization: bool = True
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `config.EMBEDDING_MODEL` | Model name/path |
| `use_optimization` | `bool` | `True` | Enable Flash Attention 2 |

### Supported Models

| Model Type | Example | Dimension |
|------------|---------|-----------|
| Qwen3 | `qwen3-0.6b`, `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| Standard | `sentence-transformers/all-MiniLM-L6-v2` | 384 |

### Internal State

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `SentenceTransformer` | Model instance |
| `dimension` | `int` | Embedding dimension |
| `model_type` | `str` | `"qwen3_sentence_transformer"` or `"sentence_transformer"` |
| `supports_query_prompt` | `bool` | Qwen3 query prompt support |

### Methods

#### `encode`

```python
def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray
```

Encode list of texts to vectors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `List[str]` | Required | Texts to encode |
| `is_query` | `bool` | `False` | Use query prompt (Qwen3) |

**Returns**: `np.ndarray` - Embeddings array

#### `encode_single`

```python
def encode_single(self, text: str, is_query: bool = False) -> np.ndarray
```

Encode single text.

#### `encode_query`

```python
def encode_query(self, queries: List[str]) -> np.ndarray
```

Encode queries with optimal settings for Qwen3 (uses query prompt).

#### `encode_documents`

```python
def encode_documents(self, documents: List[str]) -> np.ndarray
```

Encode documents (no query prompt).

### Qwen3 Query vs Document Encoding

For asymmetric retrieval, Qwen3 models support different encoding for queries vs documents:

```python
# Query encoding (for search queries)
embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True)

# Document encoding (for indexed content)
embeddings = model.encode(documents, normalize_embeddings=True)
```

### Initialization Methods

#### `_init_qwen3_sentence_transformer`

Initialize Qwen3 model with optional Flash Attention 2 optimization.

**Model Mapping**:
```python
{
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
}
```

#### `_init_standard_sentence_transformer`

Initialize standard SentenceTransformer model.

#### `_fallback_to_sentence_transformer`

Fallback to `sentence-transformers/all-MiniLM-L6-v2` on failure.

---

## Data Models

**Location**: `models/memory_entry.py`

### MemoryEntry

Atomic Entry - Self-contained memory unit indexed across three orthogonal layers.

```python
class MemoryEntry(BaseModel):
    """
    Atomic Entry - Self-contained memory unit indexed across three orthogonal layers

    Paper Reference: Section 3.1 - Eq. (3), (4)
    Generated by De-linearization: m_k = F_θ(W_t) = Φ_time ∘ Φ_coref ∘ Φ_extract(W_t)
    Indexed via: M(m_k) = {v_k (semantic), h_k (lexical), R_k (symbolic)}
    """

    # Unique identifier
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # [Semantic Layer] - Dense embedding base (v_k = E_dense(S_k))
    lossless_restatement: str = Field(
        ...,
        description="Self-contained fact with Φ_coref (no pronouns) and Φ_time (absolute timestamps)"
    )

    # [Lexical Layer] - Sparse keyword vectors (h_k = Sparse(S_k))
    keywords: List[str] = Field(
        default_factory=list,
        description="Core keywords for BM25-style exact matching"
    )

    # [Symbolic Layer] - Metadata constraints (R_k = {(key, val)})
    timestamp: Optional[str] = Field(
        None,
        description="Standardized time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
    )
    location: Optional[str] = Field(
        None,
        description="Natural language location description"
    )
    persons: List[str] = Field(
        default_factory=list,
        description="List of extracted persons"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of extracted entities (companies, products, etc.)"
    )
    topic: Optional[str] = Field(
        None,
        description="Topic phrase summarized by LLM"
    )
```

#### Field Descriptions

| Field | Type | Layer | Description |
|-------|------|-------|-------------|
| `entry_id` | `str` | - | UUID, auto-generated |
| `lossless_restatement` | `str` | Semantic | Self-contained fact, base for embedding |
| `keywords` | `List[str]` | Lexical | BM25-style keywords |
| `timestamp` | `Optional[str]` | Symbolic | ISO 8601 format |
| `location` | `Optional[str]` | Symbolic | Location description |
| `persons` | `List[str]` | Symbolic | Person names |
| `entities` | `List[str]` | Symbolic | Companies, products, etc. |
| `topic` | `Optional[str]` | Symbolic | Topic phrase |

#### Example

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

Original dialogue entry.

```python
class Dialogue(BaseModel):
    """
    Original dialogue entry
    """
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None  # ISO 8601 format

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{time_str}{self.speaker}: {self.content}"
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `dialogue_id` | `int` | Sequential identifier |
| `speaker` | `str` | Speaker name |
| `content` | `str` | Dialogue text |
| `timestamp` | `Optional[str]` | ISO 8601 format |

#### String Representation

```python
dialogue = Dialogue(
    dialogue_id=1,
    speaker="Alice",
    content="Let's meet tomorrow",
    timestamp="2025-11-15T14:30:00"
)
str(dialogue)  # "[2025-11-15T14:30:00] Alice: Let's meet tomorrow"
```

---

## Configuration Reference

**Location**: `config.py` (copied from `config.py.example`)

### LLM Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `OPENAI_API_KEY` | `str` | Required | OpenAI API key |
| `OPENAI_BASE_URL` | `str` | `None` | Custom OpenAI-compatible endpoint |
| `LLM_MODEL` | `str` | `"gpt-4o-mini"` | LLM model name |
| `EMBEDDING_MODEL` | `str` | `"Qwen/Qwen3-Embedding-0.6B"` | Embedding model |

### Advanced LLM Features

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_THINKING` | `bool` | `False` | Enable deep thinking (Qwen) |
| `USE_STREAMING` | `bool` | `False` | Enable streaming responses |
| `USE_JSON_FORMAT` | `bool` | `True` | Request JSON format from LLM |

### Memory Building Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `WINDOW_SIZE` | `int` | `10` | Dialogues per processing window |
| `OVERLAP_SIZE` | `int` | `2` | Overlap between windows |

### Retrieval Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `SEMANTIC_TOP_K` | `int` | `5` | Semantic search results |
| `KEYWORD_TOP_K` | `int` | `3` | Keyword search results |
| `STRUCTURED_TOP_K` | `int` | `10` | Structured search results |
| `ENABLE_PLANNING` | `bool` | `True` | Enable multi-query planning |
| `ENABLE_REFLECTION` | `bool` | `True` | Enable reflection loops |
| `MAX_REFLECTION_ROUNDS` | `int` | `2` | Maximum reflection iterations |

### Parallel Processing

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_PARALLEL_PROCESSING` | `bool` | `True` | Parallel memory building |
| `MAX_PARALLEL_WORKERS` | `int` | `4` | Workers for memory building |
| `ENABLE_PARALLEL_RETRIEVAL` | `bool` | `True` | Parallel query execution |
| `MAX_RETRIEVAL_WORKERS` | `int` | `3` | Workers for retrieval |

### Database Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `LANCEDB_PATH` | `str` | `"./lancedb_data"` | LanceDB storage path |
| `MEMORY_TABLE_NAME` | `str` | `"memory_entries"` | Table name |

### Example Configuration

```python
# config.py

# LLM Configuration
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Advanced Features
ENABLE_THINKING = False
USE_STREAMING = False
USE_JSON_FORMAT = True

# Memory Building
WINDOW_SIZE = 10
OVERLAP_SIZE = 2

# Retrieval
SEMANTIC_TOP_K = 5
KEYWORD_TOP_K = 3
STRUCTURED_TOP_K = 10
ENABLE_PLANNING = True
ENABLE_REFLECTION = True
MAX_REFLECTION_ROUNDS = 2

# Parallel Processing
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = 4
ENABLE_PARALLEL_RETRIEVAL = True
MAX_RETRIEVAL_WORKERS = 3

# Database
LANCEDB_PATH = "./lancedb_data"
MEMORY_TABLE_NAME = "memory_entries"
```
