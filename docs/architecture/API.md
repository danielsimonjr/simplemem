# SimpleMem API Reference

This document provides a comprehensive API reference for all public interfaces in the SimpleMem system.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [SimpleMemSystem API](#simplememsystem-api)
4. [MemoryBuilder API](#memorybuilder-api)
5. [HybridRetriever API](#hybridretriever-api)
6. [AnswerGenerator API](#answergenerator-api)
7. [VectorStore API](#vectorstore-api)
8. [LLMClient API](#llmclient-api)
9. [EmbeddingModel API](#embeddingmodel-api)
10. [Data Models API](#data-models-api)
11. [Configuration Reference](#configuration-reference)
12. [Error Handling](#error-handling)
13. [Usage Examples](#usage-examples)

---

## Overview

SimpleMem provides a layered API architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     HIGH-LEVEL API (User-Facing)                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │ SimpleMemSystem                                                  │   │ │
│  │  │  • add_dialogue()      - Add single dialogue                     │   │ │
│  │  │  • add_dialogues()     - Batch add dialogues                     │   │ │
│  │  │  • finalize()          - Process remaining buffer                │   │ │
│  │  │  • ask()               - Query the memory system                 │   │ │
│  │  │  • get_all_memories()  - Retrieve all stored memories            │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    MID-LEVEL API (Component Access)                     │ │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │ │
│  │  │ MemoryBuilder   │ │ HybridRetriever │ │ AnswerGenerator │           │ │
│  │  │ • add_dialogue()│ │ • retrieve()    │ │ • generate_     │           │ │
│  │  │ • add_dialogues │ │                 │ │   answer()      │           │ │
│  │  │ • process_      │ │                 │ │                 │           │ │
│  │  │   window()      │ │                 │ │                 │           │ │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      LOW-LEVEL API (Infrastructure)                     │ │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │ │
│  │  │ VectorStore     │ │ LLMClient       │ │ EmbeddingModel  │           │ │
│  │  │ • add_entries() │ │ • chat_         │ │ • encode()      │           │ │
│  │  │ • semantic_     │ │   completion()  │ │ • encode_       │           │ │
│  │  │   search()      │ │ • extract_json()│ │   single()      │           │ │
│  │  │ • keyword_      │ │                 │ │ • encode_query()│           │ │
│  │  │   search()      │ │                 │ │ • encode_       │           │ │
│  │  │ • structured_   │ │                 │ │   documents()   │           │ │
│  │  │   search()      │ │                 │ │                 │           │ │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from main import SimpleMemSystem, create_system
from models.memory_entry import Dialogue

# Create system with defaults from config.py
system = create_system(clear_db=True)

# Add dialogues
system.add_dialogue("Alice", "Let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00")
system.add_dialogue("Bob", "Sounds good, I'll bring the documents", "2025-11-15T14:31:00")

# Finalize processing
system.finalize()

# Query the system
answer = system.ask("When will Alice and Bob meet?")
print(answer)  # "16 November 2025 at 2:00 PM"
```

### Advanced Usage with Custom Configuration

```python
system = SimpleMemSystem(
    api_key="your-api-key",
    model="gpt-4o",
    base_url=None,  # Use OpenAI default
    db_path="./custom_db",
    table_name="my_memories",
    clear_db=True,
    enable_thinking=False,
    use_streaming=True,
    enable_planning=True,
    enable_reflection=True,
    max_reflection_rounds=3,
    enable_parallel_processing=True,
    max_parallel_workers=4,
    enable_parallel_retrieval=True,
    max_retrieval_workers=3
)
```

---

## SimpleMemSystem API

**Location:** `main.py`

The primary user-facing class that orchestrates the entire SimpleMem pipeline.

### Class: `SimpleMemSystem`

```python
class SimpleMemSystem:
    """
    SimpleMem Main System - Three-stage pipeline based on Semantic Lossless Compression
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
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[str]` | `config.OPENAI_API_KEY` | OpenAI API key |
| `model` | `Optional[str]` | `config.LLM_MODEL` | LLM model name |
| `base_url` | `Optional[str]` | `config.OPENAI_BASE_URL` | Custom OpenAI base URL |
| `db_path` | `Optional[str]` | `config.LANCEDB_PATH` | Database storage path |
| `table_name` | `Optional[str]` | `config.MEMORY_TABLE_NAME` | Memory table name |
| `clear_db` | `bool` | `False` | Clear existing database on init |
| `enable_thinking` | `Optional[bool]` | `config.ENABLE_THINKING` | Enable deep thinking mode |
| `use_streaming` | `Optional[bool]` | `config.USE_STREAMING` | Enable streaming responses |
| `enable_planning` | `Optional[bool]` | `config.ENABLE_PLANNING` | Enable multi-query planning |
| `enable_reflection` | `Optional[bool]` | `config.ENABLE_REFLECTION` | Enable reflection loops |
| `max_reflection_rounds` | `Optional[int]` | `config.MAX_REFLECTION_ROUNDS` | Max reflection iterations |
| `enable_parallel_processing` | `Optional[bool]` | `config.ENABLE_PARALLEL_PROCESSING` | Enable parallel memory building |
| `max_parallel_workers` | `Optional[int]` | `config.MAX_PARALLEL_WORKERS` | Max parallel workers for building |
| `enable_parallel_retrieval` | `Optional[bool]` | `config.ENABLE_PARALLEL_RETRIEVAL` | Enable parallel retrieval |
| `max_retrieval_workers` | `Optional[int]` | `config.MAX_RETRIEVAL_WORKERS` | Max parallel workers for retrieval |

### Methods

#### `add_dialogue()`

Add a single dialogue turn to the memory system.

```python
def add_dialogue(
    self,
    speaker: str,
    content: str,
    timestamp: Optional[str] = None
) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `speaker` | `str` | Speaker name |
| `content` | `str` | Dialogue content |
| `timestamp` | `Optional[str]` | ISO 8601 timestamp (e.g., "2025-11-15T14:30:00") |

**Example:**
```python
system.add_dialogue("Alice", "Let's meet tomorrow", "2025-11-15T14:30:00")
```

---

#### `add_dialogues()`

Batch add multiple dialogues with optional parallel processing.

```python
def add_dialogues(self, dialogues: List[Dialogue]) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dialogues` | `List[Dialogue]` | List of Dialogue objects |

**Example:**
```python
from models.memory_entry import Dialogue

dialogues = [
    Dialogue(dialogue_id=1, speaker="Alice", content="Hello", timestamp="2025-11-15T14:30:00"),
    Dialogue(dialogue_id=2, speaker="Bob", content="Hi there", timestamp="2025-11-15T14:31:00"),
]
system.add_dialogues(dialogues)
```

---

#### `finalize()`

Process any remaining dialogues in the buffer.

```python
def finalize(self) -> None
```

**Note:** In parallel processing mode, remaining dialogues are already processed, so this is primarily a safety check.

---

#### `ask()`

Query the memory system with a natural language question.

```python
def ask(self, question: str) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | `str` | Natural language question |

**Returns:** `str` - The generated answer

**Example:**
```python
answer = system.ask("When will Alice and Bob meet?")
```

---

#### `get_all_memories()`

Retrieve all stored memory entries (useful for debugging).

```python
def get_all_memories(self) -> List[MemoryEntry]
```

**Returns:** `List[MemoryEntry]` - All stored memory entries

---

#### `print_memories()`

Print all memory entries to console (for debugging).

```python
def print_memories(self) -> None
```

---

### Factory Function

```python
def create_system(
    clear_db: bool = False,
    enable_planning: Optional[bool] = None,
    enable_reflection: Optional[bool] = None,
    max_reflection_rounds: Optional[int] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None
) -> SimpleMemSystem
```

Convenience function to create a SimpleMem system using config.py defaults.

---

## MemoryBuilder API

**Location:** `core/memory_builder.py`

Implements Stage 1: Semantic Structured Compression (de-linearization transformation).

### Class: `MemoryBuilder`

```python
class MemoryBuilder:
    """
    Memory Builder - Stage 1: Semantic Structured Compression
    Implements de-linearization transformation F_θ: Dialogue → Atomic Entries
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
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `LLMClient` | Required | LLM client for extraction |
| `vector_store` | `VectorStore` | Required | Vector store for persistence |
| `window_size` | `int` | `config.WINDOW_SIZE` | Dialogues per processing window |
| `enable_parallel_processing` | `bool` | `True` | Enable parallel batch processing |
| `max_parallel_workers` | `int` | `3` | Maximum parallel workers |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `dialogue_buffer` | `List[Dialogue]` | Buffer of unprocessed dialogues |
| `processed_count` | `int` | Number of processed dialogues |
| `previous_entries` | `List[MemoryEntry]` | Previous window entries (for context) |

### Methods

#### `add_dialogue()`

Add a single dialogue to the buffer with optional auto-processing.

```python
def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dialogue` | `Dialogue` | Required | Dialogue to add |
| `auto_process` | `bool` | `True` | Auto-process when window is full |

---

#### `add_dialogues()`

Batch add dialogues with optional parallel processing.

```python
def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dialogues` | `List[Dialogue]` | Required | List of dialogues |
| `auto_process` | `bool` | `True` | Auto-process complete windows |

**Behavior:**
- If `enable_parallel_processing=True` and batch is large (> 2x window_size): uses parallel processing
- Otherwise: uses sequential processing

---

#### `add_dialogues_parallel()`

Add dialogues using parallel processing for better performance.

```python
def add_dialogues_parallel(self, dialogues: List[Dialogue]) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dialogues` | `List[Dialogue]` | List of dialogues to process in parallel |

**Note:** This method processes all dialogues including remainders, unlike sequential mode.

---

#### `process_window()`

Process the current window of dialogues.

```python
def process_window(self) -> None
```

**Behavior:**
1. Extracts window from buffer
2. Calls LLM to generate memory entries
3. Stores entries to vector store
4. Updates processed count

---

#### `process_remaining()`

Process any remaining dialogues in the buffer (fallback method).

```python
def process_remaining(self) -> None
```

**Note:** In parallel mode, this is typically not needed as all dialogues are processed immediately.

---

## HybridRetriever API

**Location:** `core/hybrid_retriever.py`

Implements Stage 3: Adaptive Query-Aware Retrieval with Pruning.

### Class: `HybridRetriever`

```python
class HybridRetriever:
    """
    Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning
    Implements hybrid scoring function S(q, m_k) across three structured layers
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
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `LLMClient` | Required | LLM client for planning/reflection |
| `vector_store` | `VectorStore` | Required | Vector store for search |
| `semantic_top_k` | `int` | `config.SEMANTIC_TOP_K` | Max semantic search results |
| `keyword_top_k` | `int` | `config.KEYWORD_TOP_K` | Max keyword search results |
| `structured_top_k` | `int` | `config.STRUCTURED_TOP_K` | Max structured search results |
| `enable_planning` | `bool` | `True` | Enable multi-query planning |
| `enable_reflection` | `bool` | `True` | Enable reflection loops |
| `max_reflection_rounds` | `int` | `2` | Maximum reflection iterations |
| `enable_parallel_retrieval` | `bool` | `True` | Enable parallel query execution |
| `max_retrieval_workers` | `int` | `3` | Maximum parallel workers |

### Methods

#### `retrieve()`

Execute retrieval with planning and optional reflection.

```python
def retrieve(
    self,
    query: str,
    enable_reflection: Optional[bool] = None
) -> List[MemoryEntry]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `enable_reflection` | `Optional[bool]` | `None` | Override global reflection setting |

**Returns:** `List[MemoryEntry]` - Relevant memory entries

**Retrieval Pipeline:**
1. If planning enabled: analyze requirements → generate targeted queries
2. Execute searches (parallel or sequential)
3. Merge and deduplicate results
4. If reflection enabled: check completeness → generate additional queries if needed

**Example:**
```python
# Standard retrieval with planning and reflection
results = retriever.retrieve("When will Alice and Bob meet?")

# Retrieval without reflection (for adversarial questions)
results = retriever.retrieve("What is Alice's favorite food?", enable_reflection=False)
```

---

#### Internal Search Methods

These methods are used internally but can be accessed directly if needed:

```python
# Semantic search (dense vectors)
def _semantic_search(self, query: str) -> List[MemoryEntry]

# Keyword search (sparse matching)
def _keyword_search(self, query: str, query_analysis: Dict[str, Any]) -> List[MemoryEntry]

# Structured search (metadata filtering)
def _structured_search(self, query_analysis: Dict[str, Any]) -> List[MemoryEntry]
```

---

## AnswerGenerator API

**Location:** `core/answer_generator.py`

Synthesizes final answers from retrieved atomic contexts.

### Class: `AnswerGenerator`

```python
class AnswerGenerator:
    """
    Answer Generator - Reconstructive Synthesis from Atomic Contexts
    Synthesizes final answer from pruned, query-specific context
    """
```

### Constructor

```python
def __init__(self, llm_client: LLMClient)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_client` | `LLMClient` | LLM client for answer generation |

### Methods

#### `generate_answer()`

Generate an answer from query and retrieved contexts.

```python
def generate_answer(
    self,
    query: str,
    contexts: List[MemoryEntry]
) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | User question |
| `contexts` | `List[MemoryEntry]` | Retrieved memory entries |

**Returns:** `str` - Generated answer (concise phrase)

**Behavior:**
- Returns `"No relevant information found"` if contexts is empty
- Formats contexts with metadata
- Prompts LLM for JSON-structured answer with reasoning
- Extracts answer field from JSON response

**Example:**
```python
answer = generator.generate_answer(
    "When will they meet?",
    [MemoryEntry(lossless_restatement="Alice and Bob will meet at 2pm on November 16, 2025", ...)]
)
# Returns: "16 November 2025 at 2:00 PM"
```

---

## VectorStore API

**Location:** `database/vector_store.py`

Implements Structured Multi-View Indexing with LanceDB backend.

### Class: `VectorStore`

```python
class VectorStore:
    """
    Structured Multi-View Indexing - Storage and retrieval for Atomic Entries
    Implements M(m_k) with three structured layers: semantic, lexical, symbolic
    """
```

### Constructor

```python
def __init__(
    self,
    db_path: str = None,
    embedding_model: EmbeddingModel = None,
    table_name: str = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `config.LANCEDB_PATH` | Database storage path |
| `embedding_model` | `EmbeddingModel` | Auto-created | Embedding model instance |
| `table_name` | `str` | `config.MEMORY_TABLE_NAME` | Table name in LanceDB |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `db` | `lancedb.Connection` | LanceDB connection |
| `table` | `lancedb.Table` | Memory entries table |
| `embedding_model` | `EmbeddingModel` | Embedding model |

### Methods

#### `add_entries()`

Batch add memory entries to the database.

```python
def add_entries(self, entries: List[MemoryEntry]) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `entries` | `List[MemoryEntry]` | Memory entries to store |

**Behavior:**
1. Generates embeddings for all entries
2. Serializes to LanceDB format
3. Adds to table

---

#### `semantic_search()`

Search using dense vector similarity.

```python
def semantic_search(
    self,
    query: str,
    top_k: int = 5
) -> List[MemoryEntry]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `top_k` | `int` | `5` | Maximum results |

**Returns:** `List[MemoryEntry]` - Matching entries sorted by similarity

**Example:**
```python
results = store.semantic_search("meeting schedule", top_k=10)
```

---

#### `keyword_search()`

Search using sparse keyword matching (BM25-style).

```python
def keyword_search(
    self,
    keywords: List[str],
    top_k: int = 3
) -> List[MemoryEntry]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keywords` | `List[str]` | Required | Keywords to match |
| `top_k` | `int` | `3` | Maximum results |

**Returns:** `List[MemoryEntry]` - Matching entries sorted by score

**Scoring:**
- +2 points for keyword list match
- +1 point for text content match

---

#### `structured_search()`

Search using metadata-based filtering.

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `persons` | `Optional[List[str]]` | `None` | Filter by person names |
| `timestamp_range` | `Optional[tuple]` | `None` | Filter by time range `(start, end)` |
| `location` | `Optional[str]` | `None` | Filter by location |
| `entities` | `Optional[List[str]]` | `None` | Filter by entities |
| `top_k` | `Optional[int]` | `None` | Maximum results (no limit if None) |

**Returns:** `List[MemoryEntry]` - Entries matching all specified filters

**Example:**
```python
results = store.structured_search(
    persons=["Alice", "Bob"],
    timestamp_range=("2025-11-15T00:00:00", "2025-11-16T23:59:59"),
    location="Starbucks"
)
```

---

#### `get_all_entries()`

Retrieve all stored memory entries.

```python
def get_all_entries(self) -> List[MemoryEntry]
```

**Returns:** `List[MemoryEntry]` - All entries in the database

---

#### `clear()`

Delete all data and reinitialize the table.

```python
def clear(self) -> None
```

---

## LLMClient API

**Location:** `utils/llm_client.py`

Unified LLM client interface with streaming and retry support.

### Class: `LLMClient`

```python
class LLMClient:
    """
    Unified LLM client interface with streaming, thinking mode, and retry support
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
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[str]` | `config.OPENAI_API_KEY` | API key |
| `model` | `Optional[str]` | `config.LLM_MODEL` | Model name |
| `base_url` | `Optional[str]` | `config.OPENAI_BASE_URL` | Custom API endpoint |
| `enable_thinking` | `Optional[bool]` | `config.ENABLE_THINKING` | Enable thinking mode |
| `use_streaming` | `Optional[bool]` | `config.USE_STREAMING` | Enable streaming |

### Methods

#### `chat_completion()`

Execute a chat completion request with retry mechanism.

```python
def chat_completion(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    response_format: Optional[Dict[str, str]] = None,
    max_retries: int = 3
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict[str, str]]` | Required | Chat messages |
| `temperature` | `float` | `0.2` | Sampling temperature |
| `response_format` | `Optional[Dict[str, str]]` | `None` | Response format (e.g., `{"type": "json_object"}`) |
| `max_retries` | `int` | `3` | Maximum retry attempts |

**Returns:** `str` - LLM response content

**Retry Behavior:**
- Exponential backoff: 1s, 2s, 4s between retries
- Raises last exception after all retries fail

**Example:**
```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    response_format={"type": "json_object"}
)
```

---

#### `extract_json()`

Extract JSON from LLM response with robust parsing.

```python
def extract_json(self, text: str) -> Any
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | LLM response text |

**Returns:** `Any` - Parsed JSON (dict, list, or primitive)

**Supported Formats:**
1. Pure JSON
2. ` ```json ... ``` ` code blocks
3. Generic ` ``` ... ``` ` code blocks
4. JSON embedded in text with common prefixes
5. Multiple JSON objects (returns first valid one)

**Raises:** `ValueError` - If no valid JSON can be extracted

---

## EmbeddingModel API

**Location:** `utils/embedding.py`

Embedding model interface supporting SentenceTransformers and Qwen3.

### Class: `EmbeddingModel`

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
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `config.EMBEDDING_MODEL` | Model name or path |
| `use_optimization` | `bool` | `True` | Use flash attention for Qwen3 |

**Supported Models:**
- `"qwen3-0.6b"` → `Qwen/Qwen3-Embedding-0.6B`
- `"qwen3-4b"` → `Qwen/Qwen3-Embedding-4B`
- `"qwen3-8b"` → `Qwen/Qwen3-Embedding-8B`
- Any SentenceTransformers model (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`)

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Active model name |
| `dimension` | `int` | Embedding dimension |
| `model_type` | `str` | `"qwen3_sentence_transformer"` or `"sentence_transformer"` |
| `supports_query_prompt` | `bool` | Whether model supports query prompts |

### Methods

#### `encode()`

Encode list of texts to vectors.

```python
def encode(
    self,
    texts: List[str],
    is_query: bool = False
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `List[str]` | Required | Texts to encode |
| `is_query` | `bool` | `False` | Use query prompt optimization |

**Returns:** `np.ndarray` - Shape `(len(texts), dimension)`

---

#### `encode_single()`

Encode a single text to a vector.

```python
def encode_single(
    self,
    text: str,
    is_query: bool = False
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | Required | Text to encode |
| `is_query` | `bool` | `False` | Use query prompt optimization |

**Returns:** `np.ndarray` - Shape `(dimension,)`

---

#### `encode_query()`

Encode queries with optimal settings for Qwen3.

```python
def encode_query(self, queries: List[str]) -> np.ndarray
```

Equivalent to `encode(queries, is_query=True)`

---

#### `encode_documents()`

Encode documents without query prompt.

```python
def encode_documents(self, documents: List[str]) -> np.ndarray
```

Equivalent to `encode(documents, is_query=False)`

---

## Data Models API

**Location:** `models/memory_entry.py`

Pydantic data models for the SimpleMem system.

### Class: `MemoryEntry`

Atomic memory entry - self-contained fact indexed across three layers.

```python
class MemoryEntry(BaseModel):
    """
    Atomic Entry - Self-contained memory unit
    Generated by: m_k = F_θ(W_t) = Φ_time ∘ Φ_coref ∘ Φ_extract(W_t)
    """
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entry_id` | `str` | `uuid.uuid4()` | Unique identifier |
| `lossless_restatement` | `str` | Required | Self-contained fact (no pronouns/relative time) |
| `keywords` | `List[str]` | `[]` | Core keywords for BM25 matching |
| `timestamp` | `Optional[str]` | `None` | ISO 8601 timestamp |
| `location` | `Optional[str]` | `None` | Location description |
| `persons` | `List[str]` | `[]` | Person names mentioned |
| `entities` | `List[str]` | `[]` | Entities (companies, products, etc.) |
| `topic` | `Optional[str]` | `None` | Topic phrase |

#### Example

```python
from models.memory_entry import MemoryEntry

entry = MemoryEntry(
    lossless_restatement="Alice discussed marketing strategy with Bob at Starbucks on 2025-11-15 at 14:30.",
    keywords=["Alice", "Bob", "marketing strategy", "Starbucks"],
    timestamp="2025-11-15T14:30:00",
    location="Starbucks, Shanghai",
    persons=["Alice", "Bob"],
    entities=["product XYZ"],
    topic="Product marketing strategy discussion"
)
```

---

### Class: `Dialogue`

Original dialogue entry input format.

```python
class Dialogue(BaseModel):
    """
    Original dialogue entry
    """
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dialogue_id` | `int` | Required | Unique dialogue identifier |
| `speaker` | `str` | Required | Speaker name |
| `content` | `str` | Required | Dialogue content |
| `timestamp` | `Optional[str]` | `None` | ISO 8601 timestamp |

#### Methods

```python
def __str__(self) -> str:
    """Format as '[timestamp] speaker: content'"""
```

#### Example

```python
from models.memory_entry import Dialogue

dialogue = Dialogue(
    dialogue_id=1,
    speaker="Alice",
    content="Let's meet at Starbucks tomorrow at 2pm",
    timestamp="2025-11-15T14:30:00"
)

print(dialogue)  # [2025-11-15T14:30:00] Alice: Let's meet at Starbucks tomorrow at 2pm
```

---

## Configuration Reference

**Location:** `config.py` (copy from `config.py.example`)

### LLM Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `OPENAI_API_KEY` | `str` | Required | OpenAI API key |
| `OPENAI_BASE_URL` | `str` | `None` | Custom OpenAI base URL |
| `LLM_MODEL` | `str` | `"gpt-4o-mini"` | LLM model name |
| `EMBEDDING_MODEL` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Local embedding model |

### Advanced LLM Features

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_THINKING` | `bool` | `False` | Enable deep thinking mode (Qwen) |
| `USE_STREAMING` | `bool` | `False` | Enable streaming responses |
| `USE_JSON_FORMAT` | `bool` | `True` | Enable JSON format mode |

### Memory Building Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `WINDOW_SIZE` | `int` | `10` | Dialogues per processing window |
| `OVERLAP_SIZE` | `int` | `2` | Window overlap for context |

### Retrieval Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `SEMANTIC_TOP_K` | `int` | `5` | Max semantic search results |
| `KEYWORD_TOP_K` | `int` | `3` | Max keyword search results |
| `STRUCTURED_TOP_K` | `int` | `10` | Max structured search results |

### Database Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `LANCEDB_PATH` | `str` | `"./lancedb_data"` | Database storage path |
| `MEMORY_TABLE_NAME` | `str` | `"memory_entries"` | Memory table name |

### Parallel Processing (Runtime)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_PARALLEL_PROCESSING` | `bool` | `True` | Parallel memory building |
| `MAX_PARALLEL_WORKERS` | `int` | `4` | Max workers for building |
| `ENABLE_PARALLEL_RETRIEVAL` | `bool` | `True` | Parallel query execution |
| `MAX_RETRIEVAL_WORKERS` | `int` | `3` | Max workers for retrieval |

### Retrieval Features (Runtime)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_PLANNING` | `bool` | `True` | Multi-query planning |
| `ENABLE_REFLECTION` | `bool` | `True` | Reflection loops |
| `MAX_REFLECTION_ROUNDS` | `int` | `2` | Max reflection iterations |

---

## Error Handling

### Common Exceptions

#### LLM API Errors

```python
try:
    response = client.chat_completion(messages)
except Exception as e:
    # Automatic retry with exponential backoff
    # After max_retries, raises the last exception
    print(f"LLM API failed: {e}")
```

#### JSON Parsing Errors

```python
try:
    data = client.extract_json(response)
except ValueError as e:
    # JSON extraction failed
    print(f"Failed to parse JSON: {e}")
```

#### Embedding Errors

```python
# EmbeddingModel automatically falls back to MiniLM-L6-v2 if Qwen3 loading fails
model = EmbeddingModel(model_name="qwen3-8b")
# If Qwen3 fails, model.model_name will be "sentence-transformers/all-MiniLM-L6-v2"
```

### Retry Mechanisms

| Component | Retries | Backoff | Fallback |
|-----------|---------|---------|----------|
| `LLMClient.chat_completion()` | 3 | Exponential (1s, 2s, 4s) | Raises exception |
| `MemoryBuilder._generate_memory_entries()` | 3 | None | Returns empty list |
| `AnswerGenerator.generate_answer()` | 3 | None | Returns raw response |
| `HybridRetriever` (all LLM calls) | 3 | None | Returns defaults |
| `MemoryBuilder.add_dialogues_parallel()` | 1 | None | Sequential fallback |
| `HybridRetriever._execute_parallel_searches()` | 1 | None | Sequential fallback |

---

## Usage Examples

### Example 1: Basic Q&A System

```python
from main import create_system

# Initialize
system = create_system(clear_db=True)

# Add conversation
system.add_dialogue("Alice", "I'll be traveling to Tokyo next month for a conference.", "2025-01-15T10:00:00")
system.add_dialogue("Bob", "That sounds exciting! When exactly?", "2025-01-15T10:01:00")
system.add_dialogue("Alice", "March 15th to 20th. The conference is at Tokyo Big Sight.", "2025-01-15T10:02:00")
system.add_dialogue("Bob", "I went there last year. The venue is amazing.", "2025-01-15T10:03:00")

# Finalize
system.finalize()

# Query
answer = system.ask("When is Alice's trip to Tokyo?")
print(answer)  # "15 to 20 March 2025"

answer = system.ask("Where is the conference?")
print(answer)  # "Tokyo Big Sight"
```

### Example 2: Batch Processing with Parallel Execution

```python
from main import SimpleMemSystem
from models.memory_entry import Dialogue

# Initialize with parallel processing
system = SimpleMemSystem(
    clear_db=True,
    enable_parallel_processing=True,
    max_parallel_workers=4,
    enable_parallel_retrieval=True,
    max_retrieval_workers=3
)

# Create batch of dialogues
dialogues = []
for i in range(100):
    dialogues.append(Dialogue(
        dialogue_id=i+1,
        speaker="User" if i % 2 == 0 else "Agent",
        content=f"Message number {i+1}",
        timestamp=f"2025-01-15T{10 + i//60:02d}:{i%60:02d}:00"
    ))

# Batch add (automatically uses parallel processing for large batches)
system.add_dialogues(dialogues)

# Query with parallel retrieval
answer = system.ask("What was message number 50?")
```

### Example 3: Custom Retrieval Configuration

```python
from main import SimpleMemSystem

# Initialize with custom retrieval settings
system = SimpleMemSystem(
    clear_db=True,
    enable_planning=True,       # Enable multi-query planning
    enable_reflection=True,     # Enable reflection loops
    max_reflection_rounds=3     # Up to 3 reflection rounds
)

# Add dialogues...
# ...

# Retrieve with reflection disabled for adversarial questions
contexts = system.hybrid_retriever.retrieve(
    "What is Alice's favorite color?",  # Likely not in memory
    enable_reflection=False              # Don't waste time reflecting
)

# Generate answer
if contexts:
    answer = system.answer_generator.generate_answer(
        "What is Alice's favorite color?",
        contexts
    )
else:
    answer = "No relevant information found"
```

### Example 4: Direct Component Access

```python
from utils.llm_client import LLMClient
from utils.embedding import EmbeddingModel
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry

# Initialize components individually
llm = LLMClient(model="gpt-4o-mini")
embedder = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = VectorStore(db_path="./my_db", embedding_model=embedder)

# Add entries directly
entries = [
    MemoryEntry(
        lossless_restatement="Alice prefers Python for data science projects.",
        keywords=["Alice", "Python", "data science"],
        persons=["Alice"],
        topic="Programming preferences"
    ),
    MemoryEntry(
        lossless_restatement="Bob uses JavaScript for web development.",
        keywords=["Bob", "JavaScript", "web development"],
        persons=["Bob"],
        topic="Programming preferences"
    )
]
store.add_entries(entries)

# Search directly
results = store.semantic_search("What programming language does Alice use?", top_k=3)
for r in results:
    print(r.lossless_restatement)
```

### Example 5: Using Different LLM Providers

```python
# OpenAI (default)
system_openai = SimpleMemSystem(
    api_key="sk-...",
    model="gpt-4o",
    base_url=None
)

# Qwen (Alibaba)
system_qwen = SimpleMemSystem(
    api_key="your-dashscope-key",
    model="qwen-plus-2025-07-28",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    enable_thinking=True,  # Qwen supports thinking mode
    use_streaming=True
)

# Local server (e.g., Ollama, vLLM)
system_local = SimpleMemSystem(
    api_key="not-needed",
    model="llama3.1:70b",
    base_url="http://localhost:11434/v1"
)
```

---

## API Versioning

Current API Version: **1.0.0**

The SimpleMem API follows semantic versioning:
- **MAJOR**: Breaking changes to public API
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

### Deprecated Features

None currently.

### Future Additions (Planned)

- Async API support (`async def add_dialogue()`, etc.)
- Batch query API (`ask_batch()`)
- Memory consolidation API
- Export/Import memory functions
