# SimpleMem Dependency Graph

This document provides a comprehensive analysis of all dependencies in the SimpleMem system, including external packages, internal module relationships, class dependencies, and initialization order.

---

## Table of Contents

1. [Overview](#overview)
2. [External Dependencies](#external-dependencies)
3. [Internal Module Structure](#internal-module-structure)
4. [Module Import Graph](#module-import-graph)
5. [Class Dependency Graph](#class-dependency-graph)
6. [Initialization Order](#initialization-order)
7. [Runtime Dependency Flows](#runtime-dependency-flows)
8. [Package Relationships](#package-relationships)
9. [Dependency Injection Patterns](#dependency-injection-patterns)
10. [Circular Dependency Analysis](#circular-dependency-analysis)
11. [Test Dependencies](#test-dependencies)
12. [Dependency Versions](#dependency-versions)

---

## Overview

SimpleMem has a layered dependency architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Dependency Architecture Overview                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         APPLICATION LAYER                               │ │
│  │  ┌──────────────┐                                                       │ │
│  │  │    main.py   │  SimpleMemSystem (orchestrator)                       │ │
│  │  │              │  Entry point, ties all components together            │ │
│  │  └──────┬───────┘                                                       │ │
│  │         │                                                               │ │
│  │         ▼                                                               │ │
│  └─────────┼───────────────────────────────────────────────────────────────┘ │
│            │                                                                 │
│  ┌─────────┼───────────────────────────────────────────────────────────────┐ │
│  │         │                   CORE LAYER                                  │ │
│  │  ┌──────┴───────────────────────────────────────────────────┐           │ │
│  │  │                                                          │           │ │
│  │  ▼                    ▼                     ▼               │           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │           │ │
│  │  │MemoryBuilder │  │HybridRetriever│  │AnswerGenerator│      │           │ │
│  │  │  (Stage 1)   │  │   (Stage 3)   │  │  (Stage 3)   │       │           │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │           │ │
│  │         │                 │                 │               │           │ │
│  └─────────┼─────────────────┼─────────────────┼───────────────┘           │
│            │                 │                 │                           │
│  ┌─────────┼─────────────────┼─────────────────┼───────────────────────────┐ │
│  │         │        INFRASTRUCTURE LAYER       │                           │ │
│  │         ▼                 ▼                 ▼                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │ VectorStore  │  │  LLMClient   │  │EmbeddingModel│                   │ │
│  │  │  (database/) │  │   (utils/)   │  │   (utils/)   │                   │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │ │
│  │         │                 │                 │                           │ │
│  └─────────┼─────────────────┼─────────────────┼───────────────────────────┘ │
│            │                 │                 │                            │
│  ┌─────────┼─────────────────┼─────────────────┼───────────────────────────┐ │
│  │         │         DATA MODEL LAYER          │                           │ │
│  │         ▼                 │                 ▼                           │ │
│  │  ┌──────────────┐         │          ┌──────────────┐                   │ │
│  │  │ MemoryEntry  │         │          │   Dialogue   │                   │ │
│  │  │  (models/)   │         │          │  (models/)   │                   │ │
│  │  └──────────────┘         │          └──────────────┘                   │ │
│  │                           │                                             │ │
│  └───────────────────────────┼─────────────────────────────────────────────┘ │
│                              │                                              │
│  ┌───────────────────────────┼─────────────────────────────────────────────┐ │
│  │                    EXTERNAL DEPENDENCIES                                │ │
│  │                           ▼                                             │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │ │
│  │  │ OpenAI  │ │ LanceDB │ │PyArrow  │ │Sentence │ │ Pydantic│           │ │
│  │  │   SDK   │ │         │ │         │ │Transform│ │         │           │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## External Dependencies

### Core Runtime Dependencies

These are the essential packages required for SimpleMem to function:

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `openai` | 2.3.0 | LLM API client | `utils/llm_client.py` |
| `lancedb` | 0.25.3 | Vector database | `database/vector_store.py` |
| `pyarrow` | 22.0.0 | Data serialization for LanceDB | `database/vector_store.py` |
| `sentence-transformers` | 5.1.1 | Embedding models | `utils/embedding.py` |
| `pydantic` | 2.12.0 | Data validation/models | `models/memory_entry.py` |
| `numpy` | 2.2.6 | Array operations | `utils/embedding.py`, `database/vector_store.py` |
| `dateparser` | 1.2.2 | Date parsing | `core/hybrid_retriever.py` |

### ML/Deep Learning Stack

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `torch` | 2.8.0 | Deep learning backend | `sentence-transformers`, `bert-score` |
| `transformers` | 4.57.0 | Transformer models | `sentence-transformers` |
| `accelerate` | 1.10.1 | Training optimization | `sentence-transformers` |
| `tokenizers` | 0.22.1 | Fast tokenization | `transformers` |
| `safetensors` | 0.6.2 | Safe tensor storage | `transformers` |
| `huggingface-hub` | 0.35.3 | Model downloading | `sentence-transformers` |

### Evaluation/Testing Dependencies

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `nltk` | 3.9.2 | Natural language processing | `test_locomo10.py` |
| `rouge-score` | 0.1.2 | ROUGE metrics | `test_locomo10.py` |
| `bert-score` | 0.3.13 | BERTScore metrics | `test_locomo10.py` |
| `scikit-learn` | 1.7.2 | Machine learning utilities | Evaluation metrics |

### Transitive Dependencies (Selected)

| Package | Required By | Purpose |
|---------|-------------|---------|
| `httpx` | `openai` | HTTP client |
| `anyio` | `openai` | Async I/O |
| `tqdm` | `sentence-transformers` | Progress bars |
| `pandas` | `lancedb` | DataFrame operations |
| `scipy` | `scikit-learn` | Scientific computing |

### Dependency Tree (Core)

```
simplemem
├── openai==2.3.0
│   ├── httpx>=0.23.0
│   ├── pydantic>=1.9.0
│   └── anyio>=3.5.0
├── lancedb==0.25.3
│   ├── pyarrow>=12.0.0
│   ├── pandas>=1.4.0
│   └── lance-namespace>=0.0.20
├── sentence-transformers==5.1.1
│   ├── torch>=1.11.0
│   ├── transformers>=4.34.0
│   ├── huggingface-hub>=0.15.1
│   └── numpy>=1.20.0
├── pydantic==2.12.0
│   └── pydantic-core>=2.20.0
└── dateparser==1.2.2
    ├── python-dateutil>=2.8.2
    └── pytz>=2021.1
```

---

## Internal Module Structure

### Directory Layout

```
simplemem/
├── main.py                    # Application entry point
├── config.py                  # Configuration (user-created)
├── config.py.example          # Configuration template
│
├── core/                      # Core business logic
│   ├── __init__.py           # Package exports
│   ├── memory_builder.py     # Stage 1: Memory compression
│   ├── hybrid_retriever.py   # Stage 3: Query retrieval
│   └── answer_generator.py   # Stage 3: Answer synthesis
│
├── database/                  # Data persistence
│   └── vector_store.py       # LanceDB vector storage
│
├── models/                    # Data models
│   ├── __init__.py           # Package exports
│   └── memory_entry.py       # MemoryEntry, Dialogue
│
├── utils/                     # Utilities
│   ├── __init__.py           # Package exports
│   ├── llm_client.py         # LLM API client
│   └── embedding.py          # Embedding generation
│
├── test_locomo10.py          # Benchmark testing
└── test_ref/                  # Reference testing
    ├── load_dataset.py       # Dataset loading utilities
    ├── test_advanced.py      # Advanced testing
    └── utils.py              # Evaluation utilities
```

### Package Exports

#### `core/__init__.py`
```python
from .memory_builder import MemoryBuilder
from .hybrid_retriever import HybridRetriever
from .answer_generator import AnswerGenerator

__all__ = ['MemoryBuilder', 'HybridRetriever', 'AnswerGenerator']
```

#### `models/__init__.py`
```python
from .memory_entry import MemoryEntry, Dialogue

__all__ = ['MemoryEntry', 'Dialogue']
```

#### `utils/__init__.py`
```python
from .llm_client import LLMClient
from .embedding import EmbeddingModel

__all__ = ['LLMClient', 'EmbeddingModel']
```

---

## Module Import Graph

### Import Relationships Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Module Import Graph                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────┐                                    │
│                           │   main.py   │                                    │
│                           └──────┬──────┘                                    │
│                                  │                                           │
│          ┌───────────────────────┼───────────────────────┐                   │
│          │                       │                       │                   │
│          ▼                       ▼                       ▼                   │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐            │
│  │models/        │      │utils/         │      │core/          │            │
│  │memory_entry.py│      │llm_client.py  │      │memory_builder │            │
│  │               │◄─────│embedding.py   │◄─────│hybrid_retriever│           │
│  │               │      │               │      │answer_generator│           │
│  └───────┬───────┘      └───────┬───────┘      └───────┬───────┘            │
│          │                      │                      │                    │
│          │                      │                      ▼                    │
│          │                      │              ┌───────────────┐            │
│          │                      └──────────────│database/      │            │
│          │                                     │vector_store.py│            │
│          │                                     └───────────────┘            │
│          │                                            ▲                     │
│          └────────────────────────────────────────────┘                     │
│                                                                              │
│  Legend:                                                                     │
│  ─────► imports from                                                         │
│  ◄───── imported by                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Import Matrix

| Module | Imports | Imported By |
|--------|---------|-------------|
| `main.py` | `models.memory_entry`, `utils.llm_client`, `utils.embedding`, `database.vector_store`, `core.memory_builder`, `core.hybrid_retriever`, `core.answer_generator`, `config` | - |
| `core/memory_builder.py` | `models.memory_entry`, `utils.llm_client`, `database.vector_store`, `config` | `main.py` |
| `core/hybrid_retriever.py` | `models.memory_entry`, `utils.llm_client`, `database.vector_store`, `config`, `dateparser` | `main.py` |
| `core/answer_generator.py` | `models.memory_entry`, `utils.llm_client`, `config` | `main.py` |
| `database/vector_store.py` | `models.memory_entry`, `utils.embedding`, `config`, `lancedb`, `pyarrow`, `numpy` | `main.py`, `core/memory_builder.py`, `core/hybrid_retriever.py` |
| `utils/llm_client.py` | `config`, `openai` | `main.py`, `core/memory_builder.py`, `core/hybrid_retriever.py`, `core/answer_generator.py` |
| `utils/embedding.py` | `config`, `numpy`, `sentence_transformers` | `main.py`, `database/vector_store.py` |
| `models/memory_entry.py` | `pydantic`, `uuid` | `main.py`, `core/*`, `database/vector_store.py` |
| `config.py` | - | All modules |

### Import Statement Details

#### `main.py`
```python
from typing import List, Optional
from models.memory_entry import Dialogue, MemoryEntry
from utils.llm_client import LLMClient
from utils.embedding import EmbeddingModel
from database.vector_store import VectorStore
from core.memory_builder import MemoryBuilder
from core.hybrid_retriever import HybridRetriever
from core.answer_generator import AnswerGenerator
import config
```

#### `core/memory_builder.py`
```python
from typing import List, Optional
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config
import json
import asyncio
import concurrent.futures
from functools import partial
```

#### `core/hybrid_retriever.py`
```python
from typing import List, Optional, Dict, Any
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config
import re
from datetime import datetime, timedelta
import dateparser
import concurrent.futures
```

#### `core/answer_generator.py`
```python
from typing import List
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
import config
```

#### `database/vector_store.py`
```python
from typing import List, Optional, Dict, Any
import lancedb
import pyarrow as pa
import numpy as np
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config
import os
```

#### `utils/llm_client.py`
```python
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import config
```

#### `utils/embedding.py`
```python
from typing import List, Optional, Dict, Any
import numpy as np
import config
import os
# Conditional import:
from sentence_transformers import SentenceTransformer
```

#### `models/memory_entry.py`
```python
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
```

---

## Class Dependency Graph

### Class Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Class Dependency Graph                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌─────────────────┐                                   │
│                        │ SimpleMemSystem │                                   │
│                        │     (main.py)   │                                   │
│                        └────────┬────────┘                                   │
│                                 │                                            │
│          ┌──────────────────────┼──────────────────────┐                     │
│          │ has-a                │ has-a                │ has-a               │
│          ▼                      ▼                      ▼                     │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐              │
│  │ MemoryBuilder │     │HybridRetriever│     │AnswerGenerator│              │
│  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘              │
│          │                     │                     │                       │
│          │uses                 │uses                 │uses                   │
│          ▼                     ▼                     ▼                       │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐              │
│  │  LLMClient    │◄────│  VectorStore  │────►│  LLMClient    │              │
│  └───────────────┘     └───────┬───────┘     └───────────────┘              │
│                                │                                             │
│                                │uses                                         │
│                                ▼                                             │
│                        ┌───────────────┐                                     │
│                        │EmbeddingModel │                                     │
│                        └───────────────┘                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       Data Model Classes                                ││
│  │                                                                         ││
│  │  ┌───────────────┐                   ┌───────────────┐                  ││
│  │  │  MemoryEntry  │  (Pydantic)       │   Dialogue    │  (Pydantic)      ││
│  │  │               │                   │               │                  ││
│  │  │ - entry_id    │                   │ - dialogue_id │                  ││
│  │  │ - lossless_   │                   │ - speaker     │                  ││
│  │  │   restatement │                   │ - content     │                  ││
│  │  │ - keywords    │                   │ - timestamp   │                  ││
│  │  │ - timestamp   │                   │               │                  ││
│  │  │ - location    │                   │               │                  ││
│  │  │ - persons     │                   │               │                  ││
│  │  │ - entities    │                   │               │                  ││
│  │  │ - topic       │                   │               │                  ││
│  │  └───────────────┘                   └───────────────┘                  ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Composition Relationships

| Container Class | Composed Classes | Relationship |
|-----------------|------------------|--------------|
| `SimpleMemSystem` | `LLMClient` | has-a (owns) |
| `SimpleMemSystem` | `EmbeddingModel` | has-a (owns) |
| `SimpleMemSystem` | `VectorStore` | has-a (owns) |
| `SimpleMemSystem` | `MemoryBuilder` | has-a (owns) |
| `SimpleMemSystem` | `HybridRetriever` | has-a (owns) |
| `SimpleMemSystem` | `AnswerGenerator` | has-a (owns) |
| `MemoryBuilder` | `LLMClient` | uses (injected) |
| `MemoryBuilder` | `VectorStore` | uses (injected) |
| `HybridRetriever` | `LLMClient` | uses (injected) |
| `HybridRetriever` | `VectorStore` | uses (injected) |
| `AnswerGenerator` | `LLMClient` | uses (injected) |
| `VectorStore` | `EmbeddingModel` | has-a (owns or injected) |

### Inheritance Relationships

| Class | Inherits From | Notes |
|-------|---------------|-------|
| `MemoryEntry` | `pydantic.BaseModel` | Data validation |
| `Dialogue` | `pydantic.BaseModel` | Data validation |

---

## Initialization Order

### System Bootstrap Sequence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Initialization Order                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Configuration Loading                                                    │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ import config                                                        │ │
│     │ • Load OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL                   │ │
│     │ • Load EMBEDDING_MODEL                                               │ │
│     │ • Load WINDOW_SIZE, TOP_K parameters                                 │ │
│     │ • Load LANCEDB_PATH, MEMORY_TABLE_NAME                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  2. LLMClient Initialization                                                 │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ LLMClient(api_key, model, base_url, enable_thinking, use_streaming) │ │
│     │ • Create OpenAI client instance                                      │ │
│     │ • Configure streaming and thinking mode                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  3. EmbeddingModel Initialization                                            │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ EmbeddingModel(model_name, use_optimization)                         │ │
│     │ • Download/load model from HuggingFace (if needed)                   │ │
│     │ • Initialize SentenceTransformer                                     │ │
│     │ • Determine embedding dimension                                       │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  4. VectorStore Initialization                                               │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ VectorStore(db_path, embedding_model, table_name)                    │ │
│     │ • Create database directory                                          │ │
│     │ • Connect to LanceDB                                                 │ │
│     │ • Initialize table schema (if not exists)                            │ │
│     │ • Optional: Clear existing data                                       │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  5. MemoryBuilder Initialization                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ MemoryBuilder(llm_client, vector_store, window_size, ...)           │ │
│     │ • Store references to LLMClient and VectorStore                      │ │
│     │ • Initialize dialogue buffer                                         │ │
│     │ • Configure parallel processing settings                             │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  6. HybridRetriever Initialization                                           │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ HybridRetriever(llm_client, vector_store, top_k values, ...)        │ │
│     │ • Store references to LLMClient and VectorStore                      │ │
│     │ • Configure planning and reflection settings                         │ │
│     │ • Configure parallel retrieval settings                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  7. AnswerGenerator Initialization                                           │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ AnswerGenerator(llm_client)                                          │ │
│     │ • Store reference to LLMClient                                       │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  8. System Ready                                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │ SimpleMemSystem fully initialized                                    │ │
│     │ • Ready to accept dialogues via add_dialogue()                       │ │
│     │ • Ready to answer queries via ask()                                  │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Initialization Code Flow

```python
# In SimpleMemSystem.__init__()

# Step 1: Already imported via: import config

# Step 2: LLMClient initialization
self.llm_client = LLMClient(
    api_key=api_key,              # from config if None
    model=model,                   # from config if None
    base_url=base_url,            # from config if None
    enable_thinking=enable_thinking,
    use_streaming=use_streaming
)

# Step 3: EmbeddingModel initialization
self.embedding_model = EmbeddingModel()  # uses config.EMBEDDING_MODEL

# Step 4: VectorStore initialization (depends on EmbeddingModel)
self.vector_store = VectorStore(
    db_path=db_path,               # from config if None
    embedding_model=self.embedding_model,
    table_name=table_name          # from config if None
)

# Optional: clear database
if clear_db:
    self.vector_store.clear()

# Step 5: MemoryBuilder initialization (depends on LLMClient, VectorStore)
self.memory_builder = MemoryBuilder(
    llm_client=self.llm_client,
    vector_store=self.vector_store,
    enable_parallel_processing=enable_parallel_processing,
    max_parallel_workers=max_parallel_workers
)

# Step 6: HybridRetriever initialization (depends on LLMClient, VectorStore)
self.hybrid_retriever = HybridRetriever(
    llm_client=self.llm_client,
    vector_store=self.vector_store,
    enable_planning=enable_planning,
    enable_reflection=enable_reflection,
    max_reflection_rounds=max_reflection_rounds,
    enable_parallel_retrieval=enable_parallel_retrieval,
    max_retrieval_workers=max_retrieval_workers
)

# Step 7: AnswerGenerator initialization (depends on LLMClient)
self.answer_generator = AnswerGenerator(
    llm_client=self.llm_client
)
```

### Dependency Order Requirements

| Component | Must Initialize Before | Reason |
|-----------|----------------------|--------|
| `config` | Everything | Provides all default values |
| `LLMClient` | `MemoryBuilder`, `HybridRetriever`, `AnswerGenerator` | Required for LLM calls |
| `EmbeddingModel` | `VectorStore` | Required for embedding dimension in schema |
| `VectorStore` | `MemoryBuilder`, `HybridRetriever` | Required for storage operations |

---

## Runtime Dependency Flows

### Write Path Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Write Path Runtime Dependencies                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Input                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────┐                                                        │
│  │ SimpleMemSystem │                                                        │
│  │ .add_dialogue() │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │  MemoryBuilder  │                                                        │
│  │  .add_dialogue()│                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           │ When buffer is full:                                            │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │  MemoryBuilder  │─────►│   LLMClient     │                              │
│  │ .process_window │      │.chat_completion │                              │
│  └────────┬────────┘      └────────┬────────┘                              │
│           │                        │                                        │
│           │                        ▼                                        │
│           │               ┌─────────────────┐                              │
│           │               │   OpenAI API    │                              │
│           │               └─────────────────┘                              │
│           │                        │                                        │
│           │◄───────────────────────┘                                        │
│           │ JSON → MemoryEntry[]                                            │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │   VectorStore   │─────►│ EmbeddingModel  │                              │
│  │  .add_entries() │      │ .encode_documents│                             │
│  └────────┬────────┘      └────────┬────────┘                              │
│           │                        │                                        │
│           │                        ▼                                        │
│           │               ┌─────────────────┐                              │
│           │               │SentenceTransformer│                            │
│           │               │    .encode()     │                              │
│           │               └─────────────────┘                              │
│           │                        │                                        │
│           │◄───────────────────────┘                                        │
│           │ vectors                                                         │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    LanceDB      │                                                        │
│  │  table.add()    │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Read Path Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Read Path Runtime Dependencies                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Query                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────┐                                                        │
│  │ SimpleMemSystem │                                                        │
│  │     .ask()      │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │ HybridRetriever │─────►│   LLMClient     │  (Planning)                   │
│  │   .retrieve()   │      │.chat_completion │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                  │
│           │ Generate search queries                                         │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │   VectorStore   │─────►│ EmbeddingModel  │                              │
│  │.semantic_search │      │  .encode_query  │                              │
│  └────────┬────────┘      └────────┬────────┘                              │
│           │                        │                                        │
│           │                        ▼                                        │
│           │               ┌─────────────────┐                              │
│           │               │    LanceDB      │                              │
│           │               │  table.search() │                              │
│           │               └─────────────────┘                              │
│           │◄───────────────────────┘                                        │
│           │                                                                  │
│           │ (Optional Reflection Loop)                                      │
│           │─────────►LLMClient.chat_completion                              │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ AnswerGenerator │                                                        │
│  │.generate_answer │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │   LLMClient     │─────►│   OpenAI API    │                              │
│  │.chat_completion │      │                 │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                  │
│           ▼                                                                  │
│       Answer String                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Package Relationships

### Internal Package Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Package Dependency Diagram                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           simplemem (root)                              │ │
│  │                                                                         │ │
│  │   ┌─────────┐                                                          │ │
│  │   │ main.py │                                                          │ │
│  │   └────┬────┘                                                          │ │
│  │        │                                                               │ │
│  │        │ imports                                                       │ │
│  │        ▼                                                               │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                                                                 │  │ │
│  │   │  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐   │  │ │
│  │   │  │ models/ │     │  core/  │     │database/│     │ utils/  │   │  │ │
│  │   │  └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘   │  │ │
│  │   │       │               │               │               │        │  │ │
│  │   │       │               │               │               │        │  │ │
│  │   │       ▼               ▼               ▼               ▼        │  │ │
│  │   │  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐   │  │ │
│  │   │  │memory_  │     │memory_  │     │vector_  │     │llm_     │   │  │ │
│  │   │  │entry.py │◄────│builder. │────►│store.py │◄────│client.py│   │  │ │
│  │   │  │         │     │py       │     │         │     │         │   │  │ │
│  │   │  └─────────┘     │         │     └────┬────┘     └─────────┘   │  │ │
│  │   │       ▲          │hybrid_  │          │               ▲        │  │ │
│  │   │       │          │retriever│          │               │        │  │ │
│  │   │       │          │.py      │          │          ┌─────────┐   │  │ │
│  │   │       │          │         │          └─────────►│embedding│   │  │ │
│  │   │       │          │answer_  │                     │.py      │   │  │ │
│  │   │       │          │generator│                     └─────────┘   │  │ │
│  │   │       │          │.py      │                                   │  │ │
│  │   │       │          └────┬────┘                                   │  │ │
│  │   │       │               │                                        │  │ │
│  │   │       └───────────────┘                                        │  │ │
│  │   │                                                                │  │ │
│  │   └────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Arrows indicate import direction (A ──► B means A imports from B)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### External Package Usage by Module

| Module | External Packages |
|--------|-------------------|
| `main.py` | `typing` |
| `core/memory_builder.py` | `typing`, `json`, `asyncio`, `concurrent.futures`, `functools` |
| `core/hybrid_retriever.py` | `typing`, `re`, `datetime`, `dateparser`, `concurrent.futures` |
| `core/answer_generator.py` | `typing` |
| `database/vector_store.py` | `typing`, `lancedb`, `pyarrow`, `numpy`, `os` |
| `utils/llm_client.py` | `json`, `typing`, `openai`, `time` |
| `utils/embedding.py` | `typing`, `numpy`, `os`, `sentence_transformers` |
| `models/memory_entry.py` | `typing`, `pydantic`, `uuid` |

---

## Dependency Injection Patterns

### Constructor Injection

SimpleMem uses constructor injection for all major dependencies:

```python
# SimpleMemSystem creates dependencies and injects them
class SimpleMemSystem:
    def __init__(self, ...):
        # Create infrastructure
        self.llm_client = LLMClient(...)
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(..., embedding_model=self.embedding_model)

        # Inject into core components
        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,      # Injected
            vector_store=self.vector_store   # Injected
        )
        self.hybrid_retriever = HybridRetriever(
            llm_client=self.llm_client,      # Injected
            vector_store=self.vector_store   # Injected
        )
        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client       # Injected
        )
```

### Dependency Injection Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Dependency Injection Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SimpleMemSystem (Composition Root)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  Creates:                                                               ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │   LLMClient   │  │EmbeddingModel │  │  VectorStore  │               ││
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘               ││
│  │          │                  │                  │                        ││
│  │          │                  └─────────────────►│                        ││
│  │          │                    injected into    │                        ││
│  │          │                                     │                        ││
│  │  Injects into:                                 │                        ││
│  │          │                                     │                        ││
│  │          ▼                                     ▼                        ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │ MemoryBuilder │  │HybridRetriever│  │AnswerGenerator│               ││
│  │  │               │  │               │  │               │               ││
│  │  │ llm_client ◄──┼──┼──┬────────────┼──┼───────────────┘               ││
│  │  │ vector_store◄─┼──┼──┼────────────┘  │                               ││
│  │  │               │  │  │               │                               ││
│  │  └───────────────┘  │  └───────────────┘                               ││
│  │                     │                                                   ││
│  │                     │  Both MemoryBuilder and HybridRetriever          ││
│  │                     │  share the same LLMClient and VectorStore        ││
│  │                     │  instances (singleton pattern)                    ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits of This Pattern

1. **Testability**: Components can be tested with mock dependencies
2. **Flexibility**: Different implementations can be swapped
3. **Single Responsibility**: Each class has focused concerns
4. **Shared Resources**: LLMClient is shared, avoiding duplicate API connections

---

## Circular Dependency Analysis

### Current Status: No Circular Dependencies

The SimpleMem codebase has a clean, acyclic dependency graph:

```
models/ ◄── core/ ◄── main.py
   ▲          │
   │          │
   │          ▼
   └───── database/
              │
              ▼
          utils/
```

### Dependency Direction Rules

| Layer | Can Import From | Cannot Import From |
|-------|-----------------|-------------------|
| `main.py` | All packages | - |
| `core/` | `models/`, `utils/`, `database/`, `config` | `main.py` |
| `database/` | `models/`, `utils/`, `config` | `main.py`, `core/` |
| `utils/` | `config` | `main.py`, `core/`, `database/`, `models/` |
| `models/` | External only (pydantic) | All internal modules |

### Preventing Circular Dependencies

1. **Unidirectional Flow**: Lower layers never import from higher layers
2. **Config Isolation**: `config.py` has no internal dependencies
3. **Model Independence**: Data models have no application logic imports
4. **Interface Segregation**: Classes depend on interfaces, not implementations

---

## Test Dependencies

### `test_locomo10.py` Dependencies

```python
# Standard library
from pathlib import Path
import time
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import statistics
from collections import defaultdict

# NLP/ML Libraries
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# Internal
from main import SimpleMemSystem
from models.memory_entry import Dialogue
```

### `test_ref/` Dependencies

| File | Internal Dependencies | External Dependencies |
|------|----------------------|----------------------|
| `load_dataset.py` | None | `json`, `typing`, `dataclasses`, `pathlib` |
| `utils.py` | `load_dataset` | `nltk`, `rouge_score`, `bert_score`, `sentence_transformers`, `numpy`, `statistics` |
| `test_advanced.py` | `load_dataset`, `utils` | `memory_layer` (external), `openai`, `numpy`, `pickle`, `tqdm`, `argparse`, `logging` |

### Test Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Test Dependency Graph                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ test_locomo10.py│                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ├──────────────────────────────────────────────────────┐          │
│           │                                                      │          │
│           ▼                                                      ▼          │
│  ┌─────────────────┐                                    ┌─────────────────┐ │
│  │      main.py    │                                    │ Evaluation Libs │ │
│  │ (SimpleMemSystem)│                                   │  • nltk         │ │
│  └─────────────────┘                                    │  • rouge_score  │ │
│                                                         │  • bert_score   │ │
│                                                         │  • sentence_    │ │
│  ┌─────────────────┐                                    │    transformers │ │
│  │   test_ref/     │                                    └─────────────────┘ │
│  │                 │                                                        │
│  │  load_dataset.py│◄──────┐                                               │
│  │  utils.py       │───────┤                                               │
│  │  test_advanced. │───────┘                                               │
│  │      py         │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Versions

### Version Compatibility Matrix

| Package | Min Version | Max Tested | Notes |
|---------|-------------|------------|-------|
| Python | 3.8 | 3.12 | Type hints, async features |
| `openai` | 1.0.0 | 2.3.0 | New API structure in 1.0+ |
| `lancedb` | 0.20.0 | 0.25.3 | API stability |
| `sentence-transformers` | 4.0.0 | 5.1.1 | Qwen3 support in 5.x |
| `pydantic` | 2.0.0 | 2.12.0 | v2 model syntax |
| `torch` | 2.0.0 | 2.8.0 | CUDA compatibility |
| `numpy` | 1.24.0 | 2.2.6 | Array API changes |

### Critical Version Constraints

```
# requirements.txt (key constraints)
openai>=1.0.0,<3.0.0        # v1 API required
lancedb>=0.20.0             # Modern schema support
sentence-transformers>=4.0  # Query prompt support
pydantic>=2.0.0             # v2 model syntax
torch>=2.0.0                # Modern features
```

### Version Update Impact

| Package Update | Impact | Migration Notes |
|---------------|--------|-----------------|
| `openai` 1.x → 2.x | Low | API compatible |
| `pydantic` 1.x → 2.x | High | Model syntax changes |
| `lancedb` major | Medium | Schema may need update |
| `sentence-transformers` 4.x → 5.x | Low | New models available |
| `torch` major | Medium | CUDA version alignment |

---

## Appendix: Complete Dependency List

### Direct Dependencies (Production)

```
openai==2.3.0
lancedb==0.25.3
pyarrow==22.0.0
sentence-transformers==5.1.1
pydantic==2.12.0
numpy==2.2.6
dateparser==1.2.2
```

### Direct Dependencies (Testing)

```
nltk==3.9.2
rouge-score==0.1.2
bert-score==0.3.13
scikit-learn==1.7.2
tqdm==4.67.1
```

### Full Dependency Count

| Category | Count |
|----------|-------|
| Direct (production) | 7 |
| Direct (testing) | 5 |
| Transitive | ~130 |
| **Total** | ~145 |

### Dependency Size Impact

| Package | Installed Size | Notes |
|---------|---------------|-------|
| `torch` | ~2.5 GB | Largest dependency |
| `transformers` | ~500 MB | Model files |
| `sentence-transformers` | ~100 MB | + model downloads |
| `lancedb` | ~50 MB | Native extensions |
| Others | ~200 MB | Combined |
| **Total** | ~3.5 GB | Approximate |

### Reducing Dependency Footprint

For minimal deployment:

```python
# Minimal requirements (no testing, no CUDA)
openai>=1.0.0
lancedb>=0.20.0
pyarrow>=12.0.0
sentence-transformers>=4.0.0
pydantic>=2.0.0
numpy>=1.24.0
dateparser>=1.0.0
```

For CPU-only deployment, add:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch  # CPU-only version (~200 MB vs 2.5 GB)
```
