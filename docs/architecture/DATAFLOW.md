# SimpleMem Data Flow Documentation

This document provides a comprehensive reference for all data flows in the SimpleMem system, including detailed transformations, state changes, and data formats at each stage.

---

## Table of Contents

1. [Overview](#overview)
2. [Write Path: Memory Building](#write-path-memory-building)
3. [Read Path: Query Processing](#read-path-query-processing)
4. [Data Transformations](#data-transformations)
5. [Parallel Processing Flows](#parallel-processing-flows)
6. [LLM Interaction Flows](#llm-interaction-flows)
7. [Storage Flows](#storage-flows)
8. [Error Handling Flows](#error-handling-flows)

---

## Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SimpleMem Data Flow Overview                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            WRITE PATH                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  User Input       Memory Builder        Vector Store     LanceDB    │    │
│  │  ──────────       ─────────────        ────────────     ────────    │    │
│  │                                                                      │    │
│  │  Dialogue ───────► Buffer ──────────► LLM ──────────► Embedding     │    │
│  │  (speaker,         (window_size)       Extraction       Generation  │    │
│  │   content,                              │                  │        │    │
│  │   timestamp)                            ▼                  ▼        │    │
│  │                                    MemoryEntry[] ────► Table Storage │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                            READ PATH                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  User Query    Hybrid Retriever   Answer Generator    Response      │    │
│  │  ──────────    ────────────────   ────────────────    ────────      │    │
│  │                                                                      │    │
│  │  Question ────► Planning ────────► Search ──────────► Synthesis     │    │
│  │                 │                   │                   │           │    │
│  │                 ▼                   ▼                   ▼           │    │
│  │            Sub-queries          MemoryEntry[]       JSON Response   │    │
│  │                 │                   │                   │           │    │
│  │                 └───────────────────┘                   ▼           │    │
│  │                      Reflection Loop              Answer String     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Types Summary

| Data Type | Location | Purpose |
|-----------|----------|---------|
| `Dialogue` | Input | Raw user dialogue turn |
| `List[Dialogue]` | Buffer | Accumulated dialogues awaiting processing |
| `str` (prompt) | LLM Input | Extraction/generation prompt |
| `str` (JSON) | LLM Output | Structured JSON response |
| `MemoryEntry` | Internal | Atomic memory entry |
| `List[MemoryEntry]` | Storage | Indexed memory entries |
| `np.ndarray` | Embedding | Dense vector representation |
| `str` (answer) | Output | Final answer to user |

---

## Write Path: Memory Building

### Complete Write Path Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WRITE PATH: Detailed Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: Input Reception                                              │   │
│  │                                                                       │   │
│  │  add_dialogue(speaker, content, timestamp)                           │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Dialogue Object Created                    │                       │   │
│  │  │  dialogue_id: auto-incremented            │                       │   │
│  │  │  speaker: str                              │                       │   │
│  │  │  content: str                              │                       │   │
│  │  │  timestamp: Optional[str]                  │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  dialogue_buffer.append(dialogue)                                    │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Check: len(buffer) >= window_size?         │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ├── Yes ──► process_window()                                   │   │
│  │       │                                                               │   │
│  │       └── No ───► Wait for more dialogues                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: Window Processing                                            │   │
│  │                                                                       │   │
│  │  process_window()                                                    │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Extract Window from Buffer                 │                       │   │
│  │  │  window = buffer[:window_size]            │                       │   │
│  │  │  buffer = buffer[window_size:]            │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  _generate_memory_entries(window)                                    │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Build Dialogue Text                        │                       │   │
│  │  │  dialogue_text = "\n".join(str(d))        │                       │   │
│  │  │                                            │                       │   │
│  │  │ Format: "[timestamp] speaker: content"    │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Build Context (from previous_entries)     │                       │   │
│  │  │  - Include first 3 previous entries       │                       │   │
│  │  │  - Purpose: Avoid duplication             │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: LLM Extraction (De-linearization F_θ)                       │   │
│  │                                                                       │   │
│  │  _build_extraction_prompt(dialogue_text, dialogue_ids, context)      │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Prompt Structure                           │                       │   │
│  │  │  [Previous Window Memory Entries]         │                       │   │
│  │  │  [Current Window Dialogues]               │                       │   │
│  │  │  [Requirements]                            │                       │   │
│  │  │  [Output Format]                           │                       │   │
│  │  │  [Example]                                 │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  llm_client.chat_completion(messages, temperature=0.1)               │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ LLM Response (JSON Array)                  │                       │   │
│  │  │  [                                         │                       │   │
│  │  │    {                                       │                       │   │
│  │  │      "lossless_restatement": "...",       │                       │   │
│  │  │      "keywords": [...],                   │                       │   │
│  │  │      "timestamp": "...",                  │                       │   │
│  │  │      "location": "...",                   │                       │   │
│  │  │      "persons": [...],                    │                       │   │
│  │  │      "entities": [...],                   │                       │   │
│  │  │      "topic": "..."                       │                       │   │
│  │  │    },                                      │                       │   │
│  │  │    ...                                     │                       │   │
│  │  │  ]                                         │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: Response Parsing                                             │   │
│  │                                                                       │   │
│  │  _parse_llm_response(response, dialogue_ids)                         │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  llm_client.extract_json(response)                                   │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ For each item in JSON array:               │                       │   │
│  │  │   Create MemoryEntry(                      │                       │   │
│  │  │     entry_id=uuid4(),                     │                       │   │
│  │  │     lossless_restatement=item[...],       │                       │   │
│  │  │     keywords=item.get("keywords", []),    │                       │   │
│  │  │     timestamp=item.get("timestamp"),      │                       │   │
│  │  │     location=item.get("location"),        │                       │   │
│  │  │     persons=item.get("persons", []),      │                       │   │
│  │  │     entities=item.get("entities", []),    │                       │   │
│  │  │     topic=item.get("topic")               │                       │   │
│  │  │   )                                        │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  List[MemoryEntry]                                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: Storage                                                      │   │
│  │                                                                       │   │
│  │  vector_store.add_entries(entries)                                   │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Generate Embeddings                        │                       │   │
│  │  │  restatements = [e.lossless_restatement]  │                       │   │
│  │  │  vectors = embedding_model.encode_documents│                       │   │
│  │  │            (restatements)                  │                       │   │
│  │  │                                            │                       │   │
│  │  │  Result: np.ndarray (N x dimension)       │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Build Data Records                         │                       │   │
│  │  │  for entry, vector in zip(entries, vectors):│                      │   │
│  │  │    {                                       │                       │   │
│  │  │      "entry_id": entry.entry_id,          │                       │   │
│  │  │      "lossless_restatement": ...,         │                       │   │
│  │  │      "keywords": [...],                   │                       │   │
│  │  │      "timestamp": "...",                  │                       │   │
│  │  │      "location": "...",                   │                       │   │
│  │  │      "persons": [...],                    │                       │   │
│  │  │      "entities": [...],                   │                       │   │
│  │  │      "topic": "...",                      │                       │   │
│  │  │      "vector": vector.tolist()            │                       │   │
│  │  │    }                                       │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  table.add(data)  ──► LanceDB Storage                                │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Update State:                                                       │   │
│  │    previous_entries = entries                                        │   │
│  │    processed_count += len(window)                                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Write Path Data Format Examples

#### Input: Dialogue

```python
Dialogue(
    dialogue_id=1,
    speaker="Alice",
    content="Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product",
    timestamp="2025-11-15T14:30:00"
)
```

**String Representation**:
```
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
```

#### Intermediate: LLM Prompt

```
Your task is to extract all valuable information from the following dialogues...

[Previous Window Memory Entries (for reference to avoid duplication)]
- Previous entry 1...
- Previous entry 2...

[Current Window Dialogues]
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm...
[2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials...

[Requirements]
1. **Complete Coverage**: Generate enough memory entries...
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns...
...
```

#### Intermediate: LLM Response (JSON)

```json
[
  {
    "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
    "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
    "timestamp": "2025-11-16T14:00:00",
    "location": "Starbucks",
    "persons": ["Alice", "Bob"],
    "entities": ["new product"],
    "topic": "Product discussion meeting arrangement"
  },
  {
    "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
    "keywords": ["Bob", "prepare materials", "agree"],
    "timestamp": null,
    "location": null,
    "persons": ["Bob"],
    "entities": [],
    "topic": "Meeting preparation confirmation"
  }
]
```

#### Output: MemoryEntry Objects

```python
MemoryEntry(
    entry_id="550e8400-e29b-41d4-a716-446655440000",
    lossless_restatement="Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
    keywords=["Alice", "Bob", "Starbucks", "new product", "meeting"],
    timestamp="2025-11-16T14:00:00",
    location="Starbucks",
    persons=["Alice", "Bob"],
    entities=["new product"],
    topic="Product discussion meeting arrangement"
)
```

#### Storage: LanceDB Record

```python
{
    "entry_id": "550e8400-e29b-41d4-a716-446655440000",
    "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00...",
    "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
    "timestamp": "2025-11-16T14:00:00",
    "location": "Starbucks",
    "persons": ["Alice", "Bob"],
    "entities": ["new product"],
    "topic": "Product discussion meeting arrangement",
    "vector": [0.0234, -0.0891, 0.1234, ...]  # 1024-dimensional
}
```

---

## Read Path: Query Processing

### Complete Read Path Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          READ PATH: Detailed Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: Query Reception                                              │   │
│  │                                                                       │   │
│  │  ask(question: str)                                                  │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Input: "When will Alice and Bob meet?"    │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  hybrid_retriever.retrieve(question)                                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: Information Requirements Analysis                            │   │
│  │                                                                       │   │
│  │  _analyze_information_requirements(query)                            │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ LLM Analysis Prompt                        │                       │   │
│  │  │  "Analyze the following question and      │                       │   │
│  │  │   determine what specific information     │                       │   │
│  │  │   is required to answer it..."            │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Information Plan (JSON)                    │                       │   │
│  │  │  {                                         │                       │   │
│  │  │    "question_type": "temporal",           │                       │   │
│  │  │    "key_entities": ["Alice", "Bob"],      │                       │   │
│  │  │    "required_info": [                     │                       │   │
│  │  │      {                                     │                       │   │
│  │  │        "info_type": "meeting_time",       │                       │   │
│  │  │        "description": "When they meet",   │                       │   │
│  │  │        "priority": "high"                 │                       │   │
│  │  │      }                                     │                       │   │
│  │  │    ],                                      │                       │   │
│  │  │    "relationships": ["Alice-Bob meeting"],│                       │   │
│  │  │    "minimal_queries_needed": 2            │                       │   │
│  │  │  }                                         │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: Targeted Query Generation                                    │   │
│  │                                                                       │   │
│  │  _generate_targeted_queries(query, information_plan)                 │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ LLM Query Generation                       │                       │   │
│  │  │  "Generate the minimal set of targeted    │                       │   │
│  │  │   search queries needed to gather the     │                       │   │
│  │  │   required information..."                │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Generated Queries                          │                       │   │
│  │  │  {                                         │                       │   │
│  │  │    "reasoning": "Need meeting time...",   │                       │   │
│  │  │    "queries": [                           │                       │   │
│  │  │      "When will Alice and Bob meet?",     │                       │   │
│  │  │      "Alice Bob meeting time",            │                       │   │
│  │  │      "meeting schedule Alice Bob"         │                       │   │
│  │  │    ]                                       │                       │   │
│  │  │  }                                         │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  search_queries = ["When will Alice and Bob meet?", ...]             │   │
│  │  (limited to max 4 queries)                                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: Parallel Search Execution                                    │   │
│  │                                                                       │   │
│  │  _execute_parallel_searches(search_queries)                          │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ ThreadPoolExecutor                         │                       │   │
│  │  │  max_workers = max_retrieval_workers      │                       │   │
│  │  │                                            │                       │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐     │                       │   │
│  │  │  │Worker 1 │ │Worker 2 │ │Worker 3 │     │                       │   │
│  │  │  │Query 1  │ │Query 2  │ │Query 3  │     │                       │   │
│  │  │  └────┬────┘ └────┬────┘ └────┬────┘     │                       │   │
│  │  │       │           │           │          │                       │   │
│  │  │       ▼           ▼           ▼          │                       │   │
│  │  │  _semantic_search() for each query       │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  For each query:                                                     │   │
│  │    _semantic_search(query)                                           │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Vector Search                              │                       │   │
│  │  │  1. query_vector = embedding_model        │                       │   │
│  │  │       .encode_single(query, is_query=True)│                       │   │
│  │  │  2. results = table.search(query_vector)  │                       │   │
│  │  │       .limit(semantic_top_k).to_list()    │                       │   │
│  │  │  3. Convert to List[MemoryEntry]          │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  all_results: List[MemoryEntry]                                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: Merge and Deduplication                                      │   │
│  │                                                                       │   │
│  │  _merge_and_deduplicate_entries(all_results)                         │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Deduplication by entry_id                  │                       │   │
│  │  │  seen_ids = set()                         │                       │   │
│  │  │  for entry in all_results:                │                       │   │
│  │  │    if entry.entry_id not in seen_ids:     │                       │   │
│  │  │      seen_ids.add(entry.entry_id)         │                       │   │
│  │  │      merged.append(entry)                 │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  merged_results: List[MemoryEntry]                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: Reflection Loop (Optional)                                   │   │
│  │                                                                       │   │
│  │  if enable_reflection:                                               │   │
│  │    _retrieve_with_intelligent_reflection(query, results, plan)       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ for round in range(max_reflection_rounds):│                       │   │
│  │  │   │                                        │                       │   │
│  │  │   ▼                                        │                       │   │
│  │  │  _analyze_information_completeness()      │                       │   │
│  │  │   │                                        │                       │   │
│  │  │   ├── "complete" ──► break                │                       │   │
│  │  │   │                                        │                       │   │
│  │  │   └── "incomplete" ──►                    │                       │   │
│  │  │        _generate_missing_info_queries()   │                       │   │
│  │  │        │                                   │                       │   │
│  │  │        ▼                                   │                       │   │
│  │  │       Execute additional searches          │                       │   │
│  │  │        │                                   │                       │   │
│  │  │        ▼                                   │                       │   │
│  │  │       Merge with current results          │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  final_results: List[MemoryEntry]                                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 7: Answer Generation                                            │   │
│  │                                                                       │   │
│  │  answer_generator.generate_answer(question, contexts)                │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  _format_contexts(contexts)                                          │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ Formatted Context String                   │                       │   │
│  │  │  [Context 1]                               │                       │   │
│  │  │  Content: Alice suggested at 2025-11-15...│                       │   │
│  │  │  Time: 2025-11-16T14:00:00                │                       │   │
│  │  │  Location: Starbucks                      │                       │   │
│  │  │  Persons: Alice, Bob                      │                       │   │
│  │  │                                            │                       │   │
│  │  │  [Context 2]                               │                       │   │
│  │  │  Content: Bob agreed to attend...         │                       │   │
│  │  │  ...                                       │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  _build_answer_prompt(query, context_str)                            │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  llm_client.chat_completion(messages, temperature=0.1)               │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌───────────────────────────────────────────┐                       │   │
│  │  │ LLM Response (JSON)                        │                       │   │
│  │  │  {                                         │                       │   │
│  │  │    "reasoning": "The context explicitly   │                       │   │
│  │  │      states the meeting time...",         │                       │   │
│  │  │    "answer": "16 November 2025 at 2:00 PM"│                       │   │
│  │  │  }                                         │                       │   │
│  │  └───────────────────────────────────────────┘                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  answer = result.get("answer")                                       │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  Return: "16 November 2025 at 2:00 PM"                               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Read Path Data Format Examples

#### Input: Question

```
"When will Alice and Bob meet?"
```

#### Intermediate: Information Plan

```json
{
  "question_type": "temporal",
  "key_entities": ["Alice", "Bob"],
  "required_info": [
    {
      "info_type": "meeting_time",
      "description": "The scheduled time for Alice and Bob's meeting",
      "priority": "high"
    }
  ],
  "relationships": ["Alice-Bob meeting"],
  "minimal_queries_needed": 2
}
```

#### Intermediate: Search Queries

```python
[
    "When will Alice and Bob meet?",
    "Alice Bob meeting time",
    "meeting schedule Alice Bob"
]
```

#### Intermediate: Retrieved MemoryEntries

```python
[
    MemoryEntry(
        entry_id="...",
        lossless_restatement="Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00...",
        timestamp="2025-11-16T14:00:00",
        location="Starbucks",
        persons=["Alice", "Bob"],
        ...
    ),
    MemoryEntry(
        entry_id="...",
        lossless_restatement="Bob agreed to attend the meeting...",
        ...
    )
]
```

#### Intermediate: Formatted Context

```
[Context 1]
Content: Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.
Time: 2025-11-16T14:00:00
Location: Starbucks
Persons: Alice, Bob
Related Entities: new product
Topic: Product discussion meeting arrangement

[Context 2]
Content: Bob agreed to attend the meeting and committed to prepare relevant materials.
Persons: Bob
Topic: Meeting preparation confirmation
```

#### Intermediate: Answer Prompt

```
Answer the user's question based on the provided context.

User Question: When will Alice and Bob meet?

Relevant Context:
[Context 1]
Content: Alice suggested at 2025-11-15T14:30:00...
...

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase)
...
```

#### Output: Answer

```
"16 November 2025 at 2:00 PM"
```

---

## Data Transformations

### Transformation Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Transformation Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WRITE PATH TRANSFORMATIONS                                                  │
│  ─────────────────────────────                                               │
│                                                                              │
│  T1: User Input → Dialogue                                                  │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ (speaker, content, timestamp) → Dialogue(dialogue_id, ...)     │         │
│  │ • Auto-generate dialogue_id                                     │         │
│  │ • Validate ISO 8601 timestamp format                           │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T2: Dialogue → String Representation                                       │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ Dialogue.__str__() → "[timestamp] speaker: content"            │         │
│  │ • Format: "[2025-11-15T14:30:00] Alice: Hello"                │         │
│  │ • Omit timestamp prefix if None                                │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T3: List[Dialogue] → Dialogue Text                                         │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ "\n".join([str(d) for d in dialogues])                        │         │
│  │ • Concatenate string representations                           │         │
│  │ • Newline-separated                                            │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T4: Dialogue Text → LLM Prompt                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ _build_extraction_prompt(dialogue_text, ids, context)         │         │
│  │ • Add context from previous entries                            │         │
│  │ • Add requirements and output format                           │         │
│  │ • Add example for few-shot learning                            │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T5: LLM Response → JSON Array                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ llm_client.extract_json(response) → List[Dict]                │         │
│  │ • Handle multiple JSON formats                                 │         │
│  │ • Clean trailing commas, comments                              │         │
│  │ • Extract balanced JSON structures                             │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T6: JSON Dict → MemoryEntry                                                │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ MemoryEntry(                                                   │         │
│  │   entry_id=uuid4(),                                           │         │
│  │   lossless_restatement=item["lossless_restatement"],          │         │
│  │   keywords=item.get("keywords", []),                          │         │
│  │   ...                                                          │         │
│  │ )                                                              │         │
│  │ • Auto-generate UUID for entry_id                             │         │
│  │ • Apply defaults for optional fields                           │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T7: MemoryEntry → Embedding Vector                                         │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ embedding_model.encode_documents([e.lossless_restatement])    │         │
│  │ → np.ndarray (N x dimension)                                   │         │
│  │ • Batch encoding for efficiency                                │         │
│  │ • Document encoding (no query prompt)                          │         │
│  │ • Normalized embeddings                                        │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T8: (MemoryEntry, Vector) → Storage Record                                 │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ {                                                              │         │
│  │   "entry_id": entry.entry_id,                                 │         │
│  │   "lossless_restatement": entry.lossless_restatement,         │         │
│  │   "keywords": entry.keywords,                                 │         │
│  │   "timestamp": entry.timestamp or "",                         │         │
│  │   "location": entry.location or "",                           │         │
│  │   "persons": entry.persons,                                   │         │
│  │   "entities": entry.entities,                                 │         │
│  │   "topic": entry.topic or "",                                 │         │
│  │   "vector": vector.tolist()                                   │         │
│  │ }                                                              │         │
│  │ • Convert None to empty strings for LanceDB                   │         │
│  │ • Convert numpy array to list                                  │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│                                                                              │
│  READ PATH TRANSFORMATIONS                                                   │
│  ────────────────────────────                                                │
│                                                                              │
│  T9: Query → Query Vector                                                   │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ embedding_model.encode_single(query, is_query=True)           │         │
│  │ → np.ndarray (dimension,)                                      │         │
│  │ • Single vector encoding                                       │         │
│  │ • Query prompt for Qwen3 (asymmetric retrieval)               │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T10: Query Vector → Search Results                                         │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ table.search(query_vector).limit(top_k).to_list()             │         │
│  │ → List[Dict] (raw LanceDB results)                            │         │
│  │ • Vector similarity search (cosine)                            │         │
│  │ • Returns top_k most similar entries                           │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T11: Search Result Dict → MemoryEntry                                      │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ MemoryEntry(                                                   │         │
│  │   entry_id=result["entry_id"],                                │         │
│  │   lossless_restatement=result["lossless_restatement"],        │         │
│  │   keywords=list(result["keywords"]) if result["keywords"]...  │         │
│  │   ...                                                          │         │
│  │ )                                                              │         │
│  │ • Convert arrays to lists                                      │         │
│  │ • Handle None/empty values                                     │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T12: List[MemoryEntry] → Formatted Context                                 │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ _format_contexts(contexts) → str                              │         │
│  │ • Format: "[Context N]\nContent: ...\nTime: ...\n..."         │         │
│  │ • Include non-empty metadata fields                            │         │
│  │ • Newline-separated entries                                    │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T13: (Query, Context) → Answer Prompt                                      │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ _build_answer_prompt(query, context_str) → str                │         │
│  │ • Include question and formatted context                       │         │
│  │ • Add requirements and output format                           │         │
│  │ • Add example for consistency                                  │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  T14: LLM Response → Answer                                                 │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ llm_client.extract_json(response) → Dict                      │         │
│  │ result.get("answer") → str                                    │         │
│  │ • Extract JSON from response                                   │         │
│  │ • Get "answer" field                                           │         │
│  │ • Fallback to raw response on parse failure                    │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### De-linearization Transformation (F_θ)

The core transformation that converts dialogues to atomic entries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    De-linearization Transformation F_θ                       │
│                                                                              │
│                    F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Dialogue Window W_t                                                  │
│  ──────────────────────────                                                  │
│  [2025-11-15T14:30:00] Alice: He'll meet Bob tomorrow at 2pm                │
│  [2025-11-15T14:31:00] Bob: I'll prepare the materials                      │
│                                                                              │
│                               │                                              │
│                               ▼                                              │
│                                                                              │
│  STEP 1: Φ_extract (Information Extraction)                                 │
│  ───────────────────────────────────────────                                 │
│  Extract raw facts from dialogue:                                           │
│  • Someone will meet Bob                                                    │
│  • Time: tomorrow at 2pm                                                    │
│  • Bob will prepare materials                                               │
│                                                                              │
│                               │                                              │
│                               ▼                                              │
│                                                                              │
│  STEP 2: Φ_coref (Coreference Resolution)                                   │
│  ─────────────────────────────────────────                                   │
│  Resolve all pronouns and references:                                       │
│  • "He" → "Alice" (speaker context)                                         │
│  • "I" → "Bob" (speaker)                                                    │
│  • All entities explicitly named                                            │
│                                                                              │
│                               │                                              │
│                               ▼                                              │
│                                                                              │
│  STEP 3: Φ_time (Temporal Anchoring)                                        │
│  ────────────────────────────────────                                        │
│  Convert relative to absolute time:                                         │
│  • "tomorrow" → "2025-11-16" (relative to dialogue timestamp)               │
│  • "at 2pm" → "T14:00:00"                                                   │
│  • Full: "2025-11-16T14:00:00"                                              │
│                                                                              │
│                               │                                              │
│                               ▼                                              │
│                                                                              │
│  OUTPUT: Atomic Entries {m_k}                                               │
│  ────────────────────────────                                                │
│  [                                                                           │
│    {                                                                         │
│      "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00       │
│        to meet with Bob at Starbucks on 2025-11-16T14:00:00...",           │
│      "keywords": ["Alice", "Bob", "Starbucks", "meeting"],                 │
│      "timestamp": "2025-11-16T14:00:00",                                   │
│      "location": "Starbucks",                                               │
│      "persons": ["Alice", "Bob"],                                          │
│      "entities": ["new product"],                                          │
│      "topic": "Meeting arrangement"                                         │
│    },                                                                        │
│    {                                                                         │
│      "lossless_restatement": "Bob agreed to attend the meeting and         │
│        committed to prepare relevant materials.",                           │
│      ...                                                                     │
│    }                                                                         │
│  ]                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Parallel Processing Flows

### Parallel Memory Building

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Parallel Memory Building Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  add_dialogues_parallel(dialogues)                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ dialogue_buffer.extend(dialogues)                                      │  │
│  │ Total: N dialogues                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Split into Windows                                                     │  │
│  │                                                                        │  │
│  │ windows_to_process = []                                               │  │
│  │ while len(buffer) >= window_size:                                     │  │
│  │     window = buffer[:window_size]                                     │  │
│  │     buffer = buffer[window_size:]                                     │  │
│  │     windows_to_process.append(window)                                 │  │
│  │                                                                        │  │
│  │ if buffer:  # Remaining dialogues                                     │  │
│  │     windows_to_process.append(buffer)                                 │  │
│  │     buffer = []                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       │  Example: 35 dialogues, window_size=10                              │
│       │  → [10, 10, 10, 5] = 4 windows                                      │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ThreadPoolExecutor(max_workers=max_parallel_workers)                   │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Task Submission                             │  │  │
│  │  │                                                                  │  │  │
│  │  │  for i, window in enumerate(windows):                           │  │  │
│  │  │      future = executor.submit(                                  │  │  │
│  │  │          _generate_memory_entries_worker,                       │  │  │
│  │  │          window,                                                 │  │  │
│  │  │          dialogue_ids,                                           │  │  │
│  │  │          window_num=i+1                                          │  │  │
│  │  │      )                                                           │  │  │
│  │  │      future_to_window[future] = (window, i+1)                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                           │                                            │  │
│  │                           ▼                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Parallel Execution                             │  │  │
│  │  │                                                                  │  │  │
│  │  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                   │  │  │
│  │  │   │ Worker 1 │   │ Worker 2 │   │ Worker 3 │                   │  │  │
│  │  │   │ Window 1 │   │ Window 2 │   │ Window 3 │                   │  │  │
│  │  │   │ (10 dlgs)│   │ (10 dlgs)│   │ (10 dlgs)│                   │  │  │
│  │  │   └────┬─────┘   └────┬─────┘   └────┬─────┘                   │  │  │
│  │  │        │              │              │                          │  │  │
│  │  │        │    LLM API Calls (Concurrent)                          │  │  │
│  │  │        │              │              │                          │  │  │
│  │  │        ▼              ▼              ▼                          │  │  │
│  │  │   [entries1]     [entries2]     [entries3]                     │  │  │
│  │  │        │              │              │                          │  │  │
│  │  │        └──────────────┴──────────────┘                          │  │  │
│  │  │                       │                                          │  │  │
│  │  │   ┌───────────────────┴───────────────────┐                    │  │  │
│  │  │   │ Window 4 (remaining 5 dialogues)       │                    │  │  │
│  │  │   │ Processed when worker becomes free     │                    │  │  │
│  │  │   └───────────────────────────────────────┘                    │  │  │
│  │  │                       │                                          │  │  │
│  │  │                       ▼                                          │  │  │
│  │  │               [entries4]                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                           │                                            │  │
│  │                           ▼                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Result Collection                              │  │  │
│  │  │                                                                  │  │  │
│  │  │  all_entries = []                                               │  │  │
│  │  │  for future in as_completed(future_to_window):                  │  │  │
│  │  │      entries = future.result()                                  │  │  │
│  │  │      all_entries.extend(entries)                                │  │  │
│  │  │                                                                  │  │  │
│  │  │  Total: entries1 + entries2 + entries3 + entries4              │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Batch Storage                                                          │  │
│  │                                                                        │  │
│  │ vector_store.add_entries(all_entries)                                 │  │
│  │ • Generate embeddings for all entries at once                         │  │
│  │ • Single batch insert to LanceDB                                      │  │
│  │                                                                        │  │
│  │ previous_entries = all_entries[-10:]  # Keep for context              │  │
│  │ processed_count += total_dialogues                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parallel Query Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Parallel Query Execution Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  _execute_parallel_searches(search_queries)                                 │
│       │                                                                      │
│       │  search_queries = ["query1", "query2", "query3"]                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ThreadPoolExecutor(max_workers=max_retrieval_workers)                  │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Task Submission                             │  │  │
│  │  │                                                                  │  │  │
│  │  │  for i, query in enumerate(search_queries, 1):                  │  │  │
│  │  │      future = executor.submit(                                  │  │  │
│  │  │          _semantic_search_worker,                               │  │  │
│  │  │          query,                                                  │  │  │
│  │  │          query_num=i                                             │  │  │
│  │  │      )                                                           │  │  │
│  │  │      future_to_query[future] = (query, i)                       │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                           │                                            │  │
│  │                           ▼                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Parallel Vector Searches                       │  │  │
│  │  │                                                                  │  │  │
│  │  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │  │  │
│  │  │   │   Search 1   │   │   Search 2   │   │   Search 3   │       │  │  │
│  │  │   │   "query1"   │   │   "query2"   │   │   "query3"   │       │  │  │
│  │  │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │  │  │
│  │  │          │                  │                  │                │  │  │
│  │  │          ▼                  ▼                  ▼                │  │  │
│  │  │   ┌──────────────────────────────────────────────────────┐     │  │  │
│  │  │   │              For each search:                         │     │  │  │
│  │  │   │                                                       │     │  │  │
│  │  │   │  1. query_vector = embedding_model.encode_single(    │     │  │  │
│  │  │   │         query, is_query=True)                        │     │  │  │
│  │  │   │                                                       │     │  │  │
│  │  │   │  2. results = table.search(query_vector)             │     │  │  │
│  │  │   │         .limit(semantic_top_k).to_list()             │     │  │  │
│  │  │   │                                                       │     │  │  │
│  │  │   │  3. Convert results to List[MemoryEntry]             │     │  │  │
│  │  │   └──────────────────────────────────────────────────────┘     │  │  │
│  │  │          │                  │                  │                │  │  │
│  │  │          ▼                  ▼                  ▼                │  │  │
│  │  │   [results1]           [results2]          [results3]          │  │  │
│  │  │   (5 entries)          (5 entries)         (5 entries)         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                           │                                            │  │
│  │                           ▼                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Result Collection                              │  │  │
│  │  │                                                                  │  │  │
│  │  │  all_results = []                                               │  │  │
│  │  │  for future in as_completed(future_to_query):                   │  │  │
│  │  │      results = future.result()                                  │  │  │
│  │  │      all_results.extend(results)                                │  │  │
│  │  │                                                                  │  │  │
│  │  │  Total: up to 15 entries (potentially with duplicates)         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Deduplication                                                          │  │
│  │                                                                        │  │
│  │ _merge_and_deduplicate_entries(all_results)                           │  │
│  │                                                                        │  │
│  │ seen_ids = set()                                                      │  │
│  │ merged = []                                                           │  │
│  │ for entry in all_results:                                             │  │
│  │     if entry.entry_id not in seen_ids:                                │  │
│  │         seen_ids.add(entry.entry_id)                                  │  │
│  │         merged.append(entry)                                          │  │
│  │                                                                        │  │
│  │ Result: Unique entries (e.g., 8 unique from 15 total)                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  merged_results: List[MemoryEntry]                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Interaction Flows

### Chat Completion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LLM Chat Completion Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  llm_client.chat_completion(messages, temperature, response_format)         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Build API Request                                                      │  │
│  │                                                                        │  │
│  │ kwargs = {                                                            │  │
│  │     "model": self.model,                                              │  │
│  │     "messages": messages,                                              │  │
│  │     "temperature": temperature                                         │  │
│  │ }                                                                      │  │
│  │                                                                        │  │
│  │ if response_format:  # JSON mode                                      │  │
│  │     kwargs["response_format"] = {"type": "json_object"}               │  │
│  │                                                                        │  │
│  │ if is_qwen_api:                                                       │  │
│  │     if streaming and enable_thinking:                                 │  │
│  │         kwargs["extra_body"] = {"enable_thinking": True}              │  │
│  │     else:                                                              │  │
│  │         kwargs["extra_body"] = {"enable_thinking": False}             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Retry Loop (max_retries=3)                                             │  │
│  │                                                                        │  │
│  │ for attempt in range(max_retries):                                    │  │
│  │     try:                                                               │  │
│  │         │                                                              │  │
│  │         ▼                                                              │  │
│  │     ┌─────────────────────────────────────────────────────────────┐   │  │
│  │     │ if use_streaming:                                            │   │  │
│  │     │     return _handle_streaming_response(**kwargs)             │   │  │
│  │     │ else:                                                        │   │  │
│  │     │     response = client.chat.completions.create(**kwargs)     │   │  │
│  │     │     return response.choices[0].message.content              │   │  │
│  │     └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                        │  │
│  │     except Exception as e:                                            │  │
│  │         if attempt < max_retries - 1:                                 │  │
│  │             sleep(2 ** attempt)  # Exponential backoff                │  │
│  │         else:                                                          │  │
│  │             raise                                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  response: str (LLM response content)                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### JSON Extraction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JSON Extraction Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  llm_client.extract_json(text)                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Remove common LLM prefixes                                     │  │
│  │                                                                        │  │
│  │ Prefixes removed:                                                      │  │
│  │ - "Here's the JSON:"                                                  │  │
│  │ - "Here is the response:"                                             │  │
│  │ - "The result is:"                                                    │  │
│  │ - etc.                                                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: Try direct JSON parsing                                        │  │
│  │                                                                        │  │
│  │ try:                                                                   │  │
│  │     return json.loads(text.strip())                                   │  │
│  │ except json.JSONDecodeError:                                          │  │
│  │     pass  # Continue to other methods                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: Try extracting from ```json block                             │  │
│  │                                                                        │  │
│  │ pattern: ```json\n(.*?)\n```                                          │  │
│  │                                                                        │  │
│  │ if match:                                                              │  │
│  │     json_str = _clean_json_string(match.group(1))                     │  │
│  │     return json.loads(json_str)                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 4: Try extracting from generic ``` block                         │  │
│  │                                                                        │  │
│  │ pattern: ```\n?(.*?)\n?```                                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 5: Try finding balanced JSON structure                            │  │
│  │                                                                        │  │
│  │ _extract_balanced_json(text, '{')  # For objects                      │  │
│  │ _extract_balanced_json(text, '[')  # For arrays                       │  │
│  │                                                                        │  │
│  │ Algorithm:                                                             │  │
│  │ - Find first '{' or '['                                               │  │
│  │ - Track depth (increment on '{[', decrement on '}]')                  │  │
│  │ - Handle strings (ignore brackets inside quotes)                       │  │
│  │ - Return when depth returns to 0                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 6: Clean and retry                                                │  │
│  │                                                                        │  │
│  │ _clean_json_string(json_str):                                         │  │
│  │ - Remove trailing commas before } or ]                                │  │
│  │ - Remove // comments                                                  │  │
│  │ - Remove /* */ comments                                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  result: Dict | List (parsed JSON)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Storage Flows

### Embedding Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Embedding Generation Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DOCUMENT ENCODING (Write Path)                                             │
│  ──────────────────────────────                                              │
│                                                                              │
│  embedding_model.encode_documents(texts)                                    │
│       │                                                                      │
│       ▼                                                                      │
│  encode(texts, is_query=False)                                              │
│       │                                                                      │
│       ├── Qwen3 Model ──► _encode_standard(texts)                           │
│       │                   • No query prompt                                 │
│       │                   • normalize_embeddings=True                       │
│       │                                                                      │
│       └── Standard Model ──► _encode_standard(texts)                        │
│                              • model.encode(texts)                          │
│       │                                                                      │
│       ▼                                                                      │
│  np.ndarray (N x dimension)                                                 │
│                                                                              │
│                                                                              │
│  QUERY ENCODING (Read Path)                                                 │
│  ──────────────────────────                                                  │
│                                                                              │
│  embedding_model.encode_single(query, is_query=True)                        │
│       │                                                                      │
│       ▼                                                                      │
│  encode([query], is_query=True)                                             │
│       │                                                                      │
│       ├── Qwen3 Model + supports_query_prompt                               │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   _encode_with_query_prompt(texts)                                  │
│       │   • model.encode(texts, prompt_name="query")                        │
│       │   • Uses Qwen3's asymmetric retrieval optimization                  │
│       │                                                                      │
│       └── Standard Model ──► _encode_standard(texts)                        │
│       │                                                                      │
│       ▼                                                                      │
│  np.ndarray (dimension,) [0]  # Return first (single) embedding            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LanceDB Storage Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LanceDB Storage Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ADD ENTRIES                                                                │
│  ───────────                                                                 │
│                                                                              │
│  vector_store.add_entries(entries)                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Generate Embeddings                                                    │  │
│  │                                                                        │  │
│  │ restatements = [e.lossless_restatement for e in entries]              │  │
│  │ vectors = embedding_model.encode_documents(restatements)              │  │
│  │                                                                        │  │
│  │ Result: np.ndarray (N x dimension)                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Build Data Records                                                     │  │
│  │                                                                        │  │
│  │ data = []                                                             │  │
│  │ for entry, vector in zip(entries, vectors):                           │  │
│  │     data.append({                                                     │  │
│  │         "entry_id": entry.entry_id,                                   │  │
│  │         "lossless_restatement": entry.lossless_restatement,           │  │
│  │         "keywords": entry.keywords,                                   │  │
│  │         "timestamp": entry.timestamp or "",                           │  │
│  │         "location": entry.location or "",                             │  │
│  │         "persons": entry.persons,                                     │  │
│  │         "entities": entry.entities,                                   │  │
│  │         "topic": entry.topic or "",                                   │  │
│  │         "vector": vector.tolist()                                     │  │
│  │     })                                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Insert to LanceDB                                                      │  │
│  │                                                                        │  │
│  │ table.add(data)                                                       │  │
│  │                                                                        │  │
│  │ • Batch insert                                                        │  │
│  │ • Automatic indexing                                                  │  │
│  │ • Transaction handling                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                                                              │
│  VECTOR SEARCH                                                              │
│  ─────────────                                                               │
│                                                                              │
│  vector_store.semantic_search(query, top_k)                                 │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Generate Query Vector                                                  │  │
│  │                                                                        │  │
│  │ query_vector = embedding_model.encode_single(query, is_query=True)    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Execute Vector Search                                                  │  │
│  │                                                                        │  │
│  │ results = table.search(query_vector.tolist())                         │  │
│  │              .limit(top_k)                                             │  │
│  │              .to_list()                                                │  │
│  │                                                                        │  │
│  │ • Cosine similarity search                                            │  │
│  │ • Returns top_k most similar                                          │  │
│  │ • Includes _distance field                                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Convert to MemoryEntry                                                 │  │
│  │                                                                        │  │
│  │ entries = []                                                          │  │
│  │ for result in results:                                                │  │
│  │     entry = MemoryEntry(                                              │  │
│  │         entry_id=result["entry_id"],                                  │  │
│  │         lossless_restatement=result["lossless_restatement"],          │  │
│  │         keywords=list(result["keywords"]) if ... else [],             │  │
│  │         timestamp=result["timestamp"] if ... else None,               │  │
│  │         location=result["location"] if ... else None,                 │  │
│  │         persons=list(result["persons"]) if ... else [],               │  │
│  │         entities=list(result["entities"]) if ... else [],             │  │
│  │         topic=result["topic"] if ... else None                        │  │
│  │     )                                                                 │  │
│  │     entries.append(entry)                                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│  List[MemoryEntry]                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Flows

### Retry Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Retry Mechanism Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXTRACTION RETRY (Memory Builder)                                          │
│  ─────────────────────────────────                                           │
│                                                                              │
│  _generate_memory_entries(dialogues)                                        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ max_retries = 3                                                        │  │
│  │                                                                        │  │
│  │ for attempt in range(max_retries):                                    │  │
│  │     try:                                                               │  │
│  │         response = llm_client.chat_completion(...)                    │  │
│  │         entries = _parse_llm_response(response, ...)                  │  │
│  │         return entries  ──────────────────────────► Success           │  │
│  │                                                                        │  │
│  │     except Exception as e:                                            │  │
│  │         if attempt < max_retries - 1:                                 │  │
│  │             print(f"Attempt {attempt + 1} failed: {e}")               │  │
│  │             print("Retrying...")                                      │  │
│  │             continue  ─────────────────────────────► Retry            │  │
│  │         else:                                                          │  │
│  │             print(f"All {max_retries} attempts failed: {e}")          │  │
│  │             return []  ────────────────────────────► Return empty     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                                                              │
│  API RETRY (LLM Client)                                                     │
│  ──────────────────────                                                      │
│                                                                              │
│  llm_client.chat_completion(messages, max_retries=3)                        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ for attempt in range(max_retries):                                    │  │
│  │     try:                                                               │  │
│  │         response = client.chat.completions.create(...)                │  │
│  │         return response.choices[0].message.content ──► Success        │  │
│  │                                                                        │  │
│  │     except Exception as e:                                            │  │
│  │         if attempt < max_retries - 1:                                 │  │
│  │             wait_time = 2 ** attempt  # 1, 2, 4 seconds               │  │
│  │             print(f"Attempt {attempt + 1} failed, waiting {wait_time}s")  │
│  │             time.sleep(wait_time)                                     │  │
│  │             continue  ─────────────────────────────► Retry            │  │
│  │         else:                                                          │  │
│  │             raise  ────────────────────────────────► Propagate error  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                                                              │
│  PARALLEL FALLBACK (Memory Builder)                                         │
│  ──────────────────────────────────                                          │
│                                                                              │
│  add_dialogues_parallel(dialogues)                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ try:                                                                   │  │
│  │     # Parallel processing                                             │  │
│  │     _process_windows_parallel(windows)                                │  │
│  │                                                                        │  │
│  │ except Exception as e:                                                │  │
│  │     print(f"Parallel processing failed: {e}")                         │  │
│  │     print("Falling back to sequential processing...")                 │  │
│  │                                                                        │  │
│  │     # Sequential fallback                                             │  │
│  │     for window in windows_to_process:                                 │  │
│  │         dialogue_buffer = window + dialogue_buffer                    │  │
│  │         process_window()                                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Answer Generation Fallback

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Answer Generation Fallback Flow                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  answer_generator.generate_answer(query, contexts)                          │
│       │                                                                      │
│       ├── contexts empty? ──► return "No relevant information found"        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ max_retries = 3                                                        │  │
│  │                                                                        │  │
│  │ for attempt in range(max_retries):                                    │  │
│  │     try:                                                               │  │
│  │         response = llm_client.chat_completion(messages, ...)          │  │
│  │         result = llm_client.extract_json(response)                    │  │
│  │         return result.get("answer", response.strip())                 │  │
│  │                        │                                               │  │
│  │                        ├── JSON has "answer" ──► return answer        │  │
│  │                        │                                               │  │
│  │                        └── No "answer" key ──► return raw response    │  │
│  │                                                                        │  │
│  │     except Exception as e:                                            │  │
│  │         if attempt < max_retries - 1:                                 │  │
│  │             print(f"Attempt {attempt + 1} failed: {e}. Retrying...")  │  │
│  │             continue                                                   │  │
│  │         else:                                                          │  │
│  │             print(f"Failed after {max_retries} attempts: {e}")        │  │
│  │                                                                        │  │
│  │             # Final fallback                                          │  │
│  │             if 'response' in locals():                                │  │
│  │                 return response.strip()  ──► Return raw LLM response  │  │
│  │             else:                                                      │  │
│  │                 return "Failed to generate answer" ──► Error message  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data State Transitions

### Memory Builder State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Memory Builder State Transitions                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Initial State                                                              │
│  ─────────────                                                               │
│  dialogue_buffer = []                                                       │
│  processed_count = 0                                                        │
│  previous_entries = []                                                      │
│                                                                              │
│                                                                              │
│  After add_dialogue() / add_dialogues()                                     │
│  ───────────────────────────────────────                                     │
│  dialogue_buffer = [d1, d2, d3, ...]  # Accumulated dialogues              │
│  processed_count = 0                   # Not yet processed                  │
│  previous_entries = []                 # No previous context               │
│                                                                              │
│                                                                              │
│  After process_window() (first window)                                      │
│  ─────────────────────────────────────                                       │
│  dialogue_buffer = [d11, d12, ...]     # Remaining after window_size       │
│  processed_count = window_size         # e.g., 10                          │
│  previous_entries = [e1, e2, e3, ...]  # Entries from first window         │
│                                                                              │
│                                                                              │
│  After process_window() (subsequent windows)                                │
│  ───────────────────────────────────────────                                 │
│  dialogue_buffer = [d21, d22, ...]     # Remaining                         │
│  processed_count = N * window_size     # Total dialogues processed         │
│  previous_entries = [eN1, eN2, ...]    # Entries from last window          │
│                                                                              │
│                                                                              │
│  After finalize() / process_remaining()                                     │
│  ──────────────────────────────────────                                      │
│  dialogue_buffer = []                  # Cleared                           │
│  processed_count = total_dialogues     # All dialogues processed           │
│  previous_entries = [...]              # Last entries for context          │
│                                                                              │
│                                                                              │
│  Parallel Processing State Changes                                          │
│  ─────────────────────────────────                                           │
│                                                                              │
│  add_dialogues_parallel(dialogues):                                         │
│                                                                              │
│  1. dialogue_buffer.extend(dialogues)  # All added to buffer               │
│  2. Split into windows                 # buffer → windows_to_process       │
│  3. dialogue_buffer = []               # Buffer cleared                    │
│  4. Process all windows in parallel                                        │
│  5. processed_count += total           # All processed                     │
│  6. previous_entries = last_10         # Context for future                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Retrieval State (per query)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Retrieval State Transitions                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: query = "When will Alice and Bob meet?"                             │
│                                                                              │
│                                                                              │
│  Stage 1: Information Analysis                                              │
│  ─────────────────────────────                                               │
│                                                                              │
│  information_plan = {                                                       │
│      "question_type": "temporal",                                           │
│      "key_entities": ["Alice", "Bob"],                                      │
│      "required_info": [...],                                                │
│      "minimal_queries_needed": 2                                            │
│  }                                                                           │
│                                                                              │
│                                                                              │
│  Stage 2: Query Generation                                                  │
│  ─────────────────────────                                                   │
│                                                                              │
│  search_queries = [                                                         │
│      "When will Alice and Bob meet?",                                       │
│      "Alice Bob meeting time",                                              │
│      "meeting schedule Alice Bob"                                           │
│  ]                                                                           │
│                                                                              │
│                                                                              │
│  Stage 3: Parallel Search                                                   │
│  ────────────────────────                                                    │
│                                                                              │
│  all_results = [                                                            │
│      # Results from query 1                                                 │
│      MemoryEntry(...), MemoryEntry(...), ...                                │
│      # Results from query 2                                                 │
│      MemoryEntry(...), MemoryEntry(...), ...                                │
│      # Results from query 3                                                 │
│      MemoryEntry(...), MemoryEntry(...), ...                                │
│  ]  # May have duplicates                                                   │
│                                                                              │
│                                                                              │
│  Stage 4: Deduplication                                                     │
│  ──────────────────────                                                      │
│                                                                              │
│  merged_results = [                                                         │
│      MemoryEntry(entry_id="abc..."),                                        │
│      MemoryEntry(entry_id="def..."),                                        │
│      MemoryEntry(entry_id="ghi..."),                                        │
│      ...                                                                     │
│  ]  # Unique entries only                                                   │
│                                                                              │
│                                                                              │
│  Stage 5: Reflection (if enabled)                                           │
│  ────────────────────────────────                                            │
│                                                                              │
│  Round 1:                                                                   │
│    completeness = "incomplete"                                              │
│    additional_queries = ["Alice Bob Starbucks meeting"]                     │
│    additional_results = [MemoryEntry(...), ...]                             │
│    merged_results = deduplicate(merged_results + additional_results)        │
│                                                                              │
│  Round 2:                                                                   │
│    completeness = "complete"                                                │
│    break                                                                     │
│                                                                              │
│                                                                              │
│  Final State: contexts = merged_results                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This document has detailed the complete data flow through the SimpleMem system:

1. **Write Path**: Dialogue → Buffer → LLM Extraction → MemoryEntry → Embedding → LanceDB
2. **Read Path**: Query → Planning → Parallel Search → Merge → Reflection → Answer
3. **Transformations**: 14 distinct transformations from input to output
4. **Parallel Processing**: ThreadPoolExecutor for both memory building and retrieval
5. **Error Handling**: Retry mechanisms with exponential backoff and fallbacks
6. **State Management**: Buffer management, context preservation, and deduplication

The data flow is designed to maximize efficiency through:
- Batch processing and parallel execution
- Coreference resolution at write-time (not read-time)
- Multi-query planning for comprehensive retrieval
- Reflection loops for completeness verification
