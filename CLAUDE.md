# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleMem is a research implementation for efficient long-term memory for LLM agents. It implements a three-stage pipeline based on **Semantic Lossless Compression**:

1. **Semantic Structured Compression** (Stage 1): Converts dialogues into atomic entries via de-linearization, coreference resolution, and temporal anchoring
2. **Structured Indexing** (Stage 2): Indexes memories across semantic (dense vectors), lexical (BM25 keywords), and symbolic (metadata) layers
3. **Adaptive Query-Aware Retrieval** (Stage 3): Retrieves with complexity-aware pruning and planning-based multi-query decomposition

## Commands

### Setup
```bash
pip install -r requirements.txt
cp config.py.example config.py
# Edit config.py with your API key
```

### Run Benchmark
```bash
# Full LoCoMo-10 benchmark
python test_locomo10.py

# Subset evaluation
python test_locomo10.py --num-samples 5

# With LLM-as-judge evaluation
python test_locomo10.py --llm-judge

# Parallel question processing
python test_locomo10.py --parallel-questions

# Custom output file
python test_locomo10.py --result-file my_results.json
```

### Quick Test
```bash
python main.py
```

## Architecture

### Core Pipeline (`main.py` - `SimpleMemSystem`)
The main class orchestrates the pipeline:
- `add_dialogue()` / `add_dialogues()` → MemoryBuilder → VectorStore (Stage 1)
- `finalize()` → processes remaining dialogue buffer
- `ask()` → HybridRetriever → AnswerGenerator (Stage 3)

### Key Components

**`core/memory_builder.py` - MemoryBuilder**
- Implements de-linearization transformation F_θ
- Batches dialogues by `window_size` before LLM extraction
- Supports parallel processing via ThreadPoolExecutor
- Generates `MemoryEntry` atomic facts with disambiguated references

**`core/hybrid_retriever.py` - HybridRetriever**
- Planning-based retrieval: analyzes query complexity, generates targeted sub-queries
- Reflection loops: checks answer adequacy, generates additional queries if needed
- Parallel search execution across queries
- Merges and deduplicates results

**`core/answer_generator.py` - AnswerGenerator**
- Synthesizes answers from retrieved atomic contexts
- JSON-structured output with reasoning

**`database/vector_store.py` - VectorStore**
- LanceDB-backed storage
- Three search methods: `semantic_search()` (vector), `keyword_search()` (BM25-style), `structured_search()` (metadata filtering)

**`models/memory_entry.py`**
- `MemoryEntry`: Atomic entry with `lossless_restatement`, `keywords`, `timestamp`, `location`, `persons`, `entities`, `topic`
- `Dialogue`: Raw input format

**`utils/embedding.py` - EmbeddingModel**
- SentenceTransformers-based (supports Qwen3 embedding models)
- Query-vs-document encoding distinction for Qwen3

**`utils/llm_client.py` - LLMClient**
- OpenAI-compatible client with streaming support
- Robust JSON extraction from LLM outputs
- Retry mechanism with exponential backoff

### Configuration (`config.py`)
Key settings:
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`
- `EMBEDDING_MODEL` (local SentenceTransformers model)
- `WINDOW_SIZE` (dialogues per batch)
- `SEMANTIC_TOP_K`, `KEYWORD_TOP_K`, `STRUCTURED_TOP_K` (retrieval limits)
- `ENABLE_THINKING`, `USE_STREAMING`, `USE_JSON_FORMAT` (LLM features)

## Data Flow

```
Dialogues → MemoryBuilder.add_dialogues()
         → LLM extraction (de-linearization)
         → MemoryEntry[] with embeddings
         → VectorStore.add_entries()

Question → HybridRetriever.retrieve()
        → Planning: analyze requirements, generate queries
        → Parallel semantic search
        → Optional reflection loops
        → AnswerGenerator.generate_answer()
        → Answer string
```

## Testing

`test_locomo10.py` runs the LoCoMo-10 benchmark:
- Loads samples from `test_ref/locomo10.json`
- Category 5 questions (adversarial) use special binary-choice answer generation
- Metrics: F1, ROUGE, BLEU, BERTScore, METEOR, sentence similarity
- Optional LLM-as-judge evaluation via `--llm-judge`
