# HCL_HACKATHON_2026
Campus Placement Hackathon
# ResearchIQ

A multimodal research assistant for a collection of AI/ML research papers. Upload 8–10 PDFs on one topic and ask questions. Answers come from the papers' text, tables and figures, with page and figure citations. When the evidence isn't there, ResearchIQ says so instead of guessing.

Built for the HCL Campus Placement Hackathon 2026.

## Features

- **Multimodal ingestion:** parses text, structured tables and figures. Scanned pages go through OCR, and a vision-language model describes figures.
- **Hybrid retrieval:** dense vector search and BM25 keyword search, merged with Reciprocal Rank Fusion and then reranked with a cross-encoder.
- **Grounded answers:** every claim cites a retrieved source (paper, page, table or figure). AI suggestions are labelled separately from reported findings.
- **Evidence checks before answering:** if the retrieved evidence isn't relevant enough, the assistant abstains and explains what is missing.
- **Answer validation after generation:** checks that citations exist, that numbers appear in the cited text, that wording overlaps the evidence, and runs an LLM verifier. Failed claims get one retry, then are removed.
- **Paper comparison:** side by side on methodology, architecture, datasets, evaluation metrics, reported results and limitations.
- **Security guardrails:** upload validation, prompt-injection detection (Llama Prompt Guard 2 plus rule-based patterns), stripping of instruction-like text from retrieved evidence, secret redaction and rate limiting.

## How it works

### Pipeline A: ingestion (once per PDF)

1. **Validate:** checks the extension, size (≤ 50 MB), PDF signature, page count (≤ 80), encryption and active content (`/JavaScript`, `/Launch`). Duplicates are detected by SHA-256 hash.
2. **Parse:** Docling first, with PyMuPDF for verification and fallback. Pages with fewer than 200 characters of text go to PaddleOCR.
3. **Tables and figures:** tables are stored in structured form. Figures are described by Qwen2.5-VL as structured JSON, cached by image hash.
4. **Chunk:** child chunks of about 300 tokens (50-token overlap) are linked to parent chunks of about 1,200 tokens. Each table and figure is its own chunk.
5. **Index:** embeddings (`text-embedding-3-small`) go into Qdrant, metadata into SQLite, plus a BM25 index.

### Pipeline B: question answering

1. **Guard:** length and format checks, a rate limit (20 queries per minute) and injection detection. Questions that *discuss* prompt injection are allowed.
2. **Refine:** the LLM classifies the intent (factual, methodology, results, visual, comparison, limitations, synthesis) and rewrites the query. A rule-based fallback is used if the LLM is unavailable.
3. **Hybrid search:** dense top-20 and BM25 top-20 for each query variant, with extra table and figure passes when needed. Results are fused with RRF (k = 60).
4. **Rerank:** `BAAI/bge-reranker-base` keeps the best 6 (8 for comparison and synthesis questions).
5. **Expand evidence:** adds parent context, captions and figure assets, and strips instruction-like sentences.
6. **Sufficiency check:** relevance score, term coverage, requested figure or table labels, presence of numbers for results questions, and per-paper coverage for comparisons.
7. **Generate:** a structured answer with claims and source IDs, plus visual QA on the figure image when relevant.
8. **Validate:** the checks described under Features, then one retry, then a partial answer or abstention.

## Tech stack

| Area | Tools |
|---|---|
| UI | Streamlit |
| PDF parsing | Docling, PyMuPDF, PaddleOCR |
| Figure understanding | Qwen2.5-VL (hosted, OpenAI-compatible API) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | `gpt-4o-mini` by default (any OpenAI-compatible endpoint) |
| Vector store | Qdrant (embedded or server) |
| Keyword search | rank-bm25 |
| Reranker | `BAAI/bge-reranker-base` |
| Injection detection | `meta-llama/Llama-Prompt-Guard-2-22M` + rule-based patterns |
| Metadata | SQLite |
| Tests | pytest |

## Project structure

```
ResearchIQ/
├── app.py                # Streamlit entry point
├── config.py             # Settings loaded from .env
├── ingestion/            # Upload validation, parsing, tables, figures, chunking, embedding, indexing
├── retrieval/            # Query guard, refiner, vector + BM25 search, RRF fusion, reranker, evidence expansion
├── generation/           # LLM client, prompts, answer generation, visual QA, paper comparison
├── guardrails/           # Input, injection, evidence, citation and output validators
├── database/             # Qdrant and SQLite stores
├── ui/                   # Streamlit tabs
├── evaluation/           # Security test suite and saved results
├── tests/                # pytest suite
└── data/                 # Papers, chunks, figures, tables, DBs (git-ignored)
```

## Setup

Requires Python 3.11+ (3.12 recommended).

```bash
git clone https://github.com/tambarikajathin/HCL_HACKATHON_2026.git
cd HCL_HACKATHON_2026/ResearchIQ

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your keys
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings (also used for the LLM if `LLM_API_KEY` is blank) |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Optional | Any OpenAI-compatible LLM endpoint |
| `VLM_API_KEY`, `VLM_BASE_URL`, `VLM_MODEL` | For figures | Qwen2.5-VL endpoint (default: OpenRouter) |
| `HF_TOKEN` | For Prompt Guard | Prompt Guard 2 is a gated model; accept Meta's license first |
| `PROMPT_GUARD_ENABLED` | Optional | Set `false` until gated access is approved |
| `QDRANT_URL`, `QDRANT_API_KEY` | Optional | Leave empty to use embedded local Qdrant |

Limits, retrieval sizes and validation thresholds can also be changed in `.env`; see `.env.example`.

## Usage

**Run the app:**
```bash
streamlit run app.py
```
It has five tabs: **Dashboard**, **Paper Management**, **Research Chat**, **Paper Comparison** and **Evaluation**.

**Ingest papers from the command line:**
```bash
python -m ingestion.batch paper1.pdf paper2.pdf
python -m ingestion.batch --folder path/to/papers
```
A report is written to `data/cache/last_batch_report.json`.

## Testing

```bash
pytest tests/
```
There are 57 tests across ingestion, retrieval, generation, guardrails, the database and end-to-end flows.

**Security evaluation:**
```bash
python -m evaluation.security_tests                 # rule-based checks
python -m evaluation.security_tests --prompt-guard  # also Prompt Guard 2 (needs HF_TOKEN)
```
Results are saved to `evaluation/results/security_results.json`.

The latest saved run: **46/46 passed**.

| Category | Tests |
|---|---|
| Injection | 23 |
| Upload | 12 |
| Validation | 5 |
| Indirect (poisoned documents) | 5 |
| Leakage | 2 |

That run was made without Prompt Guard 2, so it covers the rule-based layer only.

## Limitations

- The security suite does not measure answer accuracy or retrieval quality.
- Figure descriptions come from a vision model and may be imperfect. They are marked as model-derived in the evidence.
- Embeddings, the LLM and figure understanding need external API keys.
