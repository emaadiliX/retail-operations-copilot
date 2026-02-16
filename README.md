# Retail Operations Copilot

A multi-agent AI system that turns retail and CPG business questions into structured, citation-grounded deliverables. Built with the OpenAI Agents SDK, Streamlit, and ChromaDB.

## Project Overview

This project implements an **Enterprise Multi-Agent Copilot** for the Retail / CPG industry (Project #6). A user submits a business question, and four coordinated AI agents work through a five-stage pipeline to produce a decision-ready deliverable:

**Plan → Research → Draft → Verify → Deliver**

Every claim in the final output is grounded in a knowledge base of 12 public retail and CPG documents. If evidence is missing, the system explicitly states **"Not found in sources"** instead of guessing.

## Key Features

- **Four specialized agents** — Planner, Researcher, Writer, and Verifier, each with a focused role
- **RAG retrieval with citations** — every finding references its source as `DocumentName, Page X, Chunk Y`
- **Hallucination prevention** — the Verifier cross-checks every claim against the original source text
- **Structured outputs** — Executive Summary (max 150 words), Client-ready Email, Action Items (with owner, due date, confidence), and Sources
- **Prompt injection defense** — input guard with pattern detection and relevance checking
- **Real-time trace log** — per-agent timing, inputs, outputs, and metadata visible in the UI
- **Evaluation harness** — 10 test scenarios and 4 edge-case tests (injection, out-of-scope, hallucination)

## Repository Structure

```
app/              Streamlit UI (main dashboard)
copilot_agents/   Agent definitions, orchestrator, tools, tracing
retrieval/        PDF ingestion, chunking, embedding, vector search
data/             12 retail/CPG knowledge-base PDFs (see data/README.md)
eval/             Evaluation harness with test prompts and grading
chroma_db/        Vector store (generated at runtime, not committed)
```

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd retail-operations-copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the vector index

This reads all PDFs from `data/`, generates embeddings, and stores them in ChromaDB:

```bash
python -m retrieval.indexing
```

You only need to run this once. Re-run it if you add or replace documents in the `data/` folder.

### 5. Launch the app

```bash
streamlit run app/main.py
```

Open your browser at **http://localhost:8501**.

## How to Use the UI

1. **Enter a question** — type a retail or CPG business question in the text box
2. **Click "Run Workflow"** — the pipeline runs through all five stages (typically 30–90 seconds)
3. **Watch the sidebar** — it shows real-time progress as each agent completes its stage
4. **View results** — output is organized in seven tabs:

| Tab                  | What it shows                                                        |
| -------------------- | -------------------------------------------------------------------- |
| Executive Summary    | Short summary for leadership (max 150 words)                         |
| Client Email         | Professional, ready-to-send email                                    |
| Action Items         | Table with action, owner, due date, and confidence level             |
| Research & Sources   | Individual findings with citations, plus any information gaps        |
| Execution Plan       | The planner's breakdown of sub-tasks and research queries            |
| Verification Details | Per-claim verdicts (Supported / Partially Supported / Not Supported) |
| Agent Trace Log      | Timing, inputs, outputs, and metadata for each agent                 |

5. **Check the verdict badge** — above the tabs, a badge shows the overall result:
   - **PASS** — all claims verified against sources
   - **PARTIAL** — some claims only partially supported
   - **FAIL** — unsupported claims detected (check Verification tab for details)

## Example Queries

The UI includes built-in example buttons. Here are three to try:

- _"What are the top supply chain visibility challenges for CPG companies, and what technologies are being adopted to address them?"_
- _"Summarize the state of retail returns in 2024 and recommend strategies to reduce return rates while improving customer satisfaction."_
- _"What digital transformation strategies should a CPG company prioritize to stay competitive over the next 3-5 years?"_

## Troubleshooting

| Problem                      | What to do                                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Pipeline times out or hangs  | Re-run the query. LLM API calls can occasionally be slow.                                                        |
| No results or empty findings | Make sure you built the vector index first (`python -m retrieval.indexing`).                                     |
| "Input rejected" message     | The system only accepts retail/CPG business questions. Rephrase your query.                                      |
| FAIL verdict                 | The Verifier found unsupported claims. Open the **Verification Details** tab to see which claims failed and why. |
| ChromaDB errors on startup   | Delete the `chroma_db/` folder and rebuild with `python -m retrieval.indexing`.                                  |

## Evaluation

The `eval/` folder contains an evaluation harness with 10 graded test scenarios and 4 edge-case tests.

**Run all tests:**

```bash
python eval/run_eval.py
```

**Run a single test:**

```bash
python eval/run_eval.py --test inventory_accuracy
```

**Run with full output:**

```bash
python eval/run_eval.py --verbose
```

**Edge-case tests** cover prompt injection attempts, out-of-scope queries (non-retail topics), and hallucination traps (topics not present in the knowledge base).

## Tech Stack

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| Agent framework | OpenAI Agents SDK                        |
| LLM             | GPT-4o-mini (agents) / GPT-4o (verifier) |
| Embeddings      | text-embedding-3-small (1536-dim)        |
| Vector store    | ChromaDB (cosine similarity)             |
| UI              | Streamlit                                |
| PDF parsing     | pypdf                                    |
