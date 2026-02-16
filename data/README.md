# Data Folder — Knowledge Base Documents

This folder contains the **knowledge base** for the Enterprise Multi-Agent Copilot (Project #6).
These documents serve as the ground-truth source of information that the system's retrieval layer
searches through when answering user queries about retail and CPG (Consumer Packaged Goods) operations.

## What Is in This Folder

Twelve PDF documents covering key topics in retail operations, supply chain, and omnichannel strategy:

| Document                                                                           | Topic                             |
| ---------------------------------------------------------------------------------- | --------------------------------- |
| Building-omnichannel-excellence.pdf                                                | Omnichannel strategy              |
| cpg_digital_transformation_ebook.pdf                                               | Digital transformation in CPG     |
| deloitte-global-powers-of-retailing-2025.pdf                                       | Global retail industry analysis   |
| GS1_SupplyChainVisibility_WhitePaper.pdf                                           | Supply chain visibility standards |
| GS1-US-Autonomous-Fulfillment-Whitepaper-2020.pdf                                  | Autonomous fulfillment            |
| Improving Inventory Accuracy Through Innovation.pdf                                | Inventory management              |
| mck_retail-ops-2020_fullissue-rgb-hyperlinks-011620.pdf                            | Retail operations                 |
| Omni-Channel-Strategies-and Considerations-for-CPG-Companies.pdf                   | CPG omnichannel strategies        |
| retail-strategy-2023-2027.pdf                                                      | Retail strategy planning          |
| Supply-chain-of-the-future.pdf                                                     | Future supply chain trends        |
| The State of Returns Report 2024 — Optoro.pdf                                      | Returns management                |
| the-winning-formula-what-it-takes-to-build-leading-omnichannel-operations-2022.pdf | Omnichannel operations            |

## Source Type

All documents are **publicly available** whitepapers and reports. They do not contain any
confidential, proprietary, or sensitive information.

## How the Retrieval Layer Uses These Documents

The copilot uses a **RAG (Retrieval-Augmented Generation)** pipeline to ground its answers in
real evidence. Here is how it works at a high level:

1. **Load** — Each PDF is read and its text is extracted page by page.
2. **Chunk** — The extracted text is split into smaller pieces (~1 000 characters each, with
   a 200-character overlap between consecutive chunks so context is not lost at boundaries).
3. **Embed** — Every chunk is converted into a numerical vector (embedding) that captures its
   meaning.
4. **Index** — The embeddings are stored in a ChromaDB vector database for fast similarity search.
5. **Retrieve** — When a user asks a question, the query is also embedded and compared against
   the stored chunks using cosine similarity. The most relevant chunks are returned.
6. **Cite** — Each returned chunk carries a citation in the format
   `DocumentName, Page X, Chunk Y`, so the final output can reference exactly where each
   piece of information came from.

## Assumptions and Limitations

- **English only** — All documents are in English; the system does not support other languages.
- **PDF extraction quality** — Some pages with complex tables, charts, or multi-column layouts
  may not extract perfectly, which can reduce retrieval accuracy for those sections.
- **Static knowledge base** — The documents are not updated automatically. To add or replace
  sources, place new PDFs in this folder and rebuild the vector index.
- **Domain scope** — The knowledge base is focused on retail and CPG topics. Queries outside
  this domain will likely return weak or no results.
