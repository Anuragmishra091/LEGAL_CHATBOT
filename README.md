# Legal Chatbot — Indian Law RAG System

An intelligent legal research assistant built on Retrieval-Augmented Generation (RAG) for Indian law. Covers the Constitution of India, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), Bharatiya Sakshya Adhiniyam (BSA), and 48,000+ Supreme Court judgments from the JUDIS dataset.

---

## Features

- **Multi-source retrieval** — semantic search across statutes and Supreme Court judgments
- **Knowledge graph for Constitution** — hard edges (cross-references) + soft edges (semantic similarity) for enriched article retrieval
- **Query routing** — automatically classifies queries as Constitution / Judgment / Statute and routes to the appropriate retriever
- **NVIDIA LLM backend** — powered by Llama 3.3-70B via NVIDIA's OpenAI-compatible API
- **Flask web UI** — chat interface with source attribution and relevance scores
- **Evaluation suite** — 7 RAG quality metrics (faithfulness, relevancy, citation accuracy, semantic correctness, and more)

---

## Architecture

```
User Query
    │
    ▼
Query Router (main.py)
    │
    ├─── Constitution ──► Graph Retriever (ChromaDB + SQLite graph)
    ├─── Judgment ──────► Statute Retriever (ChromaDB)
    └─── Statute ───────► Statute Retriever (ChromaDB)
                                │
                                ▼
                        NVIDIA Llama 3.3-70B
                                │
                                ▼
                        Answer + Sources
```

---

## Project Structure

```
LEGAL_CHATBOT/
├── scripts/
│   ├── app.py                    # Flask web server & API endpoints
│   ├── main.py                   # LegalChatbot class & query routing
│   ├── Retrieval_function.py     # Base StatuteRetriever (ChromaDB)
│   ├── graph_retriever.py        # ConstitutionGraphRetriever (ChromaDB + graph)
│   ├── constitution_graph.py     # Knowledge graph builder for Constitution
│   ├── ingest_constitution.py    # Constitution PDF ingestion pipeline
│   ├── ingest_judgments.py       # JUDIS judgment ingestion pipeline
│   ├── bns_layout_agent.py       # BNS section-level chunking agent
│   ├── eval_metrics.py           # RAG evaluation metrics
│   ├── eval_runner.py            # Evaluation orchestrator
│   ├── eval_testset.json         # 50+ test questions with ground truth
│   ├── eval_results.json         # Latest evaluation results
│   ├── constitution_manifest.json
│   ├── judgments_manifest.json
│   └── templates/
│       └── index.html            # Web chat UI
├── data/
│   ├── BNS.pdf
│   ├── BNSS.pdf
│   ├── BSA.pdf
│   └── Constitution of India.pdf
├── vectorstore/                  # ChromaDB persistent storage (gitignored)
├── .env                          # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anuragmishra091/LEGAL_CHATBOT.git
cd LEGAL_CHATBOT
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Get your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com).

---

## Data Ingestion

Run ingestion scripts once to populate the vector database before starting the app.

### Ingest statutes (BNS, BNSS, BSA, Constitution)

```bash
python scripts/ingest_constitution.py
python scripts/bns_layout_agent.py
```

### Ingest Supreme Court judgments

```bash
# Ingest all (~48k judgments — takes time)
python scripts/ingest_judgments.py

# Test with a small batch first
python scripts/ingest_judgments.py --limit 100
```

### Build the Constitution knowledge graph

```bash
python scripts/constitution_graph.py
```

---

## Running the App

```bash
python scripts/app.py
```

Open your browser at `http://localhost:5000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web chat UI |
| `POST` | `/chat` | Send a query, returns answer + sources |
| `POST` | `/eval/run` | Start an evaluation run |
| `GET` | `/eval/status` | Poll evaluation progress |
| `GET` | `/eval/results` | Fetch latest evaluation results |

**Example chat request:**
```json
POST /chat
{
  "query": "What is the procedure for arrest under BNSS?",
  "history": []
}
```

---

## Evaluation

The system includes a built-in evaluation suite with 7 metrics:

| Metric | Type | Description |
|--------|------|-------------|
| Faithfulness | Level 1 | Are claims in the answer supported by retrieved context? |
| Answer Relevancy | Level 1 | Is the answer relevant to the question? |
| Context Relevancy | Level 1 | Are retrieved chunks relevant to the query? |
| Citation Accuracy | Level 1 | Are citations verifiable? |
| Keyword Recall | Level 2 | % of expected keywords present in answer |
| Source Accuracy | Level 2 | Did the retriever find the expected documents? |
| Semantic Correctness | Level 2 | Embedding similarity to expected answer |

Run evaluation:
```bash
python scripts/eval_runner.py

# Run on a subset
python scripts/eval_runner.py --limit 10

# Filter by category
python scripts/eval_runner.py --category Constitution
```

---

## Legal Data Sources

| Source | Description |
|--------|-------------|
| Constitution of India | All articles with sub-clause chunking |
| BNS 2023 | Bharatiya Nyaya Sanhita — section-level chunks |
| BNSS 2023 | Bharatiya Nagarik Suraksha Sanhita |
| BSA 2023 | Bharatiya Sakshya Adhiniyam (Evidence Act) |
| JUDIS | 48,000+ Supreme Court judgments (Kaggle dataset) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | NVIDIA Llama 3.3-70B (OpenAI-compatible API) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Knowledge Graph | SQLite |
| PDF Extraction | PyMuPDF (fitz) |
| Web Framework | Flask |
| Frontend | HTML/CSS/JS (Jinja2 template) |

---

## License

This project is intended for educational and research purposes only. Legal data sources are publicly available Indian government documents and the JUDIS public dataset.
