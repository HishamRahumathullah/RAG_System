# Production RAG System with Real Eval Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system with a local-first, offline-ready hybrid search pipeline and a comprehensive evaluation benchmark dashboard.

---

## 🏗️ Architecture

```
                       User Query
                           ↓
                   [Query Rewriting]  ← (Reformulates conversational/multi-turn queries)
                           ↓
                   [Hybrid Retriever] ← (Dense embeddings + Sparse BM25)
                           ↓
                      [Re-ranker]     ← (Cross-encoder scores top-k candidates)
                           ↓
                  [Context Assembler] ← (Deduplicates & maps citations [1], [2])
                           ↓
                    [LLM Generator]   ← (Generates grounded answer with citations)
                           ↓
                    [Eval Pipeline]   ← (Async scoring: Faithfulness, Relevance, Hallucination)
```

---

## ⚡ Key Features

| Feature | Naive RAG (Baseline) | Production RAG |
| :--- | :--- | :--- |
| **Search Method** | Dense Similarity Search | **Hybrid Search (Dense + Sparse BM25)** |
| **Fusion Logic** | None (Single search result) | **Reciprocal Rank Fusion (RRF)** |
| **Re-ranking** | None (Retrieval rank = Generation rank) | **Neural Cross-Encoder Re-ranking** |
| **Query Formatting** | Direct search on user query | **LLM/Heuristic Query Rewriter** |
| **Verification** | No built-in validation | **Evaluation Suite (Faithfulness, Relevance, Hallucination)** |
| **Citations** | Standard context injection | **Structured Inline Citations & Metadata Mappings** |

---

## 📁 Project Structure

```
rag-system/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py              # FastAPI backend
│   └── frontend.py          # Streamlit UI
├── src/
│   ├── ingestion.py         # Document chunking, BM25, vector upsert
│   ├── retrieval.py         # Hybrid dense+sparse retrieval with RRF
│   ├── reranking.py         # Cross-encoder reranker
│   ├── generation.py        # Context assembler & answer generator
│   └── evaluation.py        # Benchmark & metric computation
├── eval/
│   ├── metrics.py           # Retrieval metrics (Recall, F1, MRR)
│   └── judges.py            # LLM-as-a-judge & heuristic scorers
├── data/
│   └── raw_docs/            # Source markdown documents
└── notebooks/
    └── eval_analysis.ipynb  # Interactive evaluation analysis
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (Python 3.14 recommended)
- `uv` package manager (optional, but highly recommended for fast installs)

### 2. Installation
Clone this repository and install the dependencies:

```bash
# Using uv (extremely fast)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### 3. Running the System (Local Mode)
You can run the entire system 100% offline. Local embeddings are powered by `sentence-transformers/all-MiniLM-L6-v2`, reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, and generation utilizes a custom Jaccard term-matching synthesizer.

#### Start the FastAPI Backend:
```bash
uvicorn app.main:app --port 8000 --reload
```

#### Start the Streamlit Frontend:
In a separate terminal, launch the user interface:
```bash
streamlit run app/frontend.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Running Evaluations

You can run evaluations directly in three different ways:

1. **Streamlit Dashboard**: Navigate to the **Evaluation Dashboard** tab and click the **Run Benchmark Suite** button. This renders interactive seaborn charts and comparative statistics instantly.
2. **Command Line**: Run the standalone script to index the documents and evaluate the golden dataset:
   ```bash
   # Ingestion (runs automatically if database is empty)
   python src/ingestion.py

   # Run eval suite
   python src/evaluation.py
   ```
3. **Jupyter Notebook**: Open `notebooks/eval_analysis.ipynb` to view step-by-step calculations, breakdowns, and customized plots.

---

## 🛠️ Configuration & Customization

### Enabling OpenAI Mode (Optional)
If you have an OpenAI API Key and want to use GPT-4o for generation and GPT-4o-mini as an LLM-as-a-judge:

1. Set your environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
2. Enable the **"Use OpenAI"** checkbox in the Streamlit sidebar, or run scripts with the API mode enabled.
