import sys
import os

# Add project root to Python path so 'src' imports work in In-Process mode
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Production RAG Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.title-container {
    text-align: center;
    padding: 1.5rem;
    margin-bottom: 2rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
}
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    padding: 0;
}
.sub-title {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 1.5rem;
}
.metric-box {
    background: rgba(99, 102, 241, 0.05);
    border-radius: 10px;
    padding: 1.2rem;
    border: 1px solid rgba(99, 102, 241, 0.1);
    text-align: center;
    transition: transform 0.2s, border 0.2s;
}
.metric-box:hover {
    transform: translateY(-3px);
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #a855f7;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}
.citation-tag {
    background: rgba(168, 85, 247, 0.15);
    color: #d8b4fe;
    border: 1px solid rgba(168, 85, 247, 0.3);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}
.citation-card {
    background: rgba(255, 255, 255, 0.01);
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #a855f7;
    margin-bottom: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar settings
st.sidebar.image(
    "https://img.icons8.com/isometric/512/artificial-intelligence.png", width=80
)
st.sidebar.title("Configuration")

api_mode = st.sidebar.selectbox(
    "API Connection Mode", ["API Gateway (FastAPI)", "In-Process (Local Engine)"]
)
use_openai = st.sidebar.checkbox("Use OpenAI (requires key)", value=False)
if use_openai:
    key_input = st.sidebar.text_input(
        "OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", "")
    )
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input
else:
    st.sidebar.info(
        "Running in pure local offline mode using SentenceTransformers & BM25."
    )

# Backend API URL
API_URL = "http://localhost:8000"

# Title header
st.markdown(
    """
<div class="title-container">
    <h1 class="main-title">Production RAG Hub</h1>
    <p class="sub-title">Hybrid Retrieval (Dense + BM25) • Cross-Encoder Reranking • Real Eval Pipeline</p>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_eval = st.tabs(["⚡ Chat Interface", "📊 Evaluation Dashboard"])

# -------------------------------------------------------------------------
# Tab 1: Chat Interface
# -------------------------------------------------------------------------
with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Setup session state helper to fetch pipeline in local mode
    if api_mode == "In-Process (Local Engine)":
        db_path = "data/qdrant_db"
        encoder_exists = os.path.exists(os.path.join(db_path, "bm25_encoder.pkl"))

        if not encoder_exists:
            st.warning(
                "⚠️ RAG database not indexed. Click the button below to initialize ingestion."
            )
            if st.button("🚀 Initialize Database Ingestion"):
                with st.spinner(
                    "Chunking docs, training BM25, and building vectors..."
                ):
                    try:
                        from src.ingestion import ingest_documents

                        result = ingest_documents(
                            data_dir="data/raw_docs",
                            db_path=db_path,
                            collection_name="tech_docs",
                            use_openai=use_openai,
                        )
                        st.success(f"Success! Indexed {result['num_chunks']} chunks.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
        else:
            # Pre-load local classes
            if "retriever" not in st.session_state:
                from src.retrieval import HybridRetriever
                from src.reranking import CrossEncoderReranker

                st.session_state.retriever = HybridRetriever(
                    db_path=db_path, collection_name="tech_docs", use_openai=use_openai
                )
                st.session_state.reranker = CrossEncoderReranker()

    col_chat, col_sidebar = st.columns([7, 3])

    with col_chat:
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                # Show citations if present
                if "citations" in msg and msg["citations"]:
                    with st.expander("🔍 Citations & Sources"):
                        for cit_id, details in msg["citations"].items():
                            st.markdown(
                                f"""
                            <div class="citation-card">
                                <span class="citation-tag">{cit_id}</span> <strong>{details["source"]}</strong> (Score: {details["score"]:.4f})<<br>
                                <p style='color: #d1d5db; margin-top: 5px; font-size: 0.9rem;'>{details["text"]}</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

        # Chat Input
        if query := st.chat_input(
            "Ask about API Gateway, OAuth2, or Istio Service Mesh..."
        ):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            # Query backend/local pipeline
            with st.chat_message("assistant"):
                with st.spinner(
                    "Retrieving, reranking, and generating grounded answer..."
                ):
                    answer_text = ""
                    citations = {}
                    rewritten = ""
                    latency = 0.0

                    if api_mode == "API Gateway (FastAPI)":
                        try:
                            # Format chat history
                            history = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                            res = requests.post(
                                f"{API_URL}/ask",
                                json={
                                    "query": query,
                                    "chat_history": history,
                                    "use_openai": use_openai,
                                },
                                timeout=60,  # INCREASED from 15 to 60
                            )
                            if res.status_code == 200:
                                data = res.json()
                                answer_text = data["answer"]
                                citations = data["citation_details"]
                                rewritten = data["rewritten_query"]
                                latency = data["latency"]
                            else:
                                st.error(
                                    f"Error from API Gateway: {res.json()['detail']}"
                                )
                        except Exception as e:
                            st.error(
                                f"Failed to connect to API Gateway at {API_URL}. Switch connection mode to 'In-Process' in the sidebar to run locally. Error: {e}"
                            )
                    else:
                        # Local execution
                        import time

                        start_time = time.time()
                        try:
                            from src.retrieval import rewrite_query
                            from src.evaluation import run_production_rag

                            history = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                            result = run_production_rag(
                                query=query,
                                retriever=st.session_state.retriever,
                                reranker=st.session_state.reranker,
                                chat_history=history,
                                use_openai=use_openai,
                            )
                            answer_text = result["answer"]
                            citations = result["citation_details"]
                            rewritten = result.get("rewritten_query", query)
                            latency = time.time() - start_time
                        except Exception as e:
                            st.error(f"Local query execution failed: {e}")

                    if answer_text:
                        st.write(answer_text)
                        # Display citations inline expander
                        if citations:
                            with st.expander("🔍 Citations & Sources"):
                                for cit_id, details in citations.items():
                                    st.markdown(
                                        f"""
                                    <div class="citation-card">
                                        <span class="citation-tag">{cit_id}</span> <strong>{details["source"]}</strong> (Score: {details["score"]:.4f})<<br>
                                        <p style='color: #d1d5db; margin-top: 5px; font-size: 0.9rem;'>{details["text"]}</p>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations,
                            }
                        )

                        # Set query attributes for details column
                        st.session_state.latest_query_meta = {
                            "rewritten": rewritten,
                            "latency": latency,
                            "num_citations": len(citations),
                        }
                        st.rerun()

    with col_sidebar:
        st.markdown("### ⚡ Pipeline Tracing")
        if "latest_query_meta" in st.session_state:
            meta = st.session_state.latest_query_meta
            st.markdown(
                f"""
            <div class="glass-panel">
                <div style='font-size: 0.9rem; color: #9ca3af;'>Rewritten Query:</div>
                <div style='font-weight: 600; margin-bottom: 15px;'>"{meta["rewritten"]}"</div>

                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div class="metric-box">
                        <div class="metric-val">{meta["latency"]:.2f}s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{meta["num_citations"]}</div>
                        <div class="metric-label">Citations</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Submit a question to see real-time pipeline tracing, rewriting output, and latencies."
            )

# -------------------------------------------------------------------------
# Tab 2: Evaluation Dashboard
# -------------------------------------------------------------------------
with tab_eval:
    st.markdown("### 📊 Automated Evaluation Dashboard")
    st.markdown(
        "This dashboard compares the **Naive RAG (Baseline)** pipeline (direct vector similarity search) against our **Production RAG** pipeline (Query Rewriting + Hybrid Retrieval + RRF + Cross-Encoder Reranking) over the golden evaluation dataset."
    )

    # Ingestion Refresh helper
    col_btn1, col_btn2 = st.columns([3, 7])
    with col_btn1:
        trigger_eval = st.button(
            "🚀 Run Benchmark Suite (20 Test Cases)", use_container_width=True
        )
    with col_btn2:
        if st.button("🔄 Refresh / Re-ingest Documents", use_container_width=True):
            with st.spinner("Re-ingesting and updating indices..."):
                try:
                    if api_mode == "API Gateway (FastAPI)":
                        res = requests.post(
                            f"{API_URL}/ingest", params={"use_openai": use_openai}
                        )
                        if res.status_code == 200:
                            st.success("Re-ingestion completed on API Gateway.")
                        else:
                            st.error("Ingestion failed on API Gateway.")
                    else:
                        from src.ingestion import ingest_documents

                        result = ingest_documents(
                            data_dir="data/raw_docs",
                            db_path="data/qdrant_db",
                            collection_name="tech_docs",
                            use_openai=use_openai,
                        )
                        st.success(
                            f"Success! Local collection recreated with {result['num_chunks']} chunks."
                        )
                except Exception as e:
                    st.error(f"Re-ingestion error: {e}")

    # Load existing results if available
    results_file = "data/eval_results.json"
    eval_data = None

    if trigger_eval:
        with st.spinner(
            "Running evaluation. Running both pipelines side-by-side, evaluating queries, calculating recall and faithfulness..."
        ):
            try:
                if api_mode == "API Gateway (FastAPI)":
                    res = requests.post(
                        f"{API_URL}/eval",
                        params={"use_openai": use_openai},
                        timeout=300,  # INCREASED from 120 to 300
                    )
                    if res.status_code == 200:
                        eval_data = res.json()
                    else:
                        st.error(f"Evaluation request failed: {res.json()['detail']}")
                else:
                    from src.evaluation import run_full_evaluation

                    eval_data = run_full_evaluation(
                        dataset_path="data/eval_dataset.jsonl",
                        db_path="data/qdrant_db",
                        collection_name="tech_docs",
                        output_path=results_file,
                        use_openai=use_openai,
                    )
                    # Reload json file structure
                    with open(results_file, "r") as f:
                        eval_data = json.load(f)
                st.success("Evaluation suite complete!")
            except Exception as e:
                st.error(f"Failed to run evaluation: {e}")

    # Check if we have saved results file
    if eval_data is None and os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                eval_data = json.load(f)
            except Exception:
                pass

    if eval_data:
        aggregates = eval_data["aggregates"]
        detailed = pd.DataFrame(eval_data["detailed_results"])

        # Display Metrics Comparison Table
        st.markdown("#### Aggregate Metrics Summary")

        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Retrieval Recall",
                    "Retrieval F1",
                    "Mean Reciprocal Rank (MRR)",
                    "Faithfulness / Groundedness",
                    "Answer Relevance",
                    "Hallucination Rate",
                    "Avg Latency",
                ],
                "Naive RAG (Baseline)": [
                    f"{aggregates['naive']['retrieval_recall']:.3f}",
                    f"{aggregates['naive']['retrieval_f1']:.3f}",
                    f"{aggregates['naive']['mrr']:.3f}",
                    f"{aggregates['naive']['faithfulness']:.3f}",
                    f"{aggregates['naive']['relevance']:.3f}",
                    f"{aggregates['naive']['hallucination_rate']:.3f}",
                    f"{aggregates['naive']['latency']:.3f}s",
                ],
                "Production RAG": [
                    f"{aggregates['production']['retrieval_recall']:.3f}",
                    f"{aggregates['production']['retrieval_f1']:.3f}",
                    f"{aggregates['production']['mrr']:.3f}",
                    f"{aggregates['production']['faithfulness']:.3f}",
                    f"{aggregates['production']['relevance']:.3f}",
                    f"{aggregates['production']['hallucination_rate']:.3f}",
                    f"{aggregates['production']['latency']:.3f}s",
                ],
            }
        )

        st.table(metrics_df)

        # Visualizing with matplotlib
        st.markdown("#### Metric Comparisons Chart")

        metrics = [
            "retrieval_recall",
            "retrieval_f1",
            "mrr",
            "faithfulness",
            "relevance",
            "hallucination_rate",
        ]
        naive_vals = [aggregates["naive"][m] for m in metrics]
        prod_vals = [aggregates["production"][m] for m in metrics]

        plot_df = pd.DataFrame(
            {
                "Metric": [m.replace("_", " ").title() for m in metrics] * 2,
                "Score": naive_vals + prod_vals,
                "System": ["Naive Baseline"] * len(metrics)
                + ["Production RAG"] * len(metrics),
            }
        )

        # Plotting
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=plot_df,
            x="Metric",
            y="Score",
            hue="System",
            palette=["#ef4444", "#a855f7"],
            ax=ax,
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score / Rate")
        ax.set_xlabel("")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        # Category breakdowns
        st.markdown("#### Performance Breakdown by Category")
        cat_metrics = detailed.groupby("category")[
            [
                "naive_retrieval_f1",
                "prod_retrieval_f1",
                "naive_faithfulness",
                "prod_faithfulness",
                "naive_hallucination_rate",
                "prod_hallucination_rate",
            ]
        ].mean()
        st.dataframe(cat_metrics.style.format("{:.3f}"))

        # Detailed test case breakdown
        st.markdown("#### Detailed Test Cases Log")
        st.dataframe(
            detailed[
                [
                    "query",
                    "category",
                    "difficulty",
                    "naive_retrieval_f1",
                    "prod_retrieval_f1",
                    "naive_faithfulness",
                    "prod_faithfulness",
                    "naive_hallucination_rate",
                    "prod_hallucination_rate",
                ]
            ]
        )
    else:
        st.info(
            "No evaluation data found. Click 'Run Benchmark Suite' to execute evaluations on the golden set and render the comparative analytics dashboard."
        )
