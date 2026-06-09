import sys
import os

# ─────────────────────────────────────────────────────────────
# 1. ROBUST PROJECT ROOT RESOLUTION (before any src imports)
# ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient

# Try to import qdrant exception; fallback gracefully for older versions
try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except Exception:
    UnexpectedResponse = None

from src.ingestion import ingest_documents
from src.retrieval import HybridRetriever
from src.reranking import CrossEncoderReranker
from src.evaluation import run_production_rag, run_full_evaluation

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Production RAG API Gateway",
    description="Backend services for hybrid retrieval, reranking, and automated evaluation.",
    version="1.1.0",
)

# ─────────────────────────────────────────────────────────────
# Globals & Constants
# ─────────────────────────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
retriever: Optional[HybridRetriever] = None
reranker: Optional[CrossEncoderReranker] = None

DB_PATH = os.path.abspath("data/qdrant_db")
COLLECTION_NAME = "tech_docs"
BM25_PATH = os.path.abspath("data/bm25_encoder.pkl")  # FIXED: actual path
RAW_DOCS_DIR = os.path.abspath("data/raw_docs")
EVAL_DATASET_PATH = os.path.abspath("data/eval_dataset.jsonl")
EVAL_OUTPUT_PATH = os.path.abspath("data/eval_results.json")


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class QueryRequest(BaseModel):
    query: str = Field(
        ..., json_schema_extra={"example": "How do I configure rate limiting?"}
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=[], description="Previous conversation thread"
    )
    use_openai: Optional[bool] = Field(
        default=False, description="Whether to use OpenAI or local models"
    )


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    citations_used: List[str]
    citation_details: Dict[str, Any]
    latency: float


class IngestResponse(BaseModel):
    status: str
    message: str
    details: Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Singleton Qdrant client with on-disk persistence."""
    global _qdrant_client
    if _qdrant_client is None:
        os.makedirs(DB_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=DB_PATH)
        logger.info(f"Qdrant client initialized at: {DB_PATH}")
    return _qdrant_client


def collection_exists(client: QdrantClient, name: str) -> bool:
    """Robust collection existence check."""
    try:
        client.get_collection(name)
        return True
    except Exception as e:
        # Handle both UnexpectedResponse (404) and older client exceptions
        if UnexpectedResponse and isinstance(e, UnexpectedResponse):
            if e.status_code == 404:
                return False
            raise
        # Fallback: treat any exception as "not found" unless it's a connection error
        err_msg = str(e).lower()
        if "not found" in err_msg or "doesn't exist" in err_msg or "404" in err_msg:
            return False
        logger.warning(f"Collection check raised {type(e).__name__}: {e}")
        return False


def get_pipelines():
    """Lazy-init retriever & reranker. Validates DB only once."""
    global retriever, reranker

    # Already initialized? Return immediately (no Qdrant round-trip)
    if retriever is not None and reranker is not None:
        return retriever, reranker

    client = get_qdrant_client()

    # Validate collection exists
    if not collection_exists(client, COLLECTION_NAME):
        raise HTTPException(
            status_code=400,
            detail="RAG database not indexed. Collection does not exist. Please trigger /ingest first.",
        )

    # Validate collection has data
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        raise HTTPException(
            status_code=400,
            detail="RAG database is empty (0 chunks). Please trigger /ingest first.",
        )

    logger.info(
        f"Initializing pipelines against '{COLLECTION_NAME}' "
        f"({info.points_count} points)."
    )

    retriever = HybridRetriever(
        client=client,
        collection_name=COLLECTION_NAME,
        use_openai=False,
    )
    reranker = CrossEncoderReranker()

    logger.info("✅ Retriever & reranker ready.")
    return retriever, reranker


# ─────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load models if the DB is already populated."""
    try:
        client = get_qdrant_client()
        if collection_exists(client, COLLECTION_NAME):
            get_pipelines()
            logger.info("✅ Models pre-loaded successfully on startup.")
        else:
            logger.warning(
                "⚠️ Collection not found. Models will load after first /ingest."
            )
    except HTTPException:
        logger.warning(
            "⚠️ DB not ready. Models will load on first request after ingestion."
        )
    except Exception as e:
        logger.error(f"⚠️ Startup error: {e}")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Production RAG API Gateway. Access /docs for Swagger UI."
    }


@app.get("/health")
def health_check():
    """Check backend and database status."""
    client = get_qdrant_client()
    db_exists = collection_exists(client, COLLECTION_NAME)
    count = 0
    if db_exists:
        count = client.get_collection(COLLECTION_NAME).points_count
    return {
        "status": "ok",
        "database_ready": db_exists,
        "chunk_count": count,
        "bm25_encoder_exists": os.path.exists(BM25_PATH),
    }


@app.post("/ingest", response_model=IngestResponse)
def run_ingestion(use_openai: bool = False):
    try:
        if not os.path.isdir(RAW_DOCS_DIR):
            raise HTTPException(
                status_code=400,
                detail=f"Raw documents directory not found: {RAW_DOCS_DIR}",
            )

        result = ingest_documents(
            data_dir=RAW_DOCS_DIR,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            use_openai=use_openai,
            client=get_qdrant_client(),
        )

        # Reset singletons so next request picks up the new data
        global retriever, reranker
        retriever = None
        reranker = None
        logger.info("Ingestion complete. Retriever cache cleared for hot-reload.")

        return IngestResponse(
            status="success",
            message="Document ingestion completed successfully.",
            details=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    start_time = time.time()

    local_retriever, local_reranker = get_pipelines()

    history_dicts = [
        {"role": msg.role, "content": msg.content} for msg in request.chat_history
    ]

    try:
        result = run_production_rag(
            query=request.query,
            retriever=local_retriever,
            reranker=local_reranker,
            chat_history=history_dicts,
            use_openai=request.use_openai,
        )

        latency = time.time() - start_time

        return QueryResponse(
            query=request.query,
            rewritten_query=result.get("rewritten_query", request.query),
            answer=result["answer"],
            citations_used=result.get("citations_used", []),
            citation_details=result.get("citation_details", {}),
            latency=latency,
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval")
def run_eval(use_openai: bool = False):
    try:
        client = get_qdrant_client()

        # FIXED: Use correct BM25 path + Qdrant collection check
        db_ready = (
            os.path.exists(BM25_PATH)
            and collection_exists(client, COLLECTION_NAME)
            and client.get_collection(COLLECTION_NAME).points_count > 0
        )

        if not db_ready:
            logger.info(
                "Database not found or empty. Triggering automated ingestion..."
            )
            if not os.path.isdir(RAW_DOCS_DIR):
                raise HTTPException(
                    status_code=400,
                    detail=f"Raw documents directory missing: {RAW_DOCS_DIR}",
                )

            ingest_documents(
                data_dir=RAW_DOCS_DIR,
                db_path=DB_PATH,
                collection_name=COLLECTION_NAME,
                use_openai=use_openai,
                client=client,
            )
            global retriever, reranker
            retriever = None
            reranker = None

        if not os.path.exists(EVAL_DATASET_PATH):
            raise HTTPException(
                status_code=400,
                detail=f"Evaluation dataset not found: {EVAL_DATASET_PATH}",
            )

        aggregates = run_full_evaluation(
            dataset_path=EVAL_DATASET_PATH,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            output_path=EVAL_OUTPUT_PATH,
            use_openai=use_openai,
            client=client,
        )

        with open(EVAL_OUTPUT_PATH, "r", encoding="utf-8") as f:
            full_results = json.load(f)

        return full_results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
