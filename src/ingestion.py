import os
import re
import pickle
import math
from collections import Counter
from typing import List, Dict, Any, Tuple
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
)
from openai import OpenAI


# -------------------------------------------------------------------------
# BM25 Sparse Vector Encoder
# -------------------------------------------------------------------------
class BM25Encoder:
    def __init__(self, b: float = 0.75, k1: float = 1.2):
        self.b = b
        self.k1 = k1
        self.vocab = {}
        self.idf = {}
        self.avg_doc_len = 0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, corpus_texts: List[str]):
        tokenized_corpus = [self.tokenize(text) for text in corpus_texts]
        doc_lengths = [len(tokens) for tokens in tokenized_corpus]
        self.avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1

        vocab_set = set()
        for tokens in tokenized_corpus:
            vocab_set.update(tokens)
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}

        df = Counter()
        for tokens in tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        num_docs = len(corpus_texts)
        for token, freq in df.items():
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def encode(self, text: str, doc_len: int = None) -> Tuple[List[int], List[float]]:
        tokens = self.tokenize(text)
        if not tokens:
            return [], []

        term_counts = Counter(tokens)
        if doc_len is None:
            doc_len = len(tokens)

        indices = []
        values = []

        for token, count in term_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                tf = count
                score = (
                    idf
                    * (tf * (self.k1 + 1))
                    / (
                        tf
                        + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    )
                )
                indices.append(idx)
                values.append(float(score))

        if not indices:
            return [], []

        sorted_pairs = sorted(zip(indices, values))
        sorted_indices, sorted_values = zip(*sorted_pairs)
        return list(sorted_indices), list(sorted_values)


# -------------------------------------------------------------------------
# Embeddings Generator
# -------------------------------------------------------------------------
class EmbeddingGenerator:
    def __init__(self, use_openai: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.use_openai = use_openai
        self.model_name = model_name
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        if self.use_openai and self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
            self.dimension = 3072
            print("Using OpenAI text-embedding-3-large for dense vectors.")
        else:
            if self.use_openai:
                print(
                    "WARNING: OpenAI API Key not found. Falling back to local SentenceTransformer."
                )
                self.use_openai = False
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(
                f"Using local SentenceTransformer '{self.model_name}' (dimension: {self.dimension}) for dense vectors."
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            response = self.client.embeddings.create(
                model="text-embedding-3-large", input=texts
            )
            return [emb.embedding for emb in response.data]
        else:
            embeddings = self.model.encode(texts)
            return [emb.tolist() for emb in embeddings]


# -------------------------------------------------------------------------
# Document Ingestion Pipeline
# -------------------------------------------------------------------------
def ingest_documents(
    data_dir: str = "data/raw_docs",
    db_path: str = "data/qdrant_db",
    collection_name: str = "tech_docs",
    use_openai: bool = False,
    client: QdrantClient = None,  # NEW: accept shared client
) -> Dict[str, Any]:

    print(f"Starting document ingestion from {data_dir}...")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    raw_files = glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(
        os.path.join(data_dir, "*.txt")
    )
    if not raw_files:
        raise ValueError(
            f"No documents found in {data_dir}. Please add .md or .txt files."
        )

    documents = []
    for filepath in raw_files:
        basename = os.path.basename(filepath)
        print(f"Loading document: {basename}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(
            {
                "content": content,
                "source": basename,
                "id": re.sub(r"[^a-zA-Z0-9_]", "_", basename),
            }
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    corpus_texts = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunk_id = f"{doc['id']}_chunk_{i}"
            chunks.append(
                {
                    "text": split,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_id": chunk_id,
                        "doc_id": doc["id"],
                        "index": i,
                        "text": split,
                    },
                }
            )
            corpus_texts.append(split)

    print(f"Total chunks created: {len(chunks)}")

    print("Fitting BM25 sparse encoder...")
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(corpus_texts)

    encoder_path = os.path.join(os.path.dirname(db_path), "bm25_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(bm25_encoder, f)
    print(f"BM25 Encoder fitted and saved to {encoder_path}")

    embedding_gen = EmbeddingGenerator(use_openai=use_openai)

    # Use provided client or create new one
    own_client = client is None
    if own_client:
        print(f"Initializing Qdrant client at {db_path}...")
        client = QdrantClient(path=db_path)

    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=embedding_gen.dimension, distance="Cosine"),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        print("Generating dense embeddings & sparse vectors...")
        chunk_texts = [c["text"] for c in chunks]
        dense_embeddings = embedding_gen.get_embeddings(chunk_texts)

        points = []
        for i, (chunk, dense_emb) in enumerate(zip(chunks, dense_embeddings)):
            sparse_indices, sparse_values = bm25_encoder.encode(chunk["text"])

            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "dense": dense_emb,
                        "sparse": SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                    },
                    payload=chunk["metadata"],
                )
            )

        print(
            f"Upserting {len(points)} points into Qdrant collection '{collection_name}'..."
        )
        client.upsert(collection_name=collection_name, points=points)
        print("Ingestion pipeline successfully completed!")

    finally:
        # Only close if we created our own client
        if own_client:
            client.close()

    return {
        "status": "success",
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "dense_dimension": embedding_gen.dimension,
        "bm25_vocab_size": len(bm25_encoder.vocab),
    }


if __name__ == "__main__":
    result = ingest_documents(
        data_dir="data/raw_docs",
        db_path="data/qdrant_db",
        collection_name="tech_docs",
        use_openai=False,
    )
    print(result)
