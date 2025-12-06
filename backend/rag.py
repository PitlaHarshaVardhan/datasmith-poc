from pathlib import Path
from typing import List
import logging

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import os

from .models import RetrievedDoc, RAGResult

logger = logging.getLogger(__name__)

# ----- Configure Gemini -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set – RAG will fail until it is configured.")

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"  # or "gemini-pro" if you prefer

DATA_DIR = Path(__file__).parent / "data"
REF_FILE = DATA_DIR / "nephrology_reference.txt"
CHROMA_DIR = Path(__file__).parent / "chroma"


# ----- Embedding helper -----

def get_embedding(text: str) -> List[float]:
    """Get embeddings from Gemini embedding model."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
    )
    # result["embedding"] is a list[float]
    return result["embedding"]


# ----- Initialize Chroma vector store -----

_chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

_COLLECTION_NAME = "nephrology_reference"
_collection = _chroma_client.get_or_create_collection(
    name=_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_vector_store_if_needed():
    if _collection.count() > 0:
        logger.info(
            "Chroma collection already initialized with %d docs",
            _collection.count(),
        )
        return

    logger.info("Building vector store from nephrology_reference.txt")
    text = REF_FILE.read_text(encoding="utf-8")
    chunks = _chunk_text(text)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": "nephrology_reference.txt", "chunk_id": i}
        for i in range(len(chunks))
    ]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    logger.info("Vector store built with %d chunks", len(chunks))


# ----- Retrieval + answer generation -----

def retrieve_docs(query: str, k: int = 4) -> List[RetrievedDoc]:
    build_vector_store_if_needed()
    query_embedding = get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    docs: List[RetrievedDoc] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append(
            RetrievedDoc(
                content=doc,
                source=str(meta.get("source", "unknown")),
                score=float(1 - dist),  # cosine similarity approx
            )
        )
    logger.info("Retrieved %d docs for query='%s'", len(docs), query)
    return docs


def generate_answer_with_rag(question: str, k: int = 4) -> RAGResult:
    docs = retrieve_docs(question, k=k)

    context_snippets = []
    for i, d in enumerate(docs, start=1):
        context_snippets.append(f"[{i}] (source: {d.source})\n{d.content}\n")
    context = "\n\n".join(context_snippets)

    prompt = (
        "You are a Clinical AI agent specialized in nephrology.\n"
        "Use the provided context from nephrology reference materials to answer the question.\n"
        "You must:\n"
        "1. Base your answer only on the context when possible.\n"
        "2. Use clear, simple language appropriate for a patient.\n"
        "3. Add inline citations like [Ref 1], [Ref 2] that correspond to the numbered context chunks.\n"
        "4. Always include the disclaimers:\n"
        "   'This is an AI assistant for educational purposes only.' and\n"
        "   'Always consult healthcare professionals for medical advice.'\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    answer = response.text or ""

    return RAGResult(
        answer=answer,
        docs=docs,
        used_web_search=False,
        web_sources=None,
    )
