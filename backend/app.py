# app.py
import os
import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import BM25SearchEngine, QdrantSearchEngine
from main import MultilingualArchitecturalRAGPipeline

# ------------ FastAPI & CORS ------------
app = FastAPI(title="Multilingual RAG Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Globals ------------
pipeline: MultilingualArchitecturalRAGPipeline | None = None


def _to_texts(items: Any) -> List[Dict[str, Any]]:
    """Ensure docs are returned in dict[text] format for BM25."""
    docs: List[Dict[str, Any]] = []
    if not items:
        return docs
    for it in items:
        if isinstance(it, dict):
            if "text" in it:
                docs.append(it)
            elif "content" in it:
                docs.append({"text": it["content"]})
            elif "page_content" in it:
                docs.append({"text": it["page_content"]})
        elif isinstance(it, str):
            docs.append({"text": it})
        else:
            v = getattr(it, "page_content", None)
            docs.append({"text": v if isinstance(v, str) else str(it)})
    return docs


# ------------ Startup: load data first ------------
@app.on_event("startup")
async def _startup():
    global pipeline

    coll = os.getenv("QDRANT_COLLECTION", "AGRICULTURE")
    print("🌟 Initializing pipeline with collection:", coll)

    qdrant = QdrantSearchEngine(collection_name=coll)

    # fetch_all_documents is sync in your QdrantSearchEngine
    docs = qdrant.fetch_all_documents()
    texts = _to_texts(docs)
    print(f"✓ Retrieved {len(texts)} docs from Qdrant")

    # Fit BM25 with docs (async)
    bm25 = BM25SearchEngine()
    await bm25.fit(texts)

    print("✓ BM25 fitted")
    pipeline = MultilingualArchitecturalRAGPipeline(bm25, qdrant)
    print("✓ Pipeline ready")


# ------------ Schemas ------------
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    query_analysis: Dict[str, Any] | None = None
    performance: Dict[str, Any] | None = None
    economics: Dict[str, Any] | None = None
    weather_context_provided: bool | None = None
    multilingual_info: Dict[str, Any] | None = None
    retrieved_documents: List[Dict[str, Any]] | None = None


# ------------ Endpoints ------------
@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    global pipeline
    if pipeline is None:
        return {"answer": "❌ Pipeline not initialized"}

    result = await pipeline.invoke(payload.query)
    return {
        "answer": result.get("answer", ""),
        "query_analysis": result.get("query_analysis"),
        "performance": result.get("performance"),
        "economics": result.get("economics"),
        "weather_context_provided": result.get("weather_context_provided"),
        "multilingual_info": result.get("multilingual_info"),
        "retrieved_documents": result.get("retrieved_documents"),
    }
