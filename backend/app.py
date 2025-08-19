import os
import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import BM25SearchEngine, QdrantSearchEngine
from main import MultilingualArchitecturalRAGPipeline
import requests
from typing import Tuple


def update_user_location(latitude: float, longitude: float) -> bool:
    """Update global location coordinates"""
    global lat, lon, user_location_obtained
    try:
        lat = float(latitude)
        lon = float(longitude)
        user_location_obtained = True
        print(f"📍 Location updated to: ({lat}, {lon})")
        return True
    except Exception as e:
        print(f"Error updating location: {e}")
        return False

def get_user_location_ip() -> Tuple[float, float, bool]:
    """Get user location via IP geolocation"""
    global lat, lon, user_location_obtained
    
    try:
        # Try IP-based geolocation
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                new_lat = float(data.get('lat', lat))
                new_lon = float(data.get('lon', lon))
                update_user_location(new_lat, new_lon)
                return lat, lon, True
    except Exception as e:
        print(f"IP geolocation failed: {e}")
    
    return lat, lon, False

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
class LocationRequest(BaseModel):
    latitude: float
    longitude: float

class LocationResponse(BaseModel):
    success: bool
    message: str
    coordinates: Dict[str, float]

# Location globals (same as main.py)
lat = 18.52  # Default fallback (Pune)
lon = 73.88  # Default fallback (Pune)
user_location_obtained = False

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

@app.post("/location/manual", response_model=LocationResponse)
async def set_location_manual(payload: LocationRequest):
    """Manually set user location coordinates"""
    success = update_user_location(payload.latitude, payload.longitude)
    return {
        "success": success,
        "message": f"Location {'updated' if success else 'update failed'}",
        "coordinates": {"latitude": lat, "longitude": lon}
    }

@app.post("/location/auto", response_model=LocationResponse)
async def set_location_auto():
    """Auto-detect user location via IP"""
    new_lat, new_lon, success = get_user_location_ip()
    return {
        "success": success,
        "message": f"Auto-location {'detected' if success else 'failed, using default'}",
        "coordinates": {"latitude": new_lat, "longitude": new_lon}
    }

@app.get("/location/current", response_model=LocationResponse)
async def get_current_location():
    """Get current location coordinates"""
    return {
        "success": user_location_obtained,
        "message": f"Current location: {'User-provided' if user_location_obtained else 'Default fallback'}",
        "coordinates": {"latitude": lat, "longitude": lon}
    }

@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    global pipeline, lat, lon  # Add lat, lon to global access
    if pipeline is None:
        return {"answer": "⚠️ Pipeline not initialized"}

    # Update the pipeline to use current coordinates
    if hasattr(pipeline, 'weather_api'):
        print(f"🌍 Using coordinates: ({lat}, {lon})")
    
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