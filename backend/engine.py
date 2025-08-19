# engine2.py
import google.generativeai as genai
from typing import List, Dict

import os
import json
import nltk
import asyncio
import hashlib
import re
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from rank_bm25 import BM25Okapi
import warnings

warnings.filterwarnings("ignore")

# Download NLTK data
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # some langchain installs expose it from langchain.text_splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("punkt_tab")


class ProcessingMode(Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class InformationType(Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"


@dataclass
class TokenizedQuery:
    raw_query: str
    tokens: List[str]
    filtered_tokens: List[str]
    language: str
    intent: str
    processing_mode: ProcessingMode


@dataclass
class InformationPacket:
    content: str
    info_type: InformationType
    confidence: float
    source: str
    relevance_score: float
    economic_value: float


@dataclass
class ReasoningResult:
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    supporting_evidence: List[str]


def make_id(content: str) -> str:
    """Create a unique ID for content"""
    return hashlib.md5(content.encode()).hexdigest()


def convert_contents_to_text(content) -> str:
    """Convert content to text"""
    return content if isinstance(content, str) else str(content)

def tokenize_words(text: str) -> List[str]:
    """Tokenize into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def detokenize_words(tokens: List[str]) -> str:
    """Rebuild text from tokens."""
    out = []
    for i, token in enumerate(tokens):
        if i > 0 and re.match(r"\w", token) and re.match(r"\w", out[-1]):
            out.append(" ")
        out.append(token)
    return "".join(out)

def chunk_simple_with_ngrams(text: str, chunk_size: int = 500, overlap: int = 50, n: int = 3) -> List[str]:
    """
    Uses RecursiveCharacterTextSplitter but prevents splitting inside n-grams.
    """
    tokens = tokenize_words(text)
    token_text = detokenize_words(tokens)  # normalize spacing

    def length_fn(t: str) -> int:
        return len(tokenize_words(t))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.split_text(token_text)
    adjusted_chunks = []

    for chunk in raw_chunks:
        chunk_tokens = tokenize_words(chunk)
        # Extend chunk so it ends on n-gram boundary
        if len(chunk_tokens) % n != 0:
            remainder = n - (len(chunk_tokens) % n)
            # Try to pull tokens from the next raw chunk if available
            next_index = raw_chunks.index(chunk) + 1
            if next_index < len(raw_chunks):
                extra_tokens = tokenize_words(raw_chunks[next_index])[:remainder]
                chunk_tokens.extend(extra_tokens)
        adjusted_chunks.append(detokenize_words(chunk_tokens))

    return adjusted_chunks


class QueryEngine:
    """Handles initial query processing and language detection"""
    
    def __init__(self):
        self.supported_languages = ["english", "hindi"]
    
    async def process_query(self, raw_query: str) -> TokenizedQuery:
        """Process raw query into structured format"""
        # Language detection (simplified)
        language = self._detect_language(raw_query)
        
        # Tokenization
        tokens = self._tokenize(raw_query)
        
        # Intent detection
        intent = self._detect_intent(raw_query)
        
        # Processing mode determination
        processing_mode = self._determine_processing_mode(intent, raw_query)
        
        # Token filtering
        filtered_tokens = self._filter_tokens(tokens)
        
        return TokenizedQuery(
            raw_query=raw_query,
            tokens=tokens,
            filtered_tokens=filtered_tokens,
            language=language,
            intent=intent,
            processing_mode=processing_mode
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return "hindi" if hindi_chars > len(text) * 0.1 else "english"
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter out stop words and irrelevant tokens"""
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "steps", "process"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "because", "reason"]):
            return "causal"
        else:
            return "general"
    
    def _determine_processing_mode(self, intent: str, query: str) -> ProcessingMode:
        """Determine if query needs offline or online processing"""
        # Simple heuristic: complex queries go offline
        if len(query.split()) > 10 or intent in ["causal", "procedural"]:
            return ProcessingMode.OFFLINE
        return ProcessingMode.ONLINE

class GeminiLLM:
    """Gemini 1.5 Flash LLM wrapper"""

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        # Use your Gemini API key
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")  # Or hardcode if needed
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Gemini"""
        prompt = self._messages_to_prompt(messages)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n\n"
        prompt += "Assistant: "
        return prompt


class ReasoningEngine:
    """Handles reasoning based on English/Hindi input"""
    
    def __init__(self):
        self.reasoning_patterns = {
            "definition": self._definitional_reasoning,
            "procedural": self._procedural_reasoning,
            "causal": self._causal_reasoning,
            "general": self._general_reasoning
        }
    
    async def reason(self, query: TokenizedQuery, context: List[Dict]) -> ReasoningResult:
        """Apply reasoning based on query intent and language"""
        reasoning_func = self.reasoning_patterns.get(query.intent, self._general_reasoning)
        return await reasoning_func(query, context)
    
    async def _definitional_reasoning(self, query: TokenizedQuery, context: List[Dict]) -> ReasoningResult:
        """Reasoning for definitional queries"""
        reasoning_path = [
            f"Identified definitional query: {query.raw_query}",
            "Searching for core concepts and definitions",
            "Synthesizing authoritative definition"
        ]
        
        # Extract key terms for definition
        key_terms = [token for token in query.filtered_tokens if len(token) > 3]
        
        conclusion = f"Based on the context, this appears to be about: {', '.join(key_terms)}"
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.8,
            reasoning_path=reasoning_path,
            supporting_evidence=[doc.get("text", "")[:100] + "..." for doc in context[:2]]
        )
    
    async def _procedural_reasoning(self, query: TokenizedQuery, context: List[Dict]) -> ReasoningResult:
        """Reasoning for how-to queries"""
        reasoning_path = [
            f"Identified procedural query: {query.raw_query}",
            "Looking for step-by-step information",
            "Organizing information in logical sequence"
        ]
        
        conclusion = "This requires a step-by-step approach based on the available information."
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.75,
            reasoning_path=reasoning_path,
            supporting_evidence=[doc.get("text", "")[:100] + "..." for doc in context[:3]]
        )
    
    async def _causal_reasoning(self, query: TokenizedQuery, context: List[Dict]) -> ReasoningResult:
        """Reasoning for why/because queries"""
        reasoning_path = [
            f"Identified causal query: {query.raw_query}",
            "Analyzing cause-effect relationships",
            "Building logical causal chain"
        ]
        
        conclusion = "This involves understanding underlying causes and relationships."
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.7,
            reasoning_path=reasoning_path,
            supporting_evidence=[doc.get("text", "")[:100] + "..." for doc in context[:2]]
        )
    
    async def _general_reasoning(self, query: TokenizedQuery, context: List[Dict]) -> ReasoningResult:
        """General reasoning for other queries"""
        reasoning_path = [
            f"Processing general query: {query.raw_query}",
            "Applying general reasoning patterns",
            "Synthesizing relevant information"
        ]
        
        conclusion = "General information synthesis based on available context."
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.6,
            reasoning_path=reasoning_path,
            supporting_evidence=[doc.get("text", "")[:100] + "..." for doc in context[:2]]
        )


class InformationFilter:
    """Filters and processes information packets"""
    
    def __init__(self):
        self.quality_threshold = 0.3
        self.relevance_threshold = 0.4
    
    async def filter_information(self, raw_docs: List[Dict], query: TokenizedQuery) -> List[InformationPacket]:
        """Filter and classify information"""
        packets = []
        
        for doc in raw_docs:
            # Classify information type
            info_type = self._classify_info_type(doc.get("text", ""), query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(doc, query)
            
            # Calculate relevance
            relevance = self._calculate_relevance(doc, query)
            
            # Calculate economic value
            economic_value = self._calculate_economic_value(doc, query)
            
            # Filter based on thresholds
            if confidence >= self.quality_threshold and relevance >= self.relevance_threshold:
                packet = InformationPacket(
                    content=doc.get("text", ""),
                    info_type=info_type,
                    confidence=confidence,
                    source=doc.get("uri", "unknown"),
                    relevance_score=relevance,
                    economic_value=economic_value
                )
                packets.append(packet)
        
        # Sort by combined score
        packets.sort(key=lambda p: p.relevance_score * p.confidence * p.economic_value, reverse=True)
        
        return packets
    
    def _classify_info_type(self, text: str, query: TokenizedQuery) -> InformationType:
        """Classify information type"""
        text_lower = text.lower()
        
        if query.intent == "definition" or any(word in text_lower for word in ["is a", "refers to", "definition"]):
            return InformationType.FACTUAL
        elif query.intent == "procedural" or any(word in text_lower for word in ["step", "process", "how to"]):
            return InformationType.PROCEDURAL
        elif any(word in text_lower for word in ["concept", "theory", "principle"]):
            return InformationType.CONCEPTUAL
        else:
            return InformationType.FACTUAL
    
    def _calculate_confidence(self, doc: Dict, query: TokenizedQuery) -> float:
        """Calculate confidence score"""
        base_score = doc.get("score", 0.5)
        
        # Boost for exact matches
        text_lower = doc.get("text", "").lower()
        exact_matches = sum(1 for token in query.filtered_tokens if token in text_lower)
        match_boost = min(exact_matches * 0.1, 0.3)
        
        return min(base_score + match_boost, 1.0)
    
    def _calculate_relevance(self, doc: Dict, query: TokenizedQuery) -> float:
        """Calculate relevance score"""
        text_lower = doc.get("text", "").lower()
        query_tokens = set(query.filtered_tokens)
        text_tokens = set(text_lower.split())
        
        # Jaccard similarity
        intersection = len(query_tokens & text_tokens)
        union = len(query_tokens | text_tokens)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_economic_value(self, doc: Dict, query: TokenizedQuery) -> float:
        """Calculate economic value (processing cost vs utility)"""
        # Simple heuristic: shorter, more relevant docs have higher economic value
        text_length = len(doc.get("text", ""))
        relevance = self._calculate_relevance(doc, query)
        
        # Inverse relationship with length, direct with relevance
        length_penalty = max(0.1, 1.0 - (text_length / 1000))
        economic_value = relevance * length_penalty
        
        return min(economic_value, 1.0)


class TokenizationDB:
    """Handles tokenization and stores processed tokens"""
    
    def __init__(self):
        self.token_cache = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    async def tokenize_and_embed(self, information_packets: List[InformationPacket]) -> Dict[str, Any]:
        """Tokenize information and create embeddings"""
        tokenization_result = {
            "tokens": [],
            "embeddings": [],
            "token_map": {}
        }
        
        for packet in information_packets:
            # Tokenize content
            tokens = self._advanced_tokenize(packet.content)
            
            # Create embeddings
            embedding = self.embedding_model.encode(packet.content)
            
            # Store in result
            packet_id = make_id(packet.content)
            tokenization_result["tokens"].append(tokens)
            tokenization_result["embeddings"].append(embedding)
            tokenization_result["token_map"][packet_id] = {
                "tokens": tokens,
                "embedding": embedding,
                "packet": packet
            }
        
        return tokenization_result
    
    def _advanced_tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with normalization"""
        # Basic tokenization
        tokens = text.lower().split()
        
        # Remove punctuation and normalize
        normalized_tokens = []
        for token in tokens:
            clean_token = re.sub(r'[^\w]', '', token)
            if len(clean_token) > 2:
                normalized_tokens.append(clean_token)
        
        return normalized_tokens


class MarketStorageEconomy:
    """Handles storage economics and market considerations"""
    
    def __init__(self):
        self.storage_costs = {"memory": 0.1, "disk": 0.05, "cloud": 0.02}
        self.processing_costs = {"cpu": 0.01, "gpu": 0.05, "api": 0.1}
        self.cache_limit = 1000
        self.cache = {}
    
    async def optimize_storage(self, tokenization_data: Dict[str, Any], query: TokenizedQuery) -> Dict[str, Any]:
        """Optimize storage based on economic factors"""
        optimized_data = {
            "high_value": [],
            "medium_value": [],
            "low_value": [],
            "storage_plan": {}
        }
        
        # Classify data by economic value
        for packet_id, data in tokenization_data["token_map"].items():
            packet = data["packet"]
            
            if packet.economic_value > 0.7:
                optimized_data["high_value"].append(data)
                storage_type = "memory"
            elif packet.economic_value > 0.4:
                optimized_data["medium_value"].append(data)
                storage_type = "disk"
            else:
                optimized_data["low_value"].append(data)
                storage_type = "cloud"
            
            optimized_data["storage_plan"][packet_id] = {
                "storage_type": storage_type,
                "cost": self.storage_costs[storage_type],
                "priority": packet.economic_value
            }
        
        # Cache high-value items
        self._update_cache(optimized_data["high_value"])
        
        return optimized_data
    
    def _update_cache(self, high_value_data: List[Dict]):
        """Update cache with high-value data"""
        for data in high_value_data:
            packet = data["packet"]
            cache_key = make_id(packet.content)
            
            if len(self.cache) >= self.cache_limit:
                # Remove lowest value item
                min_key = min(self.cache.keys(), key=lambda k: self.cache[k]["packet"].economic_value)
                del self.cache[min_key]
            
            self.cache[cache_key] = data
    
    async def calculate_processing_cost(self, query: TokenizedQuery, data_size: int) -> Dict[str, float]:
        """Calculate total processing cost"""
        base_cost = self.processing_costs["cpu"] * data_size
        
        if query.processing_mode == ProcessingMode.OFFLINE:
            # Offline processing is cheaper but slower
            total_cost = base_cost * 0.8
        else:
            # Online processing costs more for speed
            total_cost = base_cost * 1.5
        
        return {
            "base_cost": base_cost,
            "total_cost": total_cost,
            "mode": query.processing_mode.value,
            "efficiency_score": base_cost / total_cost if total_cost > 0 else 1.0
        }


class CombiningEngine:
    """Combines information from different sources"""
    
    def __init__(self):
        self.combination_strategies = {
            "weighted_average": self._weighted_average_combination,
            "max_confidence": self._max_confidence_combination,
            "consensus": self._consensus_combination
        }
    
    async def combine_information(self, 
                                optimized_data: Dict[str, Any], 
                                reasoning_result: ReasoningResult,
                                query: TokenizedQuery) -> Dict[str, Any]:
        """Combine information using multiple strategies"""
        
        # Select combination strategy based on query type
        strategy = self._select_strategy(query, reasoning_result)
        combine_func = self.combination_strategies[strategy]
        
        # Combine high and medium value information
        all_packets = []
        for data in optimized_data["high_value"] + optimized_data["medium_value"]:
            all_packets.append(data["packet"])
        
        combined_result = await combine_func(all_packets, reasoning_result, query)
        
        return {
            "combined_information": combined_result,
            "strategy_used": strategy,
            "confidence": reasoning_result.confidence,
            "source_count": len(all_packets)
        }
    
    def _select_strategy(self, query: TokenizedQuery, reasoning_result: ReasoningResult) -> str:
        """Select appropriate combination strategy"""
        if reasoning_result.confidence > 0.8:
            return "max_confidence"
        elif query.intent in ["definition", "factual"]:
            return "consensus"
        else:
            return "weighted_average"
    
    async def _weighted_average_combination(self, packets: List[InformationPacket], 
                                          reasoning: ReasoningResult, 
                                          query: TokenizedQuery) -> str:
        """Combine using weighted average of confidence scores"""
        if not packets:
            return reasoning.conclusion
        
        # Weight by confidence and relevance
        weighted_content = []
        total_weight = 0
        
        for packet in packets:
            weight = packet.confidence * packet.relevance_score
            weighted_content.append(f"({weight:.2f}) {packet.content[:200]}...")
            total_weight += weight
        
        combined = f"Reasoning: {reasoning.conclusion}\n\nWeighted Evidence:\n" + "\n".join(weighted_content)
        return combined
    
    async def _max_confidence_combination(self, packets: List[InformationPacket], 
                                        reasoning: ReasoningResult, 
                                        query: TokenizedQuery) -> str:
        """Use highest confidence information"""
        if not packets:
            return reasoning.conclusion
        
        max_packet = max(packets, key=lambda p: p.confidence)
        return f"Based on highest confidence source: {max_packet.content}"
    
    async def _consensus_combination(self, packets: List[InformationPacket], 
                                   reasoning: ReasoningResult, 
                                   query: TokenizedQuery) -> str:
        """Find consensus among sources"""
        if not packets:
            return reasoning.conclusion
        
        # Simple consensus: use most common information type
        type_counts = {}
        for packet in packets:
            type_counts[packet.info_type] = type_counts.get(packet.info_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        relevant_packets = [p for p in packets if p.info_type == dominant_type]
        
        consensus_content = "\n".join([p.content[:150] + "..." for p in relevant_packets[:3]])
        return f"Consensus from {dominant_type.value} sources:\n{consensus_content}"


# Legacy components for compatibility
class OllamaLLM:
    """Ollama LLM wrapper"""
    
    def __init__(self, model: str = "mistral"):
        self.client = ollama.Client()
        self.model = model
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Ollama"""
        prompt = self._messages_to_prompt(messages)
        try:
            result = self.client.generate(model=self.model, prompt=prompt, **kwargs)
            return result["response"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n\n"
        prompt += "Assistant: "
        return prompt



class QdrantSearchEngine:
    """Qdrant vector database search engine"""
    
    def __init__(self, collection_name: str = "AGRICULTURE", host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(
    url="https://ea333494-3863-460f-840e-f5e8c51384bd.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DG8-n96QYTB9-WadofSKUmoSF-5yKiY0FkUCr7ixloM",
    )       
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = collection_name
        self.documents: Dict[int, Dict[str, Any]] = {}


    def fetch_all_documents(self, batch_size: int = 1000):
        """Retrieve all stored documents from Qdrant collection."""
        all_docs = []
        scroll_offset = None

        while True:
            points, scroll_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=scroll_offset
            )
            for p in points:
                payload = p.payload or {}
                if "text" in payload:
                    all_docs.append(payload)
            if scroll_offset is None:
                break

        return all_docs

    
    async def fit(self, documents: List[Dict[str, Any]]):
        """Index documents in Qdrant"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts).tolist()
            
        points = []
        for doc, emb in zip(documents, embeddings):
            pid = doc.get("chunk_id")  # stable, unique per chunk
            payload = {
                "text": doc.get("text", ""),
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "chunk_index": doc.get("chunk_index"),
                "chunk_length": doc.get("chunk_length"),
                "file_name": doc.get("file_name"),
                "uri": doc.get("uri"),
                # add page_start/page_end later if you compute them
            }
            points.append(PointStruct(id=pid, vector=emb, payload=payload))
        if len(points) >= 1000:
            self.client.upsert(collection_name=self.collection_name, points=points)
            points = []        
        self.client.upsert(collection_name=self.collection_name, points=points)
        return self
    
    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        emb = self.model.encode([query])[0].tolist()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=emb,
            limit=top_k,
            query_filter=filters
        )
        results = []
        for h in hits:
            payload = h.payload or {}
            text_val = payload.get("text", "")
            results.append({"text": text_val, "score": h.score, **payload})
        return results


class BM25SearchEngine:
    """BM25 based search engine"""
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
    
    async def fit(self, documents: List[Dict[str, Any]]):
        """Fit BM25 on documents"""
        self.documents = documents
        corpus = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        return self
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [{"text": self.documents[i]["text"], "score": float(scores[i]), **self.documents[i]} for i in top_idxs]


# Shared instances
llm = GeminiLLM(model = "gemini-1.5-flash", api_key = 'AIzaSyBqfF_Ue1rt0B24NNM8Q1WkMryY9hFEui4')  # or GeminiLLM(model="gemini-1.5-flash")
query_engine = QueryEngine()
reasoning_engine = ReasoningEngine()
info_filter = InformationFilter()
tokenization_db = TokenizationDB()
market_economy = MarketStorageEconomy()
combining_engine = CombiningEngine()


class ArchitecturalRAGPipeline:
    """Complete RAG pipeline with strict document-grounded retrieval"""

    def __init__(self, bm25_engine, qdrant_engine):
        self.bm25_engine = bm25_engine
        self.qdrant_engine = qdrant_engine

    async def invoke(self, raw_query: str, top_k: int = 10, score_threshold: float = 0.75) -> Dict[str, Any]:
        """Executes the full RAG pipeline"""

        start_time = time.time()

        # 1. Query Processing
        tokenized_query = await query_engine.process_query(raw_query)

        # 2. Dense retrieval from Qdrant (primary)
        dense_results = await self.qdrant_engine.search(raw_query, top_k)
        dense_results = [r for r in dense_results if r.get("score", 0) >= score_threshold]

        # 3. Fallback to BM25 if dense retrieval is too weak
        sparse_results = []
        if not dense_results:
            sparse_results = await self.bm25_engine.search(raw_query, top_k)

        # 4. Merge final results
        all_results = dense_results if dense_results else sparse_results

        # 5. If still no results, return early
        if not all_results:
            return {
                "answer": "No relevant information found in the provided documents.",
                "query_analysis": {"original_query": raw_query},
                "performance": {
                    "processing_time": time.time() - start_time,
                    "source_documents": 0
                }
            }

        # 6. Reasoning Engine
        reasoning_result = await reasoning_engine.reason(tokenized_query, all_results)

        # 7. Information Filtering
        information_packets = await info_filter.filter_information(all_results, tokenized_query)

        # 8. Tokenization DB
        tokenization_data = await tokenization_db.tokenize_and_embed(information_packets)

        # 9. Market Storage Economy
        optimized_data = await market_economy.optimize_storage(tokenization_data, tokenized_query)
        processing_cost = await market_economy.calculate_processing_cost(tokenized_query, len(all_results))

        # 10. Combining Engine
        combined_result = await combining_engine.combine_information(
            optimized_data, reasoning_result, tokenized_query
        )

        # 11. Build context from retrieved docs
        context_text = "\n\n".join(
            r.get("text") or r.get("content", "") for r in all_results if r.get("text") or r.get("content")
        )
        context_text = context_text.strip()

        # 12. If somehow context is still empty, fail gracefully
        if not context_text:
            return {
                "answer": "No relevant information found in the provided documents.",
                "query_analysis": {"original_query": raw_query},
                "performance": {
                    "processing_time": time.time() - start_time,
                    "source_documents": len(all_results)
                }
            }

        # 13. LLM Prompt — strictly document-grounded
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Only use the provided document context to answer the question. "
                    "If the answer is not explicitly contained in the context, reply with: "
                    "'No relevant information found in the provided documents.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question:\n{raw_query}\n\n"
                    f"Reasoning summary:\n{reasoning_result.conclusion}"
                )
            }
        ]

        final_answer = await llm.generate(messages)

        processing_time = time.time() - start_time

        return {
            "answer": final_answer,
            "context_used": context_text,  # ADD THIS LINE
            "query_analysis": {
                "original_query": raw_query,
                "tokenized_query": tokenized_query,
                "processing_mode": tokenized_query.processing_mode.value,
                "language": tokenized_query.language,
                "intent": tokenized_query.intent
            },
            "reasoning": {
                "conclusion": reasoning_result.conclusion,
                "confidence": reasoning_result.confidence,
                "reasoning_path": reasoning_result.reasoning_path
            },
            "information_processing": {
                "total_packets": len(information_packets),
                "high_value_count": len(optimized_data["high_value"]),
                "medium_value_count": len(optimized_data["medium_value"]),
                "low_value_count": len(optimized_data["low_value"])
            },
            "economics": {
                "processing_cost": processing_cost,
                "storage_optimization": optimized_data["storage_plan"],
                "combination_strategy": combined_result["strategy_used"]
            },
            "performance": {
                "processing_time": processing_time,
                "source_documents": len(all_results)
            }
        }

# Legacy compatibility classes
class HybridRetriever:
    """Legacy hybrid retriever for backward compatibility"""
    
    def __init__(self, sparse_engine, dense_engine, sparse_weight: float = 0.5, dense_weight: float = 0.5):
        self.pipeline = ArchitecturalRAGPipeline(sparse_engine, dense_engine)
    
    async def invoke(self, query: str, top_k: int = 10, top_n: int = 5) -> List[Dict[str, Any]]:
        result = await self.pipeline.invoke(query, top_k)
        # Return simplified format for compatibility
        return [{"text": "Combined result", "combined_score": 1.0, "chunk_id": make_id(query)}]


class SimpleResponseGenerator:
    """Legacy response generator"""
    
    def __init__(self):
        pass
    
    async def invoke(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Use simplified approach for compatibility
        context = "\n\n".join(f"Document {i+1}:\n{d.get('text', '')}" for i, d in enumerate(docs))
        messages = [
            {"role": "system", "content": f"Answer using context: {context}"},
            {"role": "user", "content": query}
        ]
        answer = await llm.generate(messages)
        return {"answer": answer, "context": docs, "query": query}


class SimpleRAGPipeline:
    """Legacy simple pipeline that uses new architecture"""
    
    def __init__(self, retriever, generator):
        if hasattr(retriever, 'pipeline'):
            self.architectural_pipeline = retriever.pipeline
        else:
            # Fallback for non-architectural retrievers
            self.retriever = retriever
            self.generator = generator
    
    async def invoke(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if hasattr(self, 'architectural_pipeline'):
            result = await self.architectural_pipeline.invoke(query, top_k)
            return {
                "answer": result["answer"],
                "retrieved_documents": [],
                "query": query
            }
        else:
            # Fallback to simple approach
            docs = await self.retriever.invoke(query, top_k=top_k)
            response = await self.generator.invoke(query, docs)
            return {"answer": response["answer"], "retrieved_documents": docs, "query": query}
        




# chunking, reducing data feed to BM25, supporting_evidence, how to handle database info, language biasness
# whisper audio to text, issue in language conversion 