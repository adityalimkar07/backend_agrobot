# main2.py
# docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

from geopy.geocoders import Nominatim
import re

from weather_api import WeatherAPI
import asyncio
from engine import (
    BM25SearchEngine, QdrantSearchEngine,
    ArchitecturalRAGPipeline,
    query_engine, reasoning_engine, info_filter,
    tokenization_db, market_economy, combining_engine
)
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

# Global variables for user location (will be updated when location is obtained)
lat = 18.52  # Default fallback (Pune)
lon = 73.88  # Default fallback (Pune)
user_location_obtained = False

def get_user_location():
    """Get user's current location coordinates"""
    global lat, lon, user_location_obtained
    
    try:
        # For Python scripts, we can use requests to get location from IP
        import requests
        
        # Method 1: Try IP-based geolocation (most reliable for Python)
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    lat = float(data.get('lat', lat))
                    lon = float(data.get('lon', lon))
                    user_location_obtained = True
                    print(f"📍 Location obtained via IP: {data.get('city', 'Unknown')}, {data.get('regionName', 'Unknown')} ({lat}, {lon})")
                    return True
        except Exception as e:
            print(f"IP geolocation failed: {e}")
        
        # Method 2: Try alternative IP service
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'latitude' in data and 'longitude' in data:
                    lat = float(data['latitude'])
                    lon = float(data['longitude'])
                    user_location_obtained = True
                    print(f"📍 Location obtained via backup IP service: {data.get('city', 'Unknown')}, {data.get('region', 'Unknown')} ({lat}, {lon})")
                    return True
        except Exception as e:
            print(f"Backup IP geolocation failed: {e}")
            
        # If both methods fail, keep default coordinates
        print(f"⚠️ Could not obtain user location, using default: Pune ({lat}, {lon})")
        return False
        
    except Exception as e:
        print(f"Location detection error: {e}")
        return False

# Try to get user location on startup
get_user_location()

# Import translation libraries (you'll need to install these)
try:
    from deep_translator import GoogleTranslator
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    DEEP_TRANSLATOR_AVAILABLE = True
    print("✓ Deep-translator loaded successfully")
except ImportError:
    print("⚠️ Deep-translator not found. Install with: pip install deep-translator langdetect")
    DEEP_TRANSLATOR_AVAILABLE = False

# Backup: Try googletrans if available
try:
    from googletrans import Translator 
    GOOGLETRANS_AVAILABLE = True
    print("✓ Googletrans loaded as backup")
except ImportError:
    print("⚠️ Googletrans not available (this is fine, using deep-translator)")
    GOOGLETRANS_AVAILABLE = False

TRANSLATION_AVAILABLE = DEEP_TRANSLATOR_AVAILABLE or GOOGLETRANS_AVAILABLE

@dataclass
class MultilingualQuery:
    original_query: str
    detected_language: str
    english_translations: List[str]
    selected_translation: str
    selection_reason: str
    document_scores: List[float]

class MultilingualProcessor:
    def __init__(self):
        self.primary_translator = None
        self.backup_translator = None
        
        # Try deep-translator first (more stable)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                # Test with a simple translation
                test_translator = GoogleTranslator(source='en', target='hi')
                test_result = test_translator.translate("hello")
                if test_result:
                    self.primary_translator = 'deep_translator'
                    print("✓ Primary translator (deep-translator) initialized successfully")
                else:
                    print("⚠️ Deep-translator test failed")
            except Exception as e:
                print(f"⚠️ Deep-translator initialization failed: {e}")
        
        # Try googletrans as backup
        if GOOGLETRANS_AVAILABLE and not self.primary_translator:
            try:
                self.backup_translator = Translator()
                # Test the translator
                test = self.backup_translator.translate("hello", src='en', dest='hi')
                if hasattr(test, 'text'):
                    self.backup_translator_type = 'googletrans'
                    print("✓ Backup translator (googletrans) initialized successfully")
                else:
                    print("⚠️ Googletrans test failed")
                    self.backup_translator = None
            except Exception as e:
                print(f"⚠️ Googletrans initialization failed: {e}")
                self.backup_translator = None
                
        self.supported_languages = {
            'hi': 'Hindi',
            'bn': 'Bengali',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ta': 'Tamil',
            'ur': 'Urdu',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'or': 'Odia',
            'as': 'Assamese',
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic'
        }

    def detect_language(self, query: str) -> str:
        """Detect the language of the input query"""
        if not TRANSLATION_AVAILABLE:
            return 'en'  # Default to English if libraries not available
        
        try:
            detected = detect(query)
            return detected if detected in self.supported_languages else 'en'
        except:
            # Fallback: check for non-ASCII characters
            if any(ord(char) > 127 for char in query):
                # Contains non-ASCII, likely not English
                return 'hi'  # Default to Hindi for Indian languages
            return 'en'

    def _translate_with_fallback(self, text: str, src_lang: str, dest_lang: str) -> str:
        """Translate text with multiple fallback methods"""
        
        # Method 1: Deep-translator (primary)
        if self.primary_translator == 'deep_translator':
            try:
                translator = GoogleTranslator(source=src_lang, target=dest_lang)
                result = translator.translate(text)
                if result and result.strip() and result != text:
                    return result
            except Exception as e:
                print(f"Deep-translator failed: {e}")
        
        # Method 2: Googletrans (backup)
        if self.backup_translator and hasattr(self, 'backup_translator_type'):
            try:
                result = self.backup_translator.translate(text, src=src_lang, dest=dest_lang)
                if hasattr(result, 'text') and result.text and result.text != text:
                    return result.text
            except Exception as e:
                print(f"Googletrans backup failed: {e}")
        
        # Method 3: Simple word mapping for common agricultural terms (Hindi to English)
        if src_lang == 'hi' and dest_lang == 'en':
            word_mappings = {
                'चावल': 'rice',
                'भूसी': 'husk',
                'सेल्यूलोज': 'cellulose',
                'लिग्निन': 'lignin',
                'मात्रा': 'content',
                'क्या': 'what',
                'है': 'is',
                'की': 'of',
                'में': 'in',
                'और': 'and',
                'अरंडी': 'castor',
                'खेती': 'cultivation',
                'तापमान': 'temperature',
                'आवश्यकता': 'required',
                'होती': 'is',
                'के': 'for',
                'लिए': 'for',
                'किस': 'what',
                'उप-उत्पाद': 'byproduct',
                'सबसे': 'most',
                'मूल्यवान': 'valuable',
                'का': 'of',
                'बताइए': 'explain',
                'बारे': 'about'
            }
            
            translated_words = []
            for word in text.split():
                # Remove punctuation for matching
                clean_word = word.strip('।?.,!;')
                translated = word_mappings.get(clean_word, clean_word)
                translated_words.append(translated)
            
            mapped_result = ' '.join(translated_words)
            if mapped_result != text:  # Only return if some translation occurred
                return mapped_result
        
        # Method 4: English to Hindi word mapping
        elif src_lang == 'en' and dest_lang == 'hi':
            reverse_mappings = {
                'rice': 'चावल',
                'husk': 'भूसी', 
                'cellulose': 'सेल्यूलोज',
                'lignin': 'लिग्निन',
                'content': 'मात्रा',
                'what': 'क्या',
                'is': 'है',
                'of': 'की',
                'in': 'में',
                'and': 'और',
                'castor': 'अरंडी',
                'cultivation': 'खेती',
                'temperature': 'तापमान',
                'required': 'आवश्यकता',
                'for': 'के लिए',
                'byproduct': 'उप-उत्पाद',
                'most': 'सबसे',
                'valuable': 'मूल्यवान'
            }
            
            translated_words = []
            for word in text.lower().split():
                clean_word = word.strip('.,!?;')
                translated = reverse_mappings.get(clean_word, word)
                translated_words.append(translated)
            
            mapped_result = ' '.join(translated_words)
            if any(hindi_word in mapped_result for hindi_word in reverse_mappings.values()):
                return mapped_result
        
        # Method 5: Return original text if all methods fail
        print(f"All translation methods failed for: {text}")
        return text
    def generate_english_translations(self, query: str, source_lang: str) -> List[str]:
        """Generate top 3 most probable English translations"""
        if source_lang == 'en':
            return [query]  # Return original if already English

        translations = []
        
        try:
            # Direct translation
            direct_translation = self._translate_with_fallback(query, source_lang, 'en')
            if direct_translation != query:  # Only add if translation actually worked
                translations.append(direct_translation)

            # Generate variations by modifying query structure
            variations = self._generate_query_variations(query, source_lang)
            
            for variation in variations[:2]:  # Get up to 2 more variations
                try:
                    translated = self._translate_with_fallback(variation, source_lang, 'en')
                    if translated not in translations and translated != variation:
                        translations.append(translated)
                except Exception as var_e:
                    print(f"Variation translation error: {var_e}")
                    continue

            # If no translations worked, try word-by-word approach
            if not translations:
                word_translation = self._translate_with_fallback(query, source_lang, 'en')
                translations.append(word_translation)

            # Ensure we have at least one translation
            if not translations:
                translations = [query]  # Fallback to original

            # Pad to 3 translations if needed
            while len(translations) < 3:
                if len(translations) == 1:
                    # Add a more formal version
                    formal = f"What is {translations[0]}?" if not any(w in translations[0].lower() for w in ['what', 'how', 'why']) else translations[0]
                    translations.append(formal)
                else:
                    translations.append(translations[0])  # Duplicate the first one

        except Exception as e:
            print(f"Translation error: {e}")
            translations = [query]  # Fallback: return original query

        return translations[:3]

    def _generate_query_variations(self, query: str, source_lang: str) -> List[str]:
        """Generate slight variations of the query for different translation contexts"""
        variations = []
        
        # Add context words to help with ambiguous translations
        if source_lang in ['hi', 'bn', 'te', 'mr', 'ta', 'ur', 'gu', 'kn', 'ml', 'pa']:
            # For Indian languages, add agricultural context
            if source_lang == 'hi':
                variations.append(f"कृषि के बारे में: {query}")
                variations.append(f"खेती: {query}")
            else:
                variations.append(f"agriculture: {query}")
                variations.append(f"farming: {query}")
        else:
            # For other languages, add English context words
            variations.append(f"agriculture: {query}")
            variations.append(f"farming: {query}")
        
        # Add question format variation - try to add in source language
        question_words = {
            'hi': ['क्या', 'कैसे', 'कब', 'कहाँ', 'कौन'],
            'en': ['what', 'how', 'why', 'when', 'where', 'which']
        }
        
        lang_question_words = question_words.get(source_lang, question_words['en'])
        if not any(word in query.lower() for word in lang_question_words):
            if source_lang == 'hi':
                variations.append(f"{query} क्या है?")
            else:
                variations.append(f"What is {query}?")
            
        # Add explanation request
        if source_lang == 'hi':
            variations.append(f"{query} बताइए")
        else:
            variations.append(f"Please explain {query}")
        
        return variations

    def translate_to_original_language(self, text: str, target_lang: str) -> str:
        """Translate English response back to original language"""
        if target_lang == 'en':
            return text

        try:
            translated = self._translate_with_fallback(text, 'en', target_lang)
            return translated
        except Exception as e:
            print(f"Back-translation error: {e}")
            return text  # Return English if translation fails

    async def select_best_translation(self, translations: List[str], search_engines) -> tuple[str, str, List[float]]:
        """Select the translation that has the most relevant data in RAG documents"""
        if len(translations) == 1:
            return translations[0], "Only one translation available", [1.0]

        scores = []
        
        for translation in translations:
            try:
                # Quick search to get document relevance scores
                bm25_results = await search_engines['bm25'].search(translation, top_k=5)
                qdrant_results = await search_engines['qdrant'].search(translation, top_k=5)
                
                # Calculate combined relevance score
                bm25_score = sum([doc.get('score', 0) for doc in bm25_results]) / len(bm25_results) if bm25_results else 0
                qdrant_score = sum([doc.get('score', 0) for doc in qdrant_results]) / len(qdrant_results) if qdrant_results else 0
                
                combined_score = (bm25_score + qdrant_score) / 2
                scores.append(combined_score)
                
            except Exception as e:
                print(f"Error scoring translation '{translation}': {e}")
                scores.append(0.0)

        # Select translation with highest score
        best_index = scores.index(max(scores))
        best_translation = translations[best_index]
        
        selection_reason = f"Selected translation {best_index + 1}/3 with relevance score {max(scores):.3f}"
        
        return best_translation, selection_reason, scores

# Enhanced ArchitecturalRAGPipeline with multilingual support
class MultilingualArchitecturalRAGPipeline(ArchitecturalRAGPipeline):
    def __init__(self, bm25_engine, qdrant_engine):
        super().__init__(bm25_engine, qdrant_engine)
        self.multilingual_processor = MultilingualProcessor()
        self.weather_api = WeatherAPI()  # ADD THIS LINE

    async def process_multilingual_query(self, query: str) -> MultilingualQuery:
        """Process query through multilingual pipeline"""
        print(f"🌐 Processing multilingual query: {query}")
        
        # Step 1: Detect language
        detected_lang = self.multilingual_processor.detect_language(query)
        lang_name = self.multilingual_processor.supported_languages.get(detected_lang, detected_lang)
        print(f"🔍 Detected language: {lang_name} ({detected_lang})")
        
        # Step 2: Generate English translations
        translations = self.multilingual_processor.generate_english_translations(query, detected_lang)
        print(f"📝 Generated {len(translations)} English translations:")
        for i, trans in enumerate(translations, 1):
            print(f"   {i}. {trans}")
        
        # Step 3: Select best translation based on document relevance
        search_engines = {'bm25': self.bm25_engine, 'qdrant': self.qdrant_engine}
        best_translation, selection_reason, scores = await self.multilingual_processor.select_best_translation(
            translations, search_engines
        )
        print(f"✅ {selection_reason}")
        print(f"🎯 Selected: {best_translation}")
        
        return MultilingualQuery(
            original_query=query,
            detected_language=detected_lang,
            english_translations=translations,
            selected_translation=best_translation,
            selection_reason=selection_reason,
            document_scores=scores
        )


    async def invoke(self, query: str) -> Dict[str, Any]:
        """Enhanced invoke method with multilingual support and weather context"""
        # Process multilingual aspects
        multilingual_query = await self.process_multilingual_query(query)
        english_query = multilingual_query.selected_translation
        
        # ALWAYS fetch weather context (let LLM decide if it's relevant)
        global lat, lon
        weather_context = await self.weather_api.get_current_weather_context(latitude=lat, longitude=lon)
        
        # Process with existing RAG architecture
        result = await super().invoke(english_query)
        
        # Get the context that was used in RAG
        context_text = result.get('context_used', 'No relevant documents found in database.')
        
        # ENHANCE THE LLM PROMPT with both contexts
        enhanced_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to both document knowledge and current weather data. "
                    "Use the provided contexts appropriately based on the user's query. "
                    "If the query is about current/future weather, use the weather API data. "
                    "If it's about historical weather or non-weather topics, rely on document context. "
                    "If the query requires both, combine information intelligently. "
                    "Always be clear about data sources and limitations."
                )
            },
            {
                "role": "user",
                "content": f"""
    WEATHER CONTEXT:
    {weather_context}

    DOCUMENT CONTEXT:
    {context_text}

    USER QUERY: {english_query}

    Please provide a comprehensive answer using the most relevant context(s) above.
    """
            }
        ]
        
        # Import and use LLM
        from engine import llm
        enhanced_answer = await llm.generate(enhanced_messages)
        
        # Translate back if needed
        if multilingual_query.detected_language != 'en':
            translated_answer = self.multilingual_processor.translate_to_original_language(
                enhanced_answer, multilingual_query.detected_language
            )
            result['translated_answer'] = translated_answer
            result['original_answer_english'] = enhanced_answer
            result['answer'] = translated_answer
        else:
            result['answer'] = enhanced_answer
        
        # Add weather context info to result
        result['weather_context_provided'] = True
        result['weather_context'] = weather_context
        
        # Add multilingual information
        result['multilingual_info'] = {
            'original_query': multilingual_query.original_query,
            'detected_language': multilingual_query.detected_language,
            'language_name': self.multilingual_processor.supported_languages.get(multilingual_query.detected_language, multilingual_query.detected_language),
            'english_translations': multilingual_query.english_translations,
            'selected_translation': multilingual_query.selected_translation,
            'selection_reason': multilingual_query.selection_reason,
            'document_scores': multilingual_query.document_scores
        }
        
        return result

def print_enhanced_analysis(result: dict):
    """Print enhanced analysis including weather context usage"""
    print("\n" + "="*80)
    print("🧠 LLM DECISION ANALYSIS")
    print("="*80)
    
    if result.get("weather_context_provided"):
        print("✅ Weather context was provided to LLM")
        print("🤖 LLM decided whether to use weather data based on query relevance")
        
        # Try to analyze if weather data was actually used in response
        answer = result.get("answer", "").lower()
        weather_indicators = ["temperature", "weather", "forecast", "rain", "°c", "degrees", "मौसम", "तापमान"]
        used_weather = any(indicator in answer for indicator in weather_indicators)
        
        print(f"📊 Weather data likely used in response: {'Yes' if used_weather else 'No'}")
        
        if result.get("context_used") and result.get("context_used") != "No relevant documents found in database.":
            print("📚 Document context was also available")
            print("🔄 LLM combined both contexts intelligently")
        else:
            print("📚 No relevant document context found")
            
    print("="*80)

def preprocess_query(q: str) -> str:
    cleaned = q.strip()
    if any(word in cleaned.lower() for word in ["explain", "what is", "define"]):
        cleaned = f"[DEFINITION] {cleaned}"
    elif any(word in cleaned.lower() for word in ["how to", "steps", "process"]):
        cleaned = f"[PROCEDURAL] {cleaned}"
    elif any(word in cleaned.lower() for word in ["why", "because", "reason"]):
        cleaned = f"[CAUSAL] {cleaned}"
    return cleaned

def printmd(text: str):
    print(text)

def print_multilingual_analysis(result: dict):
    """Print multilingual processing details"""
    multilingual_info = result.get("multilingual_info", {})
    if not multilingual_info:
        return
        
    print("\n" + "="*80)
    print("🌐 MULTILINGUAL PROCESSING ANALYSIS")
    print("="*80)
    
    print(f"📝 Original Query: {multilingual_info.get('original_query', 'N/A')}")
    print(f"🔍 Detected Language: {multilingual_info.get('language_name', 'N/A')} ({multilingual_info.get('detected_language', 'N/A')})")
    
    translations = multilingual_info.get('english_translations', [])
    scores = multilingual_info.get('document_scores', [])
    
    print(f"\n📚 English Translation Analysis:")
    for i, (translation, score) in enumerate(zip(translations, scores), 1):
        marker = "🎯 SELECTED" if translation == multilingual_info.get('selected_translation') else "  "
        print(f"   {i}. {translation}")
        print(f"      {marker} Relevance Score: {score:.3f}")
    
    print(f"\n✅ Selection Reason: {multilingual_info.get('selection_reason', 'N/A')}")
    
    if 'original_answer_english' in result:
        print(f"\n🔄 Answer Translation: English → {multilingual_info.get('language_name', 'Original Language')}")
    
    print("="*80)

def print_architectural_analysis(result: dict):
    print("\n" + "="*80)
    print("🏗️  ARCHITECTURAL ANALYSIS")
    print("="*80)

    query_info = result.get("query_analysis", {})
    print(f"📝 Original Query: {query_info.get('original_query', 'N/A')}")
    print(f"🌐 Language Detected: {query_info.get('language', 'N/A')}")
    print(f"🎯 Intent: {query_info.get('intent', 'N/A')}")
    print(f"⚡ Processing Mode: {query_info.get('processing_mode', 'N/A')}")

    reasoning_info = result.get("reasoning", {})
    print(f"\n🧠 REASONING ENGINE:")
    print(f"   Conclusion: {reasoning_info.get('conclusion', 'N/A')}")
    print(f"   Confidence: {reasoning_info.get('confidence', 0):.2f}")
    if reasoning_info.get('reasoning_path'):
        for i, step in enumerate(reasoning_info['reasoning_path'], 1):
            print(f"     {i}. {step}")

    info_processing = result.get("information_processing", {})
    print(f"\n📊 INFORMATION PROCESSING:")
    print(f"   Total Information Packets: {info_processing.get('total_packets', 0)}")
    print(f"   High Value: {info_processing.get('high_value_count', 0)}")
    print(f"   Medium Value: {info_processing.get('medium_value_count', 0)}")
    print(f"   Low Value: {info_processing.get('low_value_count', 0)}")

    economics = result.get("economics", {})
    processing_cost = economics.get("processing_cost", {})
    print(f"\n💰 MARKET/STORAGE/ECONOMY:")
    print(f"   Processing Mode: {processing_cost.get('mode', 'N/A')}")
    print(f"   Base Cost: ${processing_cost.get('base_cost', 0):.4f}")
    print(f"   Total Cost: ${processing_cost.get('total_cost', 0):.4f}")
    print(f"   Efficiency Score: {processing_cost.get('efficiency_score', 0):.2f}")
    print(f"   Combination Strategy: {economics.get('combination_strategy', 'N/A')}")

    performance = result.get("performance", {})
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Processing Time: {performance.get('processing_time', 0):.2f}s")
    print(f"   Source Documents: {performance.get('source_documents', 0)}")
    print("="*80)

async def demonstrate_offline_processing():
    print("\n🔄 DEMONSTRATING OFFLINE PROCESSING")
    print("="*50)
    complex_query = "Explain the detailed step-by-step process of how machine learning algorithms work and why they are effective for pattern recognition in large datasets"
    tokenized_query = await query_engine.process_query(complex_query)
    print(f"Query: {complex_query}")
    print(f"Processing Mode: {tokenized_query.processing_mode.value}")
    print(f"Language: {tokenized_query.language}")
    print(f"Intent: {tokenized_query.intent}")
    print(f"Filtered Tokens: {tokenized_query.filtered_tokens[:10]}...")
    return tokenized_query

async def demonstrate_online_processing():
    print("\n⚡ DEMONSTRATING ONLINE PROCESSING")
    print("="*50)
    simple_query = "What is Python?"
    tokenized_query = await query_engine.process_query(simple_query)
    print(f"Query: {simple_query}")
    print(f"Processing Mode: {tokenized_query.processing_mode.value}")
    print(f"Language: {tokenized_query.language}")
    print(f"Intent: {tokenized_query.intent}")
    print(f"Filtered Tokens: {tokenized_query.filtered_tokens}")
    return tokenized_query

async def test_architectural_components():
    print("\n🧪 TESTING ARCHITECTURAL COMPONENTS")
    print("="*50)
    sample_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "uri": "test://python-info",
            "text": "Python is a high-level programming language known for its simplicity and readability.",
            "score": 0.9
        },
        {
            "content": "Machine learning is a subset of AI that enables computers to learn from data.",
            "uri": "test://ml-info",
            "text": "Machine learning is a subset of AI that enables computers to learn from data.",
            "score": 0.8
        }
    ]
    test_query = "What is Python programming?"
    tokenized = await query_engine.process_query(test_query)
    print(f"✓ Query Engine: Processed '{test_query}' -> Intent: {tokenized.intent}")

    reasoning = await reasoning_engine.reason(tokenized, sample_docs)
    print(f"✓ Reasoning Engine: Confidence {reasoning.confidence:.2f}")

    info_packets = await info_filter.filter_information(sample_docs, tokenized)
    print(f"✓ Information Filter: Created {len(info_packets)} packets")

    tokenization_data = await tokenization_db.tokenize_and_embed(info_packets)
    print(f"✓ Tokenization DB: Processed {len(tokenization_data['tokens'])} token sets")

    optimized = await market_economy.optimize_storage(tokenization_data, tokenized)
    cost = await market_economy.calculate_processing_cost(tokenized, len(sample_docs))
    print(f"✓ Market Economy: Optimized storage, Cost: ${cost['total_cost']:.4f}")

    combined = await combining_engine.combine_information(optimized, reasoning, tokenized)
    print(f"✓ Combining Engine: Used {combined['strategy_used']} strategy")
    print("All architectural components tested successfully! ✨")

async def test_multilingual_queries():
    """Test multilingual capabilities with sample queries"""
    print("\n🌐 TESTING MULTILINGUAL CAPABILITIES")
    print("="*50)
    
    # Sample multilingual queries for agriculture
    multilingual_test_queries = [
        "चावल की भूसी में सेल्यूलोज और लिग्निन की मात्रा क्या है?",  # Hindi
        "What is the cellulose content in rice husk?",  # English
        "ধানের তুষে সেলুলোজ কত থাকে?",  # Bengali (if supported)
        "Quel est le contenu en cellulose de la balle de riz?",  # French
    ]
    
    for query in multilingual_test_queries[:2]:  # Test first 2 for demo
        print(f"\n🔍 Testing: {query}")
        print("-" * 40)
        
        processor = MultilingualProcessor()
        detected_lang = processor.detect_language(query)
        print(f"Detected: {processor.supported_languages.get(detected_lang, detected_lang)}")
        
        translations = processor.generate_english_translations(query, detected_lang)
        print(f"Translations: {translations}")

async def main():
    print("🌟 ENHANCED MULTILINGUAL WEATHER-INSPIRED RAG ARCHITECTURE")
    print("Using AGRICULTURE collection from Qdrant with Multilingual Support")
    print("="*80)

    print("🔧 Initializing search engines...")

    # Connect to existing Qdrant collection AGRICULTURE (no fit)
    qdrant = QdrantSearchEngine(collection_name="AGRICULTURE")

    # If BM25 is needed, fetch stored docs from Qdrant
    try:
        stored_chunks = qdrant.fetch_all_documents()
        print("✓ Fetched stored documents from Qdrant")
    except AttributeError:
        stored_chunks = qdrant.fetch_all_documents()

    bm25 = await BM25SearchEngine().fit(stored_chunks)

    print("✓ BM25 and Qdrant engines ready")

    # Use enhanced multilingual pipeline
    multilingual_architectural_rag = MultilingualArchitecturalRAGPipeline(bm25, qdrant)
    print("✓ Multilingual Architectural RAG pipeline initialized")

    await test_architectural_components()
    await demonstrate_offline_processing()
    await demonstrate_online_processing()
    await test_multilingual_queries()

    print("\n🎯 TESTING MULTILINGUAL QUERIES WITH FULL ARCHITECTURE")
    print("="*60)
    
    # Test queries in multiple languages
    test_queries = [
        # Current weather queries
        "What is the price of rice in Pune today?",
        "What is the weather today in Pune?",
        "What seed variety suits this unpredictable weather?",
        "Will next week's temperature drop kill my rice yield?",
        "Can I afford to wait for the market to improve?",
        "Where can I get affordable credit, and will any state/central government policy help me with finances?",
        "Will it rain tomorrow?",
        "What will be the temperature next week?",
        
        # Historical weather queries (should use documents if available)
        "What was the weather like last year?",
        "पिछले साल बारिश कैसी थी?",
        
        # Agriculture + current weather
        "Should I plant rice considering current weather conditions?",
        "वर्तमान मौसम को देखते हुए क्या अरंडी की खेती करनी चाहिए?",
        
        # Pure agriculture (existing queries)
        "What is the cellulose content in rice husk?",
        "चावल की भूसी में सेल्यूलोज की मात्रा क्या है?",
        "What is the most valuable rice by product?",
        "What is temperature range required for castor cultivation?",
        "अरंडी की खेती के लिए किस तापमान की आवश्यकता होती है?",
        
        # Mixed queries
        "How does current temperature affect castor cultivation?",
        "Present weather impact on rice farming"
    ]

    total_processing_time = 0
    total_cost = 0

    for i, raw_query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}/{len(test_queries)}: {raw_query}")
        print("-" * 80)
        start_time = time.time()
        preprocessed = preprocess_query(raw_query)

        try:
            result = await multilingual_architectural_rag.invoke(preprocessed)
            print("💬 ANSWER:")
            printmd(result["answer"])
            with open('answers_rag3.txt', 'a', encoding='utf-8') as file:
                file.write(f'{result["answer"]}\n')

            
            # Print multilingual analysis first
            print_enhanced_analysis(result) 
            print_multilingual_analysis(result)
            
            # Then architectural analysis
            print_architectural_analysis(result)
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            economics = result.get("economics", {})
            processing_cost = economics.get("processing_cost", {})
            total_cost += processing_cost.get("total_cost", 0)
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue

        print("\n" + "="*80)

    print(f"\n📊 FINAL MULTILINGUAL ARCHITECTURAL SUMMARY")
    print("="*60)
    print(f"Total Queries Processed: {len(test_queries)}")
    print(f"Total Processing Time: {total_processing_time:.2f}s")
    print(f"Average Time per Query: {total_processing_time/len(test_queries):.2f}s")
    print(f"Total Economic Cost: ${total_cost:.4f}")
    print(f"Average Cost per Query: ${total_cost/len(test_queries):.4f}")
    print("\n🎉 MULTILINGUAL ARCHITECTURAL RAG PIPELINE COMPLETE!")
    print("\n🌐 Supported Languages: Hindi, Bengali, English, Spanish, French, German, and more!")

if __name__ == "__main__":
    asyncio.run(main())