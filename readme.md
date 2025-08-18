Integration Notes

Qdrant Setup (Lines 663–664):
Add your Qdrant URL and Qdrant API key by creating a collection and uploading the relevant documents.

Note: We are keeping our API live until 28th August 2025 so you can verify the results.

LLM Setup (Line 765):
Insert your Gemini API key. You can also choose from different Gemini models. Higher-capacity models generally provide better accuracy, but in our case gemini-flash-1.5 worked very well.

Offline Option with Ollama:
If you prefer running offline, you can use Ollama by setting:

llm = OllamaLLM("model")

We tested this with the default Mistral model, which was both fast and accurate.
