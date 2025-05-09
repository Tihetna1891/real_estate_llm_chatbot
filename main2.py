from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding

# === Load documents from data folder ===
documents = SimpleDirectoryReader("data").load_data()

# === Use smaller local LLM and embedding ===
# Settings.llm = Ollama(model="mistral", request_timeout=60)
Settings.llm = Ollama(model="llama2:7b", request_timeout=60)
# Settings.llm = Ollama(model="tinyllama", request_timeout=60)

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Create index and query engine ===
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# === FastAPI App ===
app = FastAPI()

# Allow CORS (required for Power BI or browser testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Chat endpoint ===
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        if not query:
            return {"response": "Please provide a valid query."}

        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        print("Internal error:", str(e))
        return {"response": f"Internal server error: {str(e)}"}
