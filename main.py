from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding


# === Step 1: Load documents from "data" folder ===
documents = SimpleDirectoryReader("data").load_data()

# === Step 2: Set Ollama LLM and local embedding model ===
Settings.llm = Ollama(model="llama3", request_timeout=60)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 3: Create vector index ===
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# === Step 4: Set up FastAPI ===
app = FastAPI()

# Optional: Allow CORS (so Streamlit or Power BI can connect later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Step 5: Define chat endpoint ===
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return {"response": "Please provide a valid query."}
    
    response = query_engine.query(query)
    return {"response": str(response)}
