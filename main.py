from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from utils import extract_pdf_text, chunk_text, create_faiss_index
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from uuid import uuid4
import os

load_dotenv()

app = FastAPI()

# 🔐 Validate API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 🧠 In-memory store (doc_id → {chunks, index})
store = {}

# 📦 Load embedding model on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer("all-MiniLM-L6-v2")


# 📄 Request schema
class QueryRequest(BaseModel):
    query: str
    doc_id: str


# 🔍 Retrieval helper
def retrieve_context(query: str, index, chunks, k: int = 3):
    q_embed = model.encode([query]).astype("float32")
    D, I = index.search(q_embed, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


# 📤 Upload PDF
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    try:
        file_path = f"./{file.filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract + chunk
        text = extract_pdf_text(file_path)
        chunks = chunk_text(text)

        # Create FAISS index
        index, _ = create_faiss_index(chunks)

        # Store with doc_id
        doc_id = str(uuid4())
        store[doc_id] = {
            "chunks": chunks,
            "index": index
        }

        # Cleanup file
        os.remove(file_path)

        return {
            "message": "PDF processed successfully",
            "doc_id": doc_id,
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ❓ Ask (retrieve only)
@app.post("/ask")
async def ask_question(req: QueryRequest):
    data = store.get(req.doc_id)

    if not data:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    relevant = retrieve_context(req.query, data["index"], data["chunks"])

    return {
        "query": req.query,
        "context_used": relevant
    }


# 🤖 Ask AI (RAG)
@app.post("/ask-ai")
async def ask_ai(req: QueryRequest):
    data = store.get(req.doc_id)

    if not data:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    relevant_chunks = retrieve_context(req.query, data["index"], data["chunks"])
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful assistant. Answer ONLY from the given context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{req.query}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
