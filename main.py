from fastapi import FastAPI, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
import redis, hashlib, os, json
import psycopg2
from sentence_transformers import SentenceTransformer
import fitz # PyMuPDF for PDF

app = FastAPI(title="Scalable RAG API")
Instrumentator().instrument(app).expose(app)

redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/ragdb"))
    return conn

@app.get("/health")
def health():
    return {"status": "ok", "cache": redis_client.ping()}

@app.post("/ingest")
async def ingest(file: UploadFile):
    raw = await file.read()
    # Handle PDF vs TXT
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "".join([p.get_text() for p in doc])
    else:
        text = raw.decode('utf-8', errors='ignore')

    # --- CHUNKING ADDED HERE ---
    chunk_size = 300
    overlap = 50
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE TABLE IF NOT EXISTS docs (id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))")
    cur.execute("DELETE FROM docs") # clear old data for test

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        cur.execute("INSERT INTO docs (content, embedding) VALUES (%s, %s)", (chunk, embedding))

    conn.commit()
    cur.close(); conn.close()
    return {"status": "ingested", "chunks": len(chunks)}

@app.post("/chat")
def chat(query: str):
    cache_key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        data["source"] = "redis_cache"
        return data

    q_emb = model.encode(query).tolist()
    conn = get_db()
    cur = conn.cursor()
    # top_k = 2
    cur.execute("SELECT content FROM docs ORDER BY embedding <=> %s::vector LIMIT 2", (q_emb,))
    docs = cur.fetchall()
    cur.close(); conn.close()

    if not docs:
        return {"answer": "No documents ingested yet."}

    context = "\n---\n".join([d[0] for d in docs])

    answer = context[:500] # concise answer

    result = {"answer": answer, "source": "pgvector", "retrieved_docs": len(docs)}
    redis_client.setex(cache_key, 3600, json.dumps(result))
    return result