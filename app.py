from fastapi import FastAPI
import chromadb
import ollama
from pydantic import BaseModel

class QueryRequest(BaseModel):
    q: str

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(data: QueryRequest):
    q = data.q
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"][0] else ""

    answer = ollama.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer: clearly and concisely:",
    )

    return {"answer": answer["response"]}

@app.post("/add")
def add_document(data: QueryRequest):
    doc = data.q
    doc_id = f"doc_{len(collection.get()["ids"])}"
    collection.add(documents=[doc], ids=[doc_id])
    return {"status": "Document added", "id": doc_id}  