from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "chroma_db"
MODEL_NAME = "llama3.2"

app = FastAPI()

# Load the vector store and LLM once when the server starts
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
llm = ChatOllama(model=MODEL_NAME)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    """Answer a question using retrieved context from the vector store."""
    results = vectordb.similarity_search(request.question, k=5)

    context = "\n\n".join([doc.page_content for doc in results])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in results]))

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {request.question}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content, "sources": sources}

# Serve the frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")