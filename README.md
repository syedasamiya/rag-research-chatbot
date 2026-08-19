# Research Buddy — RAG-based Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a collection of research papers using a local LLM. Built to explore document ingestion, vector search, and full-stack integration with a locally-hosted language model.

## Features

- PDF ingestion pipeline: loads papers, splits them into overlapping chunks, and generates embeddings
- Semantic search using ChromaDB as the vector store
- Local LLM inference via Ollama (no external API costs)
- FastAPI backend serving a `/chat` endpoint
- Custom chat interface (HTML/CSS/JS) with typing indicators and source citations

## Tech Stack

- **Backend:** Python, FastAPI, LangChain
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB
- **LLM:** Ollama (llama3.2, running locally)
- **Frontend:** HTML, CSS, JavaScript

## How It Works

1. PDFs are loaded and split into chunks (`ingest.py`)
2. Each chunk is embedded and stored in a Chroma vector database
3. On a user query, the top-k most relevant chunks are retrieved
4. The retrieved context is passed to the LLM along with the question
5. The LLM generates an answer grounded in the retrieved context, with source attribution

## Setup

\`\`\`bash
# Clone the repo
git clone <your-repo-url>
cd rag-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the Ollama model
ollama pull llama3.2

# Add your PDFs to the data/ folder, then run:
python ingest.py

# Start the server
uvicorn backend:app --reload
\`\`\`

Then open `http://127.0.0.1:8000` in your browser.

## Screenshots

*(Add a screenshot of the chat interface here)*

## Future Improvements

- Support for uploading PDFs directly through the UI
- Streaming responses for faster perceived response time
- Multi-turn conversation memory