import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """Load all PDF files from the data folder."""
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"Loaded: {filename}")
    return documents

def split_documents(documents):
    """Split documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """Generate embeddings for each chunk and store them in a Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    vectordb.persist()
    print(f"Vector DB created and saved at: {DB_PATH}")

def main():
    print("Step 1: Loading documents...")
    documents = load_documents()

    if not documents:
        print("No PDFs found in the 'data' folder! Please add PDFs first.")
        return

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)

    print("\nStep 3: Creating vector store...")
    create_vector_store(chunks)

    print("\nDone! You can now run chat.py.")

if __name__ == "__main__":
    main()