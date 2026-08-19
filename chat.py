from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "chroma_db"
MODEL_NAME = "llama3.2"  # the Ollama model you pulled earlier

def load_vector_store():
    """Load the existing Chroma vector store with the same embedding model used during ingestion."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    return vectordb

def get_answer(vectordb, llm, question):
    """Retrieve top relevant chunks and generate an answer using the LLM."""
    results = vectordb.similarity_search(question, k=5)

    # Debug: show what chunks were actually retrieved
    print("\n--- Retrieved Chunks (debug) ---")
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:200]}...\n")
    print("--- End ---\n")

    context = "\n\n".join([doc.page_content for doc in results])
    sources = set([doc.metadata.get("source", "unknown") for doc in results])

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, sources

def main():
    print("Loading vector store...")
    vectordb = load_vector_store()

    print(f"Connecting to Ollama model: {MODEL_NAME}...")
    llm = ChatOllama(model=MODEL_NAME)

    print("\nRAG Chatbot ready! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        answer, sources = get_answer(vectordb, llm, question)
        print(f"\nBot: {answer}")
        print(f"Sources: {', '.join(sources)}\n")

if __name__ == "__main__":
    main()