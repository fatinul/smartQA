import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding

# Define the directory where the index will be saved
PERSIST_DIR = "./storage"
KNOWLEDGE_DIR = "knowledge"

# --- RAG Step 1: Index Management (Load or Create) ---

# 1. Define the LLM and Embedding Model (always needed for querying)
ollama_llm = OllamaLLM(model="gemma3:4b", request_timeout=120.0)
ollama_embed_model = OllamaEmbedding(model_name="nomic-embed-text", request_timeout=120.0)

# 2. Check if the index already exists
if not os.path.exists(PERSIST_DIR):
    print("Starting RAG Ingestion Pipeline: Building NEW Index...")
    
    # Load documents
    loader = SimpleDirectoryReader(
        input_dir=KNOWLEDGE_DIR,
        required_exts=[".md"],
        recursive=True
    )
    documents = loader.load_data()

    # Create the Vector Store Index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=ollama_embed_model,
    )

    # Persist the index to disk for future use
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Ingestion complete. Index saved to {PERSIST_DIR}/")

else:
    print(f"Loading existing index from {PERSIST_DIR}/...")
    
    # Load the index from the saved directory
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(
        storage_context,
        embed_model=ollama_embed_model # Must pass embed_model to ensure consistency
    )
    print("Index loaded successfully.")


# --- RAG Step 2: Query Engine Construction ---

# 1. Create the Query Engine (This is fast)
query_engine = index.as_query_engine(
    llm=ollama_llm,
    similarity_top_k=3
)


# --- RAG Step 3: Query the Model ---

def ask_rag(question):
    print(f"\n--- Answering Question: {question} ---")
    
    response = query_engine.query(question)
    
    print("\n💡 RAG Answer:")
    print(response.response)

    print("\n📚 Source Nodes:")
    for doc in response.source_nodes:
        print(f" - {doc.node.metadata['file_path']} (Score: {doc.score:.4f})")

# Let's test our knowledge!
ask_rag("I want to plant a tree. Who should I contact?")
ask_rag("I like hot bath, where should I go?")
ask_rag("I want to have a friend that interested in space, who should I friend with?")