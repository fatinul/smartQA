import os
import shutil # Import for file and directory removal
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from typing import Dict, Any, List

class RAGPipeline:
    """
    A class to encapsulate the RAG ingestion, index management, and stateful 
    multi-turn chat logic using LlamaIndex and Ollama.
    The chat history is now persisted to a JSON file and can be reset.
    The knowledge index can also be rebuilt.
    """
    def __init__(
        self,
        llm_model: str = "gemma3:4b",
        embed_model_name: str = "nomic-embed-text",
        persist_dir: str = "./storage",
        knowledge_dir: str = "knowledge",
        similarity_top_k: int = 3,
        request_timeout: float = 120.0
    ):
        """
        Initializes the RAG Pipeline configuration and models.
        """
        self.llm_model = llm_model
        self.embed_model_name = embed_model_name
        self.persist_dir = persist_dir
        self.knowledge_dir = knowledge_dir
        self.similarity_top_k = similarity_top_k
        self.request_timeout = request_timeout
        self.chat_history_path = os.path.join(self.persist_dir, "chat_history.json")

        self.ollama_llm: OllamaLLM = self._initialize_llm()
        self.ollama_embed_model: OllamaEmbedding = self._initialize_embedding_model()
        
        # Initialize the Index and Chat Engine upon start
        self.chat_engine = self._initialize_chat_engine(self._load_or_build_index())

    def _initialize_llm(self) -> OllamaLLM:
        """Initializes the Ollama LLM client."""
        print(f"Initializing LLM: {self.llm_model}")
        return OllamaLLM(
            model=self.llm_model,
            request_timeout=self.request_timeout
        )

    def _initialize_embedding_model(self) -> OllamaEmbedding:
        """Initializes the Ollama Embedding client."""
        print(f"Initializing Embedding Model: {self.embed_model_name}")
        return OllamaEmbedding(
            model_name=self.embed_model_name,
            request_timeout=self.request_timeout
        )

    def _initialize_chat_engine(self, index: VectorStoreIndex) -> CondensePlusContextChatEngine:
        """
        Sets up or re-initializes the persistent chat memory and the chat engine.
        """
        # 1. Load/Create the SimpleChatStore
        if os.path.exists(self.chat_history_path):
            print(f"Loading existing chat history from {self.chat_history_path}...")
            chat_store = SimpleChatStore.from_persist_path(self.chat_history_path)
        else:
            print("Starting new chat history.")
            chat_store = SimpleChatStore()
        
        # 2. Wrap the chat store in a ChatMemoryBuffer
        self.memory = ChatMemoryBuffer.from_defaults(
            chat_store=chat_store, 
            chat_store_key="default_conversation"
        )
        
        # 3. Initialize the Chat Engine
        return index.as_chat_engine(
            llm=self.ollama_llm,
            similarity_top_k=self.similarity_top_k,
            chat_mode="condense_plus_context",
            memory=self.memory
        )


    def _load_or_build_index(self) -> VectorStoreIndex:
        """
        Checks for an existing index. Loads it if found, or builds and persists 
        a new one otherwise.
        """
        # Ensure the persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Check if the index storage files exist
        index_files_exist = any(
            os.path.exists(os.path.join(self.persist_dir, f)) 
            for f in ["docstore.json", "index_store.json", "graph_store.json"]
        )
        
        if not index_files_exist:
            print(f"Starting Ingestion: Building NEW Index from {self.knowledge_dir}/...")
            
            # Load documents
            loader = SimpleDirectoryReader(
                input_dir=self.knowledge_dir,
                required_exts=[".md"],
                recursive=True
            )
            documents = loader.load_data()

            # Create the Vector Store Index
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.ollama_embed_model,
            )

            # Persist the index to disk
            index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"Ingestion complete. Index saved to {self.persist_dir}/")
            return index

        else:
            print(f"Loading existing index from {self.persist_dir}/...")
            
            # Load the index from the saved directory
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            index = load_index_from_storage(
                storage_context,
                embed_model=self.ollama_embed_model 
            )
            print("Index loaded successfully.")
            return index

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Queries the RAG Chat engine with a question, maintaining conversation history,
        and returns the response and sources.
        """
        try:
            response = self.chat_engine.chat(question)
            
            sources = []
            for doc in response.source_nodes:
                sources.append({
                    "file_path": doc.node.metadata.get('file_path', 'N/A'),
                    "score": f"{doc.score:.4f}",
                    "text_preview": doc.text[:150] + "..."
                })
            
            # Save the updated chat history to the JSON file
            self.memory.chat_store.persist(persist_path=self.chat_history_path)

            return {
                "response": response.response,
                "sources": sources
            }
        except Exception as e:
            return {
                "response": f"An error occurred during query execution: {e}",
                "sources": []
            }
            
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the current conversation history from the persistent memory.
        """
        history = self.memory.chat_store.get_messages("default_conversation")
        
        formatted_history = []
        for msg in history:
            formatted_history.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return formatted_history

    def remove_chat_history(self) -> None:
        """
        Clears the in-memory chat history and deletes the persistent chat history file.
        """
        print("\n--- ACTION: Removing Chat History ---")
        try:
            # 1. Clear the in-memory history
            self.chat_engine.reset()
            # 2. Delete the persistent file
            if os.path.exists(self.chat_history_path):
                os.remove(self.chat_history_path)
                print(f"Successfully deleted chat history file: {self.chat_history_path}")
            else:
                print("Chat history file already absent.")
            
            # 3. Re-initialize memory/chat engine to ensure a clean slate
            self.chat_engine = self._initialize_chat_engine(self._load_or_build_index())

        except Exception as e:
            print(f"Error while removing chat history: {e}")

    def relearn_knowledge(self) -> None:
        """
        Forces a complete re-ingestion and re-indexing of the knowledge base.
        Deletes the entire persistence directory and rebuilds the index.
        """
        print("\n--- ACTION: Forcing Knowledge Relearn/Re-index ---")
        try:
            # 1. Remove the entire persistence directory
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                print(f"Successfully deleted persistence directory: {self.persist_dir}")
            else:
                print("Persistence directory already absent.")
            
            # 2. Rebuild the index (which will create the directory again)
            new_index = self._load_or_build_index()
            
            # 3. Re-initialize the chat engine with the new index
            self.chat_engine = self._initialize_chat_engine(new_index)
            print("Knowledge base successfully rebuilt and chat engine updated.")

        except Exception as e:
            print(f"Error while forcing knowledge relearn: {e}")


# Example of how to use the RAGPipeline class
if __name__ == "__main__":
    # Ensure your 'knowledge' directory exists and contains documents
    # and Ollama is running before executing this block.
    
    # 1. Initialize the pipeline
    rag_client = RAGPipeline()

    print("\n--- DEMO: Initial State & Conversation ---")
    
    # 2. Display existing history (should be empty on first run, or loaded from previous)
    print("\n[STEP 2: CHECK HISTORY]")
    history = rag_client.get_chat_history()
    if history:
        print(f"Found {len(history)} messages from previous session.")
    else:
        print("No prior chat history found.")

    # relearn_knowledge()  # Uncomment to force re-ingestion/re-indexing
    rag_client.relearn_knowledge()
        
    # 3. Start a conversation to generate history
    query_1 = "Where should I go if I want to find lighthouse with a patched lens. Give the place name."
    print(f"\n[STEP 3: TURN 1] USER: {query_1}")
    result_1 = rag_client.chat(query_1)
    print(f"ASSISTANT: {result_1['response']}...")
    
    # 7. Demonstrate the knowledge relearn function (this takes time as it re-embeds)
    # Be careful running this on large knowledge bases.
    # rag_client.relearn_knowledge() 
    # print("\n[STEP 7: KNOWLEDGE RELEARNED]")
    # print("Index was successfully rebuilt. You can now chat with the new index.")