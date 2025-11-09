import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- Configuration ---
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, os.getenv("FAISS_INDEX_DIR", "faiss_index"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # The model used for FAISS index creation

# --- Document Loading Utility (Placeholder for BM25 Indexing) ---

# NOTE: In your final project structure, this function should ideally be imported 
# from your index_builder.py or a similar data utility file to avoid duplication.
# It is included here for a complete, runnable example.

def load_and_split_documents() -> List[Document]:
    """
    Simulates the document loading and chunking process from index_builder.py.
    This is necessary to build the BM25 index.
    """
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    NEPHROLOGY_REF_PATH = os.path.join(DATA_DIR, "nephrology_reference.pdf")

    try:
        # Attempt to load the actual PDF (requires the file to exist)
        loader = PyPDFLoader(NEPHROLOGY_REF_PATH)
        documents = loader.load()
    except Exception:
        # Fallback to dummy text if the file is not found, for demonstration purposes
        print(f"Warning: Could not load {NEPHROLOGY_REF_PATH}. Using dummy text for BM25 index.")
        documents = [
            Document(page_content="Chronic Kidney Disease (CKD) Stage 3B is defined by a Glomerular Filtration Rate (GFR) between 30 and 44 mL/min/1.73 m². Dietary restrictions are crucial and typically include low-sodium, low-potassium, and low-phosphorus diets. Protein intake should be monitored and often restricted to 0.6-0.8 g/kg/day to reduce the burden on the kidneys. Patients must avoid NSAIDs and certain contrast dyes. Regular monitoring of blood pressure and blood sugar is essential for managing the progression of CKD. The primary goal is to slow the decline of kidney function and manage complications like anemia and bone disease. Fluid intake may also need to be restricted in later stages.", metadata={"source": "nephrology_reference.pdf", "page": 1}),
            Document(page_content="The recommended protein intake for CKD Stage 3B is 0.6-0.8 grams per kilogram of body weight per day. High-potassium foods like bananas, oranges, and potatoes should be limited. Instead, patients can consume low-potassium fruits and vegetables such as apples, berries, and green beans. Phosphorous is found in dairy products, nuts, and dark sodas, and these should be consumed sparingly or with phosphate binders. Anemia is a common complication and is treated with iron supplements and erythropoiesis-stimulating agents (ESAs).", metadata={"source": "nephrology_reference.pdf", "page": 2}),
        ]

    # 2. Chunking (Apply the same chunking logic used for FAISS)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- RAG Retriever Logic ---

class RAGRetriever:
    """
    Handles loading the FAISS index and performing hybrid document retrieval.
    """
    def __init__(self):
        print("Initializing RAGRetriever...")
        
        # 1. Initialize Embeddings and Load FAISS Vectorstore (Dense Retriever)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = self._load_vectorstore()
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 2. Initialize BM25 Retriever (Sparse Retriever)
        # Load all document chunks to build the BM25 index
        all_chunks = load_and_split_documents()
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 5 # Set the number of documents to retrieve
        
        # 3. Combine retrievers into an EnsembleRetriever (Hybrid)
        # The weights determine the importance of each retriever's results.
        # Adjust weights as needed (e.g., [0.3, 0.7] to favor semantic search)
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], 
            weights=[0.5, 0.5]
        )
        
        print("RAGRetriever initialized with Hybrid (BM25 + FAISS) Retrieval.")

    def _load_vectorstore(self) -> FAISS:
        """Loads the FAISS index from the disk."""
        try:
            # Load the index and the docstore
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True # Required for loading FAISS index
            )
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS index from {FAISS_INDEX_DIR}: {e}")
            print("Please run setup_data.py to build the index first.")
            # Return a dummy vectorstore to prevent application crash
            return FAISS.from_texts(["Error: Index not loaded."], self.embeddings)

    def retrieve(self, query: str) -> List[Document]:
        """
        Performs a hybrid search on the vector store and BM25 index.
        
        :param query: The user's question.
        :return: A list of relevant Document objects.
        """
        # The EnsembleRetriever handles the combination and re-ranking of results
        return self.retriever.invoke(query)

# Global instance for easy access
rag_retriever = RAGRetriever()

if __name__ == "__main__":
    # Example usage
    query = "What are the dietary restrictions for a patient with CKD Stage 3B?"
    print(f"Query: {query}")
    
    results = rag_retriever.retrieve(query)
    
    print(f"\nFound {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"--- Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}) ---")
        print(doc.page_content[:200] + "...")
