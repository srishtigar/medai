import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# --- Configuration ---
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RAG_SOURCE_PDF = os.path.join(PROJECT_ROOT, "data", "nephrology_reference.pdf")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, os.getenv("FAISS_INDEX_DIR", "faiss_index"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG Indexing Logic ---

def build_faiss_index():
    """
    Loads the PDF document, splits it into chunks, generates embeddings,
    and saves the resulting FAISS index to disk.
    """
    print(f"--- Starting RAG Index Builder ---")
    print(f"Source Document (PDF): {RAG_SOURCE_PDF}")
    
    # Check if PDF exists
    if not os.path.exists(RAG_SOURCE_PDF):
        print(f"Error: PDF file not found at {RAG_SOURCE_PDF}")
        print(f"Please place nephrology_reference.pdf in the data/ folder.")
        return

    # 1. Load PDF Document
    try:
        print("Loading PDF document...")
        loader = PyPDFLoader(RAG_SOURCE_PDF)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} page(s) from PDF.")
        
        if not documents:
            print("Error: No content loaded from the PDF.")
            return
            
    except Exception as e:
        print(f"Error loading PDF: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Split Document into Chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split document into {len(chunks)} chunks.")
    
    if not chunks:
        print("Error: No chunks created from the document.")
        return

    # 3. Initialize Embedding Model
    print(f"Initializing HuggingFace Embeddings model: {EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("✅ Embeddings model initialized.")
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Create and Save FAISS Index
    print(f"Creating FAISS index and saving to {FAISS_INDEX_DIR}...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Ensure the directory exists
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        # Save the index and the docstore (metadata)
        vectorstore.save_local(FAISS_INDEX_DIR)
        
        print(f"✅ FAISS index created and saved successfully!")
        print(f"   Location: {FAISS_INDEX_DIR}")
        print(f"   - Index file: {os.path.join(FAISS_INDEX_DIR, 'index.faiss')}")
        print(f"   - Metadata file: {os.path.join(FAISS_INDEX_DIR, 'index.pkl')}")
        
    except Exception as e:
        print(f"Error creating or saving FAISS index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_faiss_index()