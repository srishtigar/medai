import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rag_pipeline.index_builder import build_faiss_index

if __name__ == "__main__":
    print("=" * 50)
    print("Building FAISS index from PDF...")
    print("=" * 50)
    build_faiss_index()
    print("=" * 50)
    print("Done! Check above for any errors.")
    print("=" * 50)