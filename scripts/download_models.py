import os
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

def main():
    print(f"Pre-caching embedding model: {EMBEDDING_MODEL}...")
    SentenceTransformer(EMBEDDING_MODEL)
    print(f"Pre-caching reranker model: {RERANKER_MODEL}...")
    CrossEncoder(RERANKER_MODEL)
    print("✅ Models cached successfully.")

if __name__ == "__main__":
    main()
