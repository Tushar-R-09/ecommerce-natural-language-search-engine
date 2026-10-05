from src.retriever import QdrantIndexer
from src.utils.config import get_config 

def create_qdrant_index():
    cfg = get_config()
    indexor = QdrantIndexer()
    indexor.index_parquet(str(cfg.output_dir))
    print("✅ Done!")
    
if __name__ == "__main__":
    create_qdrant_index()
