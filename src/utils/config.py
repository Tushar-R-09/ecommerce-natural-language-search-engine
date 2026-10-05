import torch
from pathlib import Path
import os

class Config:

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.output_dir = self.base_dir / "data"
        self.dataset_id = "McAuley-Lab/Amazon-Reviews-2023"
        self.dataset_name = "raw_meta_Electronics"
        self.dataset_split = "full"
        self.sample_size = 100_000 
        self.brand_vocabulary_path = self.base_dir / "brand_vocab" / "brand_vocabulary.json"
        self.eval_data_path = self.base_dir / "eval_data"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        self.model_name = "BAAI/bge-base-en-v1.5"
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.batch_size = 32

        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        self.collection_name = "semantic_search"

def get_config() -> Config:
    return Config()