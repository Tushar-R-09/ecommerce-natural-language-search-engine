import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config import get_config
from pathlib import Path

class EmbeddingGenerator:

    def __init__(self):
        self.cfg = get_config()
        self.device = self.cfg.device
        self.model_name = self.cfg.model_name
        self.batch_size = self.cfg.batch_size
        self.output_dir = Path(self.cfg.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings

    def _batch_path(
        self,
        batch_id: int,
    ) -> Path:

        return self.output_dir / (f"batch_{batch_id:06d}.parquet")

    def embed_corpus(self, corpus: pd.DataFrame):

        total_batches = (len(corpus) + self.batch_size - 1) // self.batch_size

        for batch_id in tqdm(
            range(total_batches),
            desc="Embedding batches",
        ):
            start = batch_id * self.batch_size
            end = min((batch_id + 1) * self.batch_size, len(corpus))
            batch_corpus = corpus.iloc[start:end].copy()
            batch_texts = batch_corpus["document"].tolist()
            batch_embeddings = self.generate_embeddings(batch_texts)
            batch_corpus["embedding"] = batch_embeddings.tolist()

            batch_corpus.to_parquet(self._batch_path(batch_id), index=False)

        return 
        
        

        