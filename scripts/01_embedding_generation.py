import pandas as pd
from datasets import load_dataset

from src.utils.config import get_config
from src.preprocessing import build_corpus
from src.embedding_generator import EmbeddingGenerator

cfg = get_config()

dataset = load_dataset(
    cfg.dataset_id,
    cfg.dataset_name,
    split=cfg.dataset_split
)

df = dataset.to_pandas()
assert isinstance(df, pd.DataFrame)

df = df.sample(n=min(cfg.sample_size, len(df)), random_state=42).reset_index(drop=True)
print("Data loaded")

corpus_df = build_corpus(df)

print("Corpus built successfully")

generator = EmbeddingGenerator()
generator.embed_corpus(corpus_df)

