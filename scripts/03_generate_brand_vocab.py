import json
import pandas as pd
from datasets import Dataset, load_dataset


from src.utils.config import get_config

cfg = get_config()

dataset = load_dataset(
    cfg.dataset_id, cfg.dataset_name, split=cfg.dataset_split
)
assert isinstance(dataset, Dataset)

df = dataset.to_pandas()
assert isinstance(df, pd.DataFrame)


brand_counts = (
    df["store"]
    .dropna()
    .astype(str)
    .str.strip()
    .value_counts()
)

brands = (
    brand_counts[
        brand_counts >= 100
    ]
    .index
    .tolist()
)

EXCLUDED_BRANDS = {
    "amazon renewed",
    "generic",
    "wireless",
    "headphones",
    "laptop",
    "mouse",
    "gaming",
    "computer",
    "keyboard",
    "monitor",
    "camera",
    "speaker",
    "tablet",
    "phone",
    "cable",
    "charger",
    "adapter",
}

brands = [
    brand
    for brand in brands
    if brand.lower() not in EXCLUDED_BRANDS
]

cfg.brand_vocabulary_path.parent.mkdir(parents=True, exist_ok=True)

with open(
    cfg.brand_vocabulary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        brands,
        f,
        indent=2,
        ensure_ascii=False,
    )