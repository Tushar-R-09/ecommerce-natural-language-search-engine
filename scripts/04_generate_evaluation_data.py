from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.retriever import QdrantIndexer


# ============================================================
# CONFIG
# ============================================================

TOP_K = 20

OUTPUT_DIR = Path("../eval_data")
OUTPUT_FILE = OUTPUT_DIR / "evaluation_data.csv"


# ============================================================
# EVALUATION QUERIES
# ============================================================

EVALUATION_QUERIES = {

    # --------------------------------------------------------
    # Exact-ish queries
    # --------------------------------------------------------

    "exact": [
        "wireless gaming mouse",
        "mechanical gaming keyboard",
        "bluetooth headphones",
        "4K HDMI cable",
        "USB C hub",
        "wireless webcam",
        "laptop cooling pad",
        "computer speakers",
        "external hard drive",
        "gaming monitor",
    ],

    # --------------------------------------------------------
    # Attribute-heavy queries
    # --------------------------------------------------------

    "attribute": [
        "wireless mouse with rechargeable battery",
        "mechanical keyboard with RGB backlight",
        "noise cancelling wireless headphones",
        "4K monitor with HDMI port",
        "USB C hub with multiple ports",
        "wireless headphones with microphone",
        "gaming mouse with programmable buttons",
        "portable laptop stand with adjustable height",
        "webcam with built in microphone",
        "external hard drive with large storage capacity",
    ],

    # --------------------------------------------------------
    # Intent / use-case queries
    # --------------------------------------------------------

    "intent": [
        "mouse for competitive gaming",
        "keyboard for programming",
        "headphones for travelling",
        "monitor for working from home",
        "webcam for video conferencing",
        "speakers for watching movies",
        "storage device for backing up photos",
        "USB hub for connecting laptop accessories",
        "cooling pad for a gaming laptop",
        "headphones for listening to music",
    ],

    # --------------------------------------------------------
    # Natural-language queries
    # --------------------------------------------------------

    "natural_language": [
        "I need a mouse that I can use without a cable",
        "I want a keyboard that is good for long hours of typing",
        "I need headphones that block outside noise while travelling",
        "I want to connect my laptop to a 4K television",
        "I need something that lets me connect several USB devices to my laptop",
        "I want a monitor that is comfortable for programming all day",
        "I need a webcam for online meetings and video calls",
        "I want something to keep my gaming laptop from getting too hot",
        "I need extra storage to back up my photos and videos",
        "I want speakers that are good for watching movies on my computer",
    ],
}


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

indexer = QdrantIndexer()


# ============================================================
# RUN RETRIEVAL
# ============================================================

rows = []

total_queries = sum(
    len(queries)
    for queries in EVALUATION_QUERIES.values()
)

query_number = 0


for query_type, queries in EVALUATION_QUERIES.items():

    for query in queries:

        query_number += 1

        print(
            f"[{query_number}/{total_queries}] "
            f"{query_type}: {query}"
        )

        # ----------------------------------------------------
        # Retrieve top-K results
        # ----------------------------------------------------

        responses = indexer.retriever(
            query=query,
            top_k=TOP_K,
        )

        # ----------------------------------------------------
        # Store every retrieved candidate
        # ----------------------------------------------------

        for rank, point in enumerate(responses.points, start=1):

            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            rows.append(
                {
                    # Query information
                    "query_id": query_number,
                    "query": query,
                    "query_type": query_type,

                    # Retrieval information
                    "rank": rank,
                    "score": point.score,

                    # Product information
                    "id": point.id,
                    "parent_asin": payload.get("parent_asin"),

                    # Metadata
                    "title": metadata.get("title"),
                    "brand": metadata.get("brand"),
                    "price": metadata.get("price"),
                    "rating": metadata.get("rating"),
                    "rating_count": metadata.get("rating_count"),
                    "categories": metadata.get("categories"),
                    "image_url": metadata.get("image_url"),

                    # Full document
                    "document": payload.get("document"),
                    "relevance": None
                }
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

evaluation_df = pd.DataFrame(rows)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE
# ============================================================

evaluation_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("EVALUATION DATASET CREATED")
print("=" * 80)

print(f"Queries          : {total_queries}")
print(f"Candidates/query : {TOP_K}")
print(f"Expected rows    : {total_queries * TOP_K}")
print(f"Actual rows      : {len(evaluation_df)}")

print()
print("Queries by type:")
print(
    evaluation_df[
        ["query_type", "query_id"]
    ]
    .drop_duplicates()
    ["query_type"]
    .value_counts()
)

print()
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 80)