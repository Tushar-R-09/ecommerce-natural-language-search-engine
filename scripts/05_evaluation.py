from __future__ import annotations

from pathlib import Path
from src.utils.config import get_config
from src.reranking import CrossEncoderRanker
import numpy as np
import pandas as pd

reranker = CrossEncoderRanker()

# ============================================================
# CONFIG
# ============================================================
cfg = get_config()

EVAL_DATA_PATH = Path(cfg.eval_data_path)

INPUT_FILE = EVAL_DATA_PATH / "evaluation_data_labeled.csv"
OUTPUT_DIR = EVAL_DATA_PATH / "reranked_results"

K_VALUES = [5, 10, 20]

# Relevance definitions:
# 0 = Not relevant
# 1 = Clicked / relevant
# 2 = Downloaded / highly relevant


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

for query_id, group in df.groupby("query_id"):
    query = group["query"].iloc[0]
    documents = group["document"].tolist()

    reranker_score = reranker.score(query, documents)

    # Add scores
    df.loc[group.index, "reranker_score"] = reranker_score

    # Rank: highest score gets rank 1
    df.loc[group.index, "reranker_rank"] = (
        pd.Series(reranker_score, index=group.index)
        .rank(method="first", ascending=False)
        .astype(int)
    )

required_columns = {
    "query_id",
    "query",
    "query_type",
    "rank",
    "relevance",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Make sure rows are correctly ordered
df = df.sort_values(
    ["query_id", "rank"]
).reset_index(drop=True)


# Convert relevance to numeric
df["relevance"] = pd.to_numeric(
    df["relevance"],
    errors="coerce",
)


# Remove rows without labels
df = df.dropna(
    subset=["relevance"]
).copy()

df["relevance"] = df["relevance"].astype(int)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def dcg(relevances: list[int]) -> float:
    """
    Discounted Cumulative Gain.

    Relevance:
        0 = not relevant
        1 = relevant
        2 = highly relevant
    """

    return sum(
        (2 ** relevance - 1) / np.log2(rank + 1)
        for rank, relevance in enumerate(
            relevances,
            start=1,
        )
    )


def ndcg_at_k(
    relevances: list[int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain.
    """
    # relevances[:k] = [1, 0, 2, 2]
    # 

    actual = relevances[:k]

    if not actual:
        return 0.0

    actual_dcg = dcg(actual)

    # Ideal ranking: highest relevance first
    # ideal = [2, 2, 1, 0]
    ideal = sorted(
        relevances,
        reverse=True,
    )[:k]

    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def reciprocal_rank_at_k(
    relevances: list[int],
    k: int,
) -> float:
    """
    Reciprocal rank of the first relevant result.

    relevance > 0 is considered relevant.
    """
    # relevances[:k] = [0, 0, 1, 2] === > 1/2 = 0.5
    # Ideal behaviour = [2, 1, 0, 0]
    # Ideal reciprocal rank = 1/1 = 1


    for rank, relevance in enumerate(
        relevances[:k],
        start=1,
    ):
        if relevance > 0:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    relevances: list[int],
    k: int,
) -> float:
    """
    Recall@K.

    Relevant = relevance > 0.

    Ans : Out of all the relevant items, how many of them are in top k
    """

    total_relevant = sum(
        relevance > 0
        for relevance in relevances
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        relevance > 0
        for relevance in relevances[:k]
    )

    return retrieved_relevant / total_relevant


def precision_at_k(
    relevances: list[int],
    k: int,
) -> float:
    """
    Precision@K.

    Relevant = relevance > 0.

    Answers how many relevant items we are getting in top_K 
    """

    top_k = relevances[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        relevance > 0
        for relevance in top_k
    )

    return relevant / len(top_k)


def hit_rate_at_k(
    relevances: list[int],
    k: int,
) -> float:
    """
    Returns 1 if at least one relevant result
    appears in top-K, otherwise 0.
    """

    return float(
        any(
            relevance > 0
            for relevance in relevances[:k]
        )
    )


# ============================================================
# EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    relevances: list[int],
) -> dict:

    metrics = {}

    for k in K_VALUES:

        metrics[f"ndcg@{k}"] = ndcg_at_k(
            relevances,
            k,
        )

        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(
            relevances,
            k,
        )

        metrics[f"recall@{k}"] = recall_at_k(
            relevances,
            k,
        )

        metrics[f"precision@{k}"] = precision_at_k(
            relevances,
            k,
        )

        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            relevances,
            k,
        )

    return metrics


# ============================================================
# EVALUATE EVERY QUERY
# ============================================================

query_results = []

for query_id, query_df in df.groupby(
    "query_id",
    sort=True,
):

    # Retrival system evaluation 
    # query_df = query_df.sort_values(
    #     "rank"
    # )

    # complete system evauation with reranker
    query_df = query_df.sort_values(
        "reranker_rank"
    )

    relevances = (
        query_df["relevance"]
        .tolist()
    )

    first_row = query_df.iloc[0]

    metrics = evaluate_query(
        relevances
    )

    result = {
        "query_id": query_id,
        "query": first_row["query"],
        "query_type": first_row["query_type"],
        "num_results": len(query_df),
        "num_relevant": sum(
            relevance > 0
            for relevance in relevances
        ),
        "num_highly_relevant": sum(
            relevance == 2
            for relevance in relevances
        ),
    }

    result.update(metrics)

    query_results.append(result)


query_metrics_df = pd.DataFrame(
    query_results
)


# ============================================================
# OVERALL METRICS
# ============================================================

metric_columns = [
    column
    for column in query_metrics_df.columns
    if column.startswith(
        (
            "ndcg@",
            "mrr@",
            "recall@",
            "precision@",
            "hit_rate@",
        )
    )
]


overall_metrics = (
    query_metrics_df[metric_columns]
    .mean()
    .reset_index()
)

overall_metrics.columns = [
    "metric",
    "value",
]


# ============================================================
# METRICS BY QUERY TYPE
# ============================================================

metrics_by_type = (
    query_metrics_df
    .groupby("query_type")[metric_columns]
    .mean()
    .reset_index()
)


# ============================================================
# METRICS BY INDIVIDUAL QUERY
# ============================================================

per_query_output = query_metrics_df.copy()


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

overall_file = (
    OUTPUT_DIR / "overall_metrics.csv"
)

by_type_file = (
    OUTPUT_DIR / "metrics_by_query_type.csv"
)

per_query_file = (
    OUTPUT_DIR / "metrics_per_query.csv"
)

overall_metrics.to_csv(
    overall_file,
    index=False,
)

metrics_by_type.to_csv(
    by_type_file,
    index=False,
)

per_query_output.to_csv(
    per_query_file,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("RETRIEVAL SYSTEM EVALUATION")
print("=" * 80)

print()

print(
    f"Queries evaluated : "
    f"{len(query_metrics_df)}"
)

print(
    f"Candidates/query  : "
    f"{df.groupby('query_id').size().mean():.1f}"
)

print()

print("-" * 80)
print("OVERALL METRICS")
print("-" * 80)

for _, row in overall_metrics.iterrows():

    print(
        f"{row['metric']:20s}: "
        f"{row['value']:.4f}"
    )


print()

print("-" * 80)
print("METRICS BY QUERY TYPE")
print("-" * 80)

print(
    metrics_by_type.to_string(
        index=False
    )
)


print()

print("-" * 80)
print("OUTPUT FILES")
print("-" * 80)

print(
    f"Overall metrics : {overall_file}"
)

print(
    f"By query type   : {by_type_file}"
)

print(
    f"Per query       : {per_query_file}"
)

print("=" * 80)