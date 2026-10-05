from typing import Any

from sentence_transformers import CrossEncoder
from src.utils.config import get_config


class CrossEncoderRanker:
    """
    Cross-encoder based reranker.

    Takes:
        query + candidate document

    and produces:
        relevance score
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.reranker_model_name

        self.model = CrossEncoder(
            self.model_name,
            device=device,
        )

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        Score query-document pairs.
        """

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        return scores.tolist()

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Rerank dense retrieval results.
        """

        if not results:
            return []

        documents = [
            result["document"]
            for result in results
        ]

        scores = self.score(
            query=query,
            documents=documents,
        )

        reranked_results = []

        for result, score in zip(
            results,
            scores,
        ):

            result = result.copy()
            result["rerank_score"] = score
            reranked_results.append(result)

        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results