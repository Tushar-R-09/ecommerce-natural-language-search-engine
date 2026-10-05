from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from src.utils.config import get_config

cfg = get_config()

@dataclass
class ParsedQuery:
    """
    Result of parsing a user's search query.
    """

    semantic_query: str
    max_price: float | None = None
    brand: str | None = None


class QueryParser:
    """
    Parses natural-language e-commerce queries.

    Currently supports:

        Price:
            under $50
            below $100
            less than $30
            under 50 dollars
            $50 or less

        Brand:
            Logitech wireless mouse
            Sony headphones
            Apple laptop
    """

    PRICE_PATTERNS = [
        # under $50
        re.compile(
            r"\bunder\s+\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?)?\b",
            re.IGNORECASE,
        ),

        # below $50
        re.compile(
            r"\bbelow\s+\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?)?\b",
            re.IGNORECASE,
        ),

        # less than $50
        re.compile(
            r"\bless\s+than\s+\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?)?\b",
            re.IGNORECASE,
        ),

        # $50 or less
        re.compile(
            r"\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?)?\s+or\s+less\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        brand_vocabulary_path: str | Path | None = None,
    ):
        """
        Load the brand vocabulary.

        If no path is provided, automatically use the path defined in settings.
        """
        if brand_vocabulary_path is None:
            brand_vocabulary_path = cfg.brand_vocabulary_path

        brand_vocabulary_path = Path(
            brand_vocabulary_path
        )

        if not brand_vocabulary_path.exists():
            raise FileNotFoundError(
                "Brand vocabulary not found: "
                f"{brand_vocabulary_path}"
            )

        with open(
            brand_vocabulary_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.brands: list[str] = json.load(f)

        if not isinstance(self.brands, list):
            raise ValueError(
                "Brand vocabulary must contain "
                "a JSON list of brands."
            )

    def _extract_price(
        self,
        query: str,
    ) -> tuple[str, float | None]:
        """
        Extract maximum price from query.

        Returns
        -------
        tuple
            (cleaned_query, max_price)
        """

        max_price = None
        cleaned_query = query

        for pattern in self.PRICE_PATTERNS:

            match = pattern.search(
                cleaned_query
            )

            if match:

                max_price = float(
                    match.group(1)
                )

                cleaned_query = (
                    cleaned_query[:match.start()]
                    + " "
                    + cleaned_query[match.end():]
                )

                break

        return (
            cleaned_query,
            max_price,
        )

    def _extract_brand(
        self,
        query: str,
    ) -> tuple[str, str | None]:
        """
        Extract a known brand from the query.
        """

        if not self.brands:
            return query, None

        # Check longer brand names first.
        #
        # Example:
        #
        #     "Amazon Renewed"
        #
        # should be checked before:
        #
        #     "Amazon"
        #
        sorted_brands = sorted(
            self.brands,
            key=len,
            reverse=True,
        )

        for brand in sorted_brands:

            if not brand:
                continue

            pattern = re.compile(
                rf"\b{re.escape(brand)}\b",
                re.IGNORECASE,
            )

            match = pattern.search(
                query
            )

            if match:

                cleaned_query = (
                    query[:match.start()]
                    + " "
                    + query[match.end():]
                )

                return (
                    cleaned_query,
                    brand,
                )

        return query, None

    def parse(
        self,
        query: str,
    ) -> ParsedQuery:
        """
        Parse a natural-language search query.

        Example
        -------
        Input:

            "Logitech wireless mouse under $50"

        Output:

            semantic_query = "wireless mouse"
            brand = "Logitech"
            max_price = 50.0
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # ----------------------------------------------------------
        # Extract price
        # ----------------------------------------------------------

        semantic_query, max_price = (
            self._extract_price(query)
        )

        # ----------------------------------------------------------
        # Extract brand
        # ----------------------------------------------------------

        semantic_query, brand = (
            self._extract_brand(
                semantic_query
            )
        )

        semantic_query = re.sub(
            r"\s+",
            " ",
            semantic_query,
        ).strip()

        return ParsedQuery(
            semantic_query=semantic_query,
            max_price=max_price,
            brand=brand,
        )