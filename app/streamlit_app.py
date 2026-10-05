from __future__ import annotations

import re
import time

import streamlit as st

from src.retriever import QdrantIndexer
from src.reranking import CrossEncoderRanker


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Semantic Product Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main content */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Search box */
    div[data-testid="stTextInput"] input {
        font-size: 1.05rem;
        padding: 0.75rem;
    }

    /* Product card */
    .product-title {
        font-size: 1.15rem;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }

    .product-brand {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .product-price {
        font-size: 1.15rem;
        font-weight: 600;
    }

    .product-rating {
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .score-label {
        color: #666;
        font-size: 0.8rem;
    }

    .score-value {
        font-size: 1.25rem;
        font-weight: 600;
    }

    /* Search stats */
    .search-stat {
        text-align: center;
    }

    .search-stat-value {
        font-size: 1.2rem;
        font-weight: 600;
    }

    .search-stat-label {
        font-size: 0.8rem;
        color: #666;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.title("🔎 Semantic Product Search")

st.markdown(
    """
    Find products using **natural language**.

    Try searches like:
    - *wireless headphones for working out*
    - *Apple wired earphones under $30*
    - *comfortable headphones with microphone*
    """
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Number of candidates",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of products retrieved from Qdrant before reranking.",
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        **1. Semantic Retrieval**

        Qdrant finds products that are semantically similar to your query.

        **2. Cross-Encoder Reranking**

        The retrieved products are scored again using a cross-encoder.

        **3. Final Ranking**

        Products are displayed according to their reranker score.
        """
    )


# ============================================================
# Load search engine
# ============================================================

@st.cache_resource
def load_search_engine():

    indexer = QdrantIndexer()
    reranker = CrossEncoderRanker()

    return indexer, reranker


with st.spinner("Loading search engine..."):

    indexer, reranker = load_search_engine()


# ============================================================
# Search input
# ============================================================

query = st.text_input(
    "Search for a product",
    placeholder="e.g. Apple wired earphones with microphone",
    label_visibility="collapsed",
)

search_clicked = st.button(
    "🔍 Search",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Helper functions
# ============================================================

def clean_image_url(image_url: str | None) -> str | None:
    """
    Extract a real URL from either a plain URL or Markdown link.
    """

    if not image_url:
        return None

    image_url = str(image_url).strip()

    # Markdown:
    # [https://example.com/image.jpg](https://example.com/image.jpg)
    match = re.search(
        r"\]\((https?://[^)]+)\)",
        image_url,
    )

    if match:
        return match.group(1)

    # Plain URL
    if image_url.startswith("http"):
        return image_url

    return None


def format_price(price):
    """Format product price safely."""

    if price is None:
        return None

    try:
        return f"${float(price):,.2f}"
    except (ValueError, TypeError):
        return str(price)


def get_rating_text(rating, rating_count):

    if rating is None:
        return None

    try:
        rating_text = f"⭐ {float(rating):.1f}"

        if rating_count is not None:
            rating_text += f"  ·  {int(rating_count):,} reviews"

        return rating_text

    except (ValueError, TypeError):
        return str(rating)


# ============================================================
# Search
# ============================================================

if search_clicked:

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        start_time = time.perf_counter()

        with st.spinner("Searching products..."):

            # ------------------------------------------------
            # Semantic retrieval
            # ------------------------------------------------

            response = indexer.retriever(
                query=query.strip(),
                top_k=top_k,
            )

            candidates = []

            for point in response.points:

                payload = point.payload or {}

                candidates.append(
                    {
                        "id": point.id,
                        "vector_score": point.score,
                        "parent_asin": payload.get(
                            "parent_asin"
                        ),
                        "document": payload.get(
                            "document"
                        ),
                        "metadata": payload.get(
                            "metadata"
                        ) or {},
                    }
                )

            # ------------------------------------------------
            # Cross-encoder reranking
            # ------------------------------------------------

            reranked_results = reranker.rerank(
                query.strip(),
                candidates,
            )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # ====================================================
        # Results header
        # ====================================================

        st.divider()

        header_col1, header_col2, header_col3 = st.columns(3)

        with header_col1:

            st.markdown(
                f"""
                <div class="search-stat">
                    <div class="search-stat-value">
                        {len(reranked_results)}
                    </div>
                    <div class="search-stat-label">
                        Products
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with header_col2:

            st.markdown(
                f"""
                <div class="search-stat">
                    <div class="search-stat-value">
                        {latency_ms:.0f} ms
                    </div>
                    <div class="search-stat-label">
                        Search latency
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with header_col3:

            st.markdown(
                f"""
                <div class="search-stat">
                    <div class="search-stat-value">
                        {top_k}
                    </div>
                    <div class="search-stat-label">
                        Retrieved candidates
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ====================================================
        # No results
        # ====================================================

        if not reranked_results:

            st.info(
                "No products found. Try describing your product differently."
            )

        else:

            st.subheader(
                f"Search results for “{query.strip()}”"
            )

            st.caption(
                "Products are ordered by cross-encoder relevance."
            )

            # =================================================
            # Product cards
            # =================================================

            for rank, result in enumerate(
                reranked_results,
                start=1,
            ):

                metadata = (
                    result.get("metadata")
                    or {}
                )

                title = metadata.get(
                    "title",
                    "Unknown Product",
                )

                brand = metadata.get(
                    "brand",
                    "Unknown brand",
                )

                price = metadata.get(
                    "price"
                )

                rating = metadata.get(
                    "rating"
                )

                rating_count = metadata.get(
                    "rating_count"
                )

                image_url = clean_image_url(
                    metadata.get("image_url")
                )

                document = result.get(
                    "document"
                )

                rerank_score = result.get(
                    "rerank_score"
                )

                # =================================================
                # Card
                # =================================================

                with st.container(border=True):

                    image_col, info_col, score_col = st.columns(
                        [2, 7, 2],
                        vertical_alignment="center",
                    )

                    # -------------------------------------------------
                    # Image
                    # -------------------------------------------------

                    with image_col:

                        if image_url:

                            try:

                                st.image(
                                    image_url,
                                    width=160,
                                )

                            except Exception:

                                st.caption(
                                    "Image unavailable"
                                )

                        else:

                            st.caption(
                                "No image available"
                            )

                    # -------------------------------------------------
                    # Product information
                    # -------------------------------------------------

                    with info_col:

                        st.markdown(
                            f"""
                            <div class="product-title">
                                {rank}. {title}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="product-brand">
                                {brand}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Price
                        price_text = format_price(
                            price
                        )

                        if price_text:

                            st.markdown(
                                f"""
                                <div class="product-price">
                                    {price_text}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Rating
                        rating_text = get_rating_text(
                            rating,
                            rating_count,
                        )

                        if rating_text:

                            st.markdown(
                                f"""
                                <div class="product-rating">
                                    {rating_text}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Product details
                        if document:

                            with st.expander(
                                "View product details"
                            ):

                                st.write(
                                    document
                                )

                    # -------------------------------------------------
                    # Relevance score
                    # -------------------------------------------------

                    with score_col:

                        st.markdown(
                            '<div class="score-label">'
                            "Relevance"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        if rerank_score is not None:

                            st.markdown(
                                f"""
                                <div class="score-value">
                                    {rerank_score:.3f}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.caption(
                            f"Rank #{rank}"
                        )


# ============================================================
# Empty state
# ============================================================

else:

    st.info(
        "👆 Enter a product description above to start searching."
    )