import re

import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def to_list(value):
    """
    Convert numpy arrays / lists / None into
    a clean list of strings.
    """

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        cleaned = []

        for item in value:
            if item is None:
                continue

            text = clean_text(str(item))

            if text:
                cleaned.append(text)

        return cleaned

    text = clean_text(str(value))

    return [text] if text else []

def format_title(row):
    title = clean_text(str(row.get("title", "")))
    return f"Title: {title}"

def format_brand(row) -> str:

    brand = row.get("store")

    if pd.isna(brand) is True:
        return ""

    return f"Brand: {clean_text(str(brand))}"

def format_categories(row: pd.Series) -> str:

    categories = to_list(row.get("categories"))

    if not categories:
        return ""

    return "Categories: " + " > ".join(categories)

def format_features(row: pd.Series) -> str:

    features = to_list(row.get("features"))

    if not features:
        return ""

    bullets = "\n".join(f"- {feature}" for feature in features)

    return f"Features:\n{bullets}"

def format_description(row: pd.Series) -> str:

    description = to_list(row.get("description"))

    if not description:
        return ""

    text = " ".join(description)

    if len(text) > 200:  # Assuming a default max_description_chars value
        text = text[:200].rstrip()

    return f"Description:\n{text}"

def transform_product_row_to_textual_discription(row):
    
    sections = [format_title(row)]

    sections.append(format_brand(row))

    sections.append(format_categories(row))

    sections.append(format_features(row))

    sections.append(format_description(row))

    return "\n\n".join(
        section
        for section in sections
        if section.strip()
    )

def get_image_url(images):
    """Return the first available image URL."""

    if not isinstance(images, dict):
        return None

    for key in ["large", "hi_res", "thumb"]:
        urls = images.get(key)

        if urls is None:
            continue

        # Convert numpy array to list
        if isinstance(urls, np.ndarray):
            urls = urls.tolist()

        for url in urls:
            if url is not None and url != "":
                return url

    return None


def parse_price(val) -> float | None:
    """Parse price string/number to a clean float or None."""
    if val is None or pd.isna(val):
        return None
    try:
        clean_val = str(val).replace("$", "").strip()
        if clean_val.lower() == "none" or not clean_val:
            return None
        return float(clean_val)
    except (ValueError, TypeError):
        return None


def build_metadata(row: pd.Series) -> dict:
    
    return {
        "parent_asin": row.get("parent_asin"),
        "title": row.get("title"),
        "brand": row.get("store"),
        "categories": to_list(row.get("categories")),
        "price": parse_price(row.get("price")),
        "rating": row.get("average_rating"),
        "rating_count": row.get("rating_number"),
        'image_url': get_image_url(row["images"])
    }

def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw Amazon metadata dataframe into a
    searchable corpus.
    """

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "parent_asin": row["parent_asin"],
                "document": transform_product_row_to_textual_discription(row),
                "metadata": build_metadata(row),
            }
        )

    return pd.DataFrame(records)