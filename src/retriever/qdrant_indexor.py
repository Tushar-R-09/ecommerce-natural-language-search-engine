import uuid
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models

from src.utils.config import get_config

from src.embedding_generator import EmbeddingGenerator

from src.query_parser import QueryParser


class QdrantIndexer:

    def __init__(self):
        self.cfg = get_config()
        self.parser = QueryParser()

        self.client = QdrantClient(
            host=self.cfg.qdrant_host,
            port=self.cfg.qdrant_port,
        )

        self.collection_name = self.cfg.collection_name
        self.embedding_generator = EmbeddingGenerator()

    # --------------------------------------------------
    # Create collection
    # --------------------------------------------------

    def create_collection(self, vector_size: int):

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.price",
            field_schema=models.PayloadSchemaType.FLOAT,
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.brand",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print(
            f"Created collection '{self.collection_name}' "
            f"with vector size {vector_size}"
        )

    # --------------------------------------------------
    # Convert NumPy objects to Python objects
    # --------------------------------------------------

    def convert_numpy(self, obj):

        if hasattr(obj, "tolist"):
            return obj.tolist()

        if isinstance(obj, dict):
            return {
                key: self.convert_numpy(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                self.convert_numpy(value)
                for value in obj
            ]

        return obj

    # --------------------------------------------------
    # Index a batch
    # --------------------------------------------------

    def index_batch(self, batch_corpus: pd.DataFrame):

        points = []

        for _, row in batch_corpus.iterrows():

            # Convert embedding to normal Python list
            embedding = self.convert_numpy(
                row["embedding"]
            )

            # Convert metadata recursively
            metadata = self.convert_numpy(
                row["metadata"]
            )

            if isinstance(metadata, dict) and "price" in metadata:
                p = metadata["price"]
                try:
                    if p is not None and str(p).lower() != "none" and str(p).strip() != "":
                        metadata["price"] = float(str(p).replace("$", "").strip())
                    else:
                        metadata["price"] = None
                except (ValueError, TypeError):
                    metadata["price"] = None

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),

                    vector=embedding,

                    payload={
                        "parent_asin": str(row["parent_asin"]),
                        "document": str(row["document"]),
                        "metadata": metadata,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        print(f"Indexed {len(points)} points")

    # --------------------------------------------------
    # Index Parquet
    # --------------------------------------------------

    def index_parquet(
        self,
        parquet_file: str,
        batch_size: int = 1000,
    ):

        print(f"Reading: {parquet_file}")

        df = pd.read_parquet(parquet_file)

        if df.empty:
            raise ValueError("Parquet file is empty")

        print(f"Loaded {len(df)} rows")

        # Get embedding dimension
        vector_size = len(
            df["embedding"].iloc[0]
        )

        # Create collection
        self.create_collection(vector_size)

        # Upload in batches
        for start in range(
            0,
            len(df),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(df),
            )

            batch = df.iloc[start:end]

            print(
                f"Uploading rows {start} - {end - 1}"
            )

            self.index_batch(batch)

        print(
            f"\nFinished indexing "
            f"{len(df)} rows into "
            f"'{self.collection_name}'"
        )

    def retriever(self, query, top_k):

        parsed = self.parser.parse(query)

        max_price = parsed.max_price
        brand = parsed.brand

        conditions = []

        # Price filter
        if max_price is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.price",
                    range=models.Range(
                        lte=max_price,
                    ),
                )
            )

        if brand is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.brand",
                    match=models.MatchValue(
                        value=brand,
                    ),
                )
            )

        query_filter = None

        if conditions:
            query_filter = models.Filter(
                must=conditions
            )


        #print(type(max_price), brand)

        query_embedding = self.embedding_generator.generate_embeddings([parsed.semantic_query])[0]
        #print(query_embedding)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return response