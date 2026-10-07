#!/usr/bin/env python3

import argparse
import os

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "bharat_knowledge_base"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "10"
    )
)


def main():

    parser = argparse.ArgumentParser(
        description="Test semantic retrieval from Qdrant."
    )

    parser.add_argument(
        "query",
        help="Question to search in the knowledge base."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MVP 3A - SEMANTIC RETRIEVAL TEST")
    print("=" * 70)

    print(
        f"Query: {args.query}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = TextEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Embed query
    # --------------------------------------------------------

    query_vector = list(
        embedding_model.embed(
            [args.query]
        )
    )[0]

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=args.top_k,
        with_payload=True
    )

    points = results.points

    print(
        f"\nRetrieved {len(points)} result(s)\n"
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    for rank, result in enumerate(
        points,
        start=1
    ):

        payload = result.payload or {}

        print("-" * 70)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Score      : {result.score:.4f}"
        )

        print(
            f"Source     : {payload.get('source')}"
        )

        print(
            f"Category   : {payload.get('category')}"
        )

        print(
            f"Page       : {payload.get('page')}"
        )

        print(
            f"Chunk      : {payload.get('chunk_index')}"
        )

        print(
            "\nText:"
        )

        print(
            payload.get(
                "text",
                ""
            )
        )

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
