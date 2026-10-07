#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "bharat_knowledge_base"
)


def fail(message):

    print(
        f"\nFAIL: {message}"
    )

    sys.exit(1)


def main():

    print("=" * 70)
    print("MVP 3A VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check PDFs
    # --------------------------------------------------------

    pdfs = list(
        KB_DIR.rglob("*.pdf")
    )

    print(
        f"\nPDF files found: {len(pdfs)}"
    )

    if not pdfs:

        fail(
            "No PDF documents found in "
            f"{KB_DIR}"
        )

    # --------------------------------------------------------
    # Check Qdrant
    # --------------------------------------------------------

    print(
        "\nConnecting to Qdrant..."
    )

    try:

        client = QdrantClient(
            url=QDRANT_URL
        )

        client.get_collections()

    except Exception as exc:

        fail(
            f"Qdrant is not reachable: {exc}"
        )

    print(
        "Qdrant connection: PASS"
    )

    # --------------------------------------------------------
    # Check collection
    # --------------------------------------------------------

    collections = [
        collection.name
        for collection
        in client.get_collections().collections
    ]

    print(
        f"\nCollections: {collections}"
    )

    if COLLECTION_NAME not in collections:

        fail(
            f"Collection '{COLLECTION_NAME}' "
            "does not exist."
        )

    print(
        f"Collection '{COLLECTION_NAME}': PASS"
    )

    # --------------------------------------------------------
    # Check points
    # --------------------------------------------------------

    collection = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"\nPoints in collection: "
        f"{collection.points_count}"
    )

    if not collection.points_count:

        fail(
            "Collection exists but contains "
            "no vectors."
        )

    print(
        "Vector data: PASS"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "PASS: MVP 3A vector infrastructure "
        "and ingestion are ready for retrieval integration."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
