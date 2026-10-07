#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from fastembed import TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ============================================================
# Configuration
# ============================================================

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

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "600"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "100"
    )
)

EMBED_BATCH_SIZE = int(
    os.getenv(
        "EMBED_BATCH_SIZE",
        "32"
    )
)


# BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.
VECTOR_SIZE = 384


# ============================================================
# Utility functions
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize PDF extracted text.
    """

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    """
    Split text into overlapping word-based chunks.

    600 words/tokens approximately,
    with 100 overlap.

    This lightweight approach avoids requiring
    another tokenizer dependency.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def get_category(pdf_path: Path) -> str:
    """
    Determine knowledge-base category from directory.
    """

    relative = pdf_path.relative_to(KB_DIR)

    if len(relative.parts) > 1:
        return relative.parts[0]

    return "general"


def extract_pdf(pdf_path: Path):
    """
    Extract text page-by-page from a PDF.
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"WARNING: Could not extract "
                f"{pdf_path.name} page {page_number}: {exc}"
            )
            text = ""

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


def deterministic_id(
    source: str,
    page: int,
    chunk_index: int
):
    """
    Generate deterministic UUID for a chunk.

    Re-running ingestion produces the same IDs.
    """

    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return uuid5(
        NAMESPACE_URL,
        digest
    )


# ============================================================
# Main ingestion
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Ingest PDF knowledge base into Qdrant."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Qdrant collection."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MVP 3A - LOCAL KNOWLEDGE BASE INGESTION")
    print("=" * 70)

    print(f"Knowledge base : {KB_DIR}")
    print(f"Qdrant         : {QDRANT_URL}")
    print(f"Collection     : {COLLECTION_NAME}")
    print(f"Embedding      : {EMBEDDING_MODEL}")
    print(f"Chunk size     : {CHUNK_SIZE}")
    print(f"Chunk overlap  : {CHUNK_OVERLAP}")
    print("=" * 70)

    # --------------------------------------------------------
    # Locate PDFs
    # --------------------------------------------------------

    pdf_files = sorted(
        KB_DIR.rglob("*.pdf")
    )

    if not pdf_files:

        print(
            "\nERROR: No PDF files found."
        )

        print(
            f"Add PDFs under:\n{KB_DIR}"
        )

        print(
            "\nExample:"
        )

        print(
            "data/knowledge_base/isro/example.pdf"
        )

        return 1

    print(
        f"\nFound {len(pdf_files)} PDF file(s)."
    )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    print("\nConnecting to Qdrant...")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # --------------------------------------------------------
    # Create / reset collection
    # --------------------------------------------------------

    existing = [
        c.name
        for c in client.get_collections().collections
    ]

    if args.reset and COLLECTION_NAME in existing:

        print(
            f"Deleting collection: {COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        existing.remove(
            COLLECTION_NAME
        )

    if COLLECTION_NAME not in existing:

        print(
            f"Creating collection: {COLLECTION_NAME}"
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

    else:

        print(
            f"Using existing collection: {COLLECTION_NAME}"
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = TextEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    total_chunks = 0
    total_pages = 0

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: {pdf_path.name}"
        )

        category = get_category(
            pdf_path
        )

        pages = extract_pdf(
            pdf_path
        )

        print(
            f"  Pages with text: {len(pages)}"
        )

        total_pages += len(pages)

        points = []

        for page_info in pages:

            page_number = page_info["page"]
            page_text = page_info["text"]

            chunks = chunk_text(
                page_text
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):

                source = str(
                    pdf_path.relative_to(
                        PROJECT_ROOT
                    )
                )

                point_id = deterministic_id(
                    source,
                    page_number,
                    chunk_index
                )

                points.append(
                    {
                        "id": point_id,
                        "text": chunk,
                        "metadata": {
                            "source": source,
                            "filename": pdf_path.name,
                            "category": category,
                            "page": page_number,
                            "chunk_index": chunk_index,
                            "embedding_model": EMBEDDING_MODEL,
                            "chunk_size": CHUNK_SIZE,
                            "chunk_overlap": CHUNK_OVERLAP
                        }
                    }
                )

        if not points:

            print(
                "  WARNING: No text chunks found."
            )

            continue

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        print(
            f"  Chunks generated: {len(points)}"
        )

        texts = [
            point["text"]
            for point in points
        ]

        embeddings = embedding_model.embed(
            texts
        )

        embeddings = list(
            embeddings
        )

        # ----------------------------------------------------
        # Upload in batches
        # ----------------------------------------------------

        for start in range(
            0,
            len(points),
            EMBED_BATCH_SIZE
        ):

            end = min(
                start + EMBED_BATCH_SIZE,
                len(points)
            )

            batch_points = []

            for index in range(
                start,
                end
            ):

                point = points[index]

                vector = embeddings[index]

                payload = {
                    "text": point["text"],
                    **point["metadata"]
                }

                batch_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=vector.tolist(),
                        payload=payload
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points
            )

            print(
                f"  Uploaded "
                f"{end}/{len(points)} chunks"
            )

        total_chunks += len(points)

    # --------------------------------------------------------
    # Final collection information
    # --------------------------------------------------------

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"PDF files       : {len(pdf_files)}"
    )

    print(
        f"Pages processed : {total_pages}"
    )

    print(
        f"Chunks uploaded : {total_chunks}"
    )

    print(
        f"Collection      : {COLLECTION_NAME}"
    )

    print(
        f"Vector size     : {VECTOR_SIZE}"
    )

    print(
        f"Points in DB    : {collection_info.points_count}"
    )

    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
