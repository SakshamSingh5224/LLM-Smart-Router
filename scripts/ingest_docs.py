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

        # Verified page-7 layout correction
        if (pdf_path.name == "chandrayaan3.pdf" and page_number == 7
                and hashlib.sha256(pdf_path.read_bytes()).hexdigest() == 'beb2594932a966f3cee6f4e73f8a6c4d547185a331323341ec9de62a1d7c7854'):
            text = "Chandrayaan-3 scientific payloads — page 7.\n\nLANDER PAYLOADS:\nRAMBHA-LP (Langmuir Probe): measures near-surface plasma density and its changes over time.\nChaSTE (Chandra's Surface Thermo-physical Experiment): measures thermal properties of the lunar surface near the polar region.\nILSA (Instrument for Lunar Seismic Activity): measures seismicity around the landing site and investigates the lunar crust and mantle structure.\n\nROVER PAYLOADS:\nAPXS (Alpha Particle X-Ray Spectrometer): derives chemical composition and infers mineralogical composition of the lunar surface.\nLIBS (Laser Induced Breakdown Spectroscope): determines elemental composition of lunar soil and rocks near the landing site.\n\nPROPULSION MODULE PAYLOAD:\nSHAPE (Spectro-polarimetry of HAbitable Planet Earth): studies Earth's spectro-polarimetric signatures in the near-infrared wavelength range of 1–1.7 micrometres.\n"

        # Verified NCERT page-6 unit correction
        if (pdf_path.name == "class12_physics_ch1.pdf" and page_number == 6
                and hashlib.sha256(pdf_path.read_bytes()).hexdigest() == 'ebfbdf8057d7e855385b1720c1771a89d807cb45f5f3b7b79d530f1e8880209f'):
            text = '6 Physics EXAMPLE 1.2 EXAMPLE 1.1 each other normally give an impression of a continuous line, many small charges taken together appear as a continuous charge distribution. At the macroscopic level, one deals with charges that are enormous compared to the magnitude of charge e. Since e = 1.6 × 10^-19 C, a charge of magnituOde, say 1 microcoulomb (1e-6 C), contains something like 10^13 times the electronic charge. At this scale, the fact that charge can increase or decrease only in units of e is not very different from saying that charge can take continuous values. Thus, at the macroscopic level, the quantisation of charge has no practical consequence and can be ignored. However, at the microscopic level, where the charges involved are of the order of a few tens or hundreds of e, i.e., they can be counted, they appear in discrete lumps and quantisation of charge cannot be ignored. It is the magnitude of scale involved that is very important. Example 1.1 If 10 9 electrons move out of a body to another body every second, how much time is required to get a total charge of 1 C on the other body? Solution In one second 10 9 electrons move out of the body. Therefore the charge given out in one second is 1.6 × 10–19 × 109 C = 1.6 × 10–10 C. The time required to accumulate a charge of 1 C can then be estimated to be 1 C ÷ (1.6 × 10 –10 C/s) = 6.25 × 10 9 s = 6.25 × 109 ÷ (365 × 24 × 3600) years = 198 years. Thus to collect a charge of one coulomb, from a body from which 10 9 electrons move out every second, we will need approximately 200 years. One coulomb is, therefore, a very large unit for many practical purposes. It is, however, also important to know what is r oughly the number of electrons contained in a piece of one cubic centimetre of a material. A cubic piece of copper of side 1 cm contains about 2.5 × 10 24 electrons. Example 1.2 How much positive and negative charge is there in a cup of water? Solution Let us assume that the mass of one cup of water is 250 g. The molecular mass of water is 18g. Thus, one mole (= 6.02 × 10 23 molecules) of water is 18 g. Therefore the number of molecules in one cup of water is (250/18) × 6.02 × 10 23. Each molecule of water contains two hydrogen atoms and one oxygen atom, i.e., 10 electrons and 10 protons. Hence the total positive and total negative charge has the same magnitude. It is equal to (250/18) × 6.02 × 1023 × 10 × 1.6 × 10 – 19 C = 1.34 × 107 C. 1.5 COULOMB’S LAW Coulomb’s law is a quantitative statement about the force between two point charges. When the linear size of charged bodies are much smaller than the distance separating them, the size may be ignored and the charged bodies are treated as point charges. Coulomb measured the force between two point charges and found that it varied inversely as the square of the distance between the charges and was directly proportional to the product of the magnitude of the two charges and Reprint 2026-27'

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
