import sys
import os
import shutil
from pathlib import Path
import pytest

# 1. Set environment variables BEFORE importing gateway modules
test_db_path = "./test_chroma_db"
os.environ["GATEWAY_CHROMA_DB_DIR"] = test_db_path
os.environ["RETRIEVAL_DISTANCE_THRESHOLD"] = "0.7"  # Tightened for all-MiniLM-L6-v2

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.retriever import LocalRetriever

@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    yield
    # Cleanup the test database only after all tests finish
    shutil.rmtree(test_db_path, ignore_errors=True)

@pytest.fixture(scope="module")
def test_retriever():
    retriever = LocalRetriever(collection_name="test_context")
    retriever.collection.upsert(
        documents=[
            "ISRO launched Chandrayaan-3 on July 14, 2023.", 
            "The Supreme Court of India is the highest judicial court."
        ],
        metadatas=[{"source": "isro"}, {"source": "sc"}],
        ids=["doc1", "doc2"]
    )
    return retriever

def test_successful_retrieval(test_retriever):
    context = test_retriever.search("When was Chandrayaan launched?")
    assert "ISRO launched Chandrayaan-3" in context
    assert "Supreme Court" not in context

def test_out_of_domain_retrieval(test_retriever):
    # Irrelevant query should exceed distance threshold and return empty
    context = test_retriever.search("What is the recipe for chocolate chip cookies?")
    assert context == ""

def test_empty_query(test_retriever):
    # Empty query should short-circuit and return empty
    context = test_retriever.search("   ")
    assert context == ""
