import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio

from gateway.rag_client import GROUNDING_SYSTEM_PROMPT, LocalRagClient, RagRequest
from gateway.tier_clients import StreamChunk


def test_grounded_prompt_contains_only_supplied_context():
    messages = LocalRagClient.build_messages(
        "When did Chandrayaan-3 launch?",
        "[Source: isro.pdf, page: 12]\nChandrayaan-3 launched on July 14, 2023.",
    )
    assert messages[0]["content"] == GROUNDING_SYSTEM_PROMPT
    assert "ONLY the supplied CONTEXT" in messages[0]["content"]
    assert "Chandrayaan-3 launched on July 14, 2023." in messages[1]["content"]
    assert "When did Chandrayaan-3 launch?" in messages[1]["content"]


def test_local_rag_stream_delegates_to_existing_client():
    class FakeClient:
        async def stream_chat(self, messages, max_tokens=512, temperature=0.7):
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert max_tokens == 128
            assert temperature == 0.2
            yield StreamChunk(delta="grounded")
            yield StreamChunk(done=True)

    async def collect():
        chunks = []
        async for chunk in LocalRagClient(FakeClient()).stream(
            RagRequest("q", "context"), max_tokens=128, temperature=0.2
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert chunks[0].delta == "grounded"
    assert chunks[-1].done is True
