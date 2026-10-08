"""MVP 3D: grounded local RAG generation over the validated 3C context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from gateway.tier_clients import StreamChunk, StreamStats, collect_stream

LOCAL_RAG = "LOCAL-RAG"

GROUNDING_SYSTEM_PROMPT = """You are the Local RAG assistant for a verified knowledge base.
Answer the user's question using ONLY the supplied CONTEXT.
Do not use outside knowledge, assumptions, or unstated facts.
If the CONTEXT does not contain enough information to answer, reply exactly:
I don't have enough information in the provided context.
Keep the answer concise and factual. When possible, cite the source and page shown in the context.
"""


@dataclass(frozen=True)
class RagRequest:
    query: str
    context: str


class LocalRagClient:
    """Prompt an existing async stream client with strict context-only grounding."""

    def __init__(self, client) -> None:
        self.client = client

    @staticmethod
    def build_messages(query: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "CONTEXT:\n\n"
                    f"{context.strip()}\n\n"
                    "QUESTION:\n"
                    f"{query.strip()}"
                ),
            },
        ]

    def stream(self, request: RagRequest, *, max_tokens: int = 512,
               temperature: float = 0.2) -> AsyncIterator[StreamChunk]:
        return self.client.stream_chat(
            self.build_messages(request.query, request.context),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate(self, request: RagRequest, *, max_tokens: int = 512,
                       temperature: float = 0.2) -> StreamStats:
        return await collect_stream(self.stream(request, max_tokens=max_tokens, temperature=temperature))
