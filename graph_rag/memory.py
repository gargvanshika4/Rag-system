"""Conversation memory for multi-turn RAG."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Turn:
    """One user/assistant exchange."""

    query: str
    answer: str


class ConversationMemory:
    """In-memory bounded history keyed by conversation id."""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._store: dict[str, deque[Turn]] = defaultdict(lambda: deque(maxlen=max_turns))

    def add(self, conversation_id: str, query: str, answer: str) -> None:
        """Append a completed turn."""
        self._store[conversation_id].append(Turn(query=query, answer=answer))

    def context(self, conversation_id: str) -> str:
        """Return compact multi-turn context."""
        turns = self._store.get(conversation_id)
        if not turns:
            return ""
        return "\n".join(f"User: {turn.query}\nAssistant: {turn.answer}" for turn in turns)

    def rewrite_query(self, conversation_id: str, query: str) -> str:
        """Blend recent turns into retrieval query for follow-up questions."""
        history = self.context(conversation_id)
        return f"{history}\nCurrent question: {query}" if history else query

    def history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return chat history for the UI."""
        return [{"query": turn.query, "answer": turn.answer} for turn in self._store.get(conversation_id, [])]

    def clear(self, conversation_id: str) -> None:
        """Clear a conversation."""
        self._store.pop(conversation_id, None)
