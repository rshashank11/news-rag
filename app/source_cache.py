import threading
import time
from typing import Optional

from schemas import RetrievedChunk


class SourceCache:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._store: dict[tuple[str, str], tuple[list[RetrievedChunk], float]] = {}
        self._lock = threading.Lock()

    def set(self, session_id: str, source: str, chunks: list[RetrievedChunk]) -> None:
        key = (session_id, source)
        with self._lock:
            self._store[key] = (list(chunks), time.monotonic())

    def get(self, session_id: str, source: str) -> Optional[list[RetrievedChunk]]:
        key = (session_id, source)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            chunks, stored_at = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._store[key]
                return None
            return list(chunks)


source_cache = SourceCache()
