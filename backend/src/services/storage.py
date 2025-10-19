from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


class StorageService:
    """Abstract storage interface for sessions and messages.

    Replace this with a DB-backed implementation without changing routers.
    """

    def create_session(self, config: Dict[str, Any]) -> str:
        raise NotImplementedError

    def session_exists(self, session_id: str) -> bool:
        raise NotImplementedError

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def append_message(self, session_id: str, message: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def update_session(self, session_id: str, patch: Dict[str, Any]) -> None:
        """Update mutable fields in a session. In MVP, only 'config' and 'status'."""
        raise NotImplementedError


class InMemoryStorage(StorageService):
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}

    def create_session(self, config: Dict[str, Any]) -> str:
        from uuid import uuid4

        session_id = str(uuid4())
        self.sessions[session_id] = {
            "config": config,
            "createdAt": datetime.utcnow(),
            "status": "in_progress",
        }
        self.messages.setdefault(session_id, [])
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self.messages.get(session_id, [])

    def append_message(self, session_id: str, message: Dict[str, Any]) -> None:
        self.messages.setdefault(session_id, []).append(message)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, patch: Dict[str, Any]) -> None:
        sess = self.sessions.get(session_id)
        if not sess:
            return
        # merge top-level keys
        for k, v in patch.items():
            if k == "config" and isinstance(v, dict):
                sess.setdefault("config", {}).update(v)
            else:
                sess[k] = v


# Singleton provider for DI
_storage_singleton: Optional[StorageService] = None


def get_storage() -> StorageService:
    global _storage_singleton
    if _storage_singleton is None:
        _storage_singleton = InMemoryStorage()
    return _storage_singleton
