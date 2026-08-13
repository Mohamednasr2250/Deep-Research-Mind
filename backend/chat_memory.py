"""
chat_memory.py — Conversational Memory for RAG follow-up questions
"""

from datetime import datetime
from typing import List, Dict, Optional
import json
import os

MEMORY_STORE_FILE = "chat_sessions.json"


class ChatMemory:
    def __init__(self, session_id: str, max_turns: int = 5):
        self.session_id = session_id
        self.max_turns  = max_turns
        self.turns: List[Dict] = []

    def add_turn(self, role: str, content: str):
        self.turns.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def get_history_text(self) -> str:
        if not self.turns:
            return ""
        lines = [f"{t['role']}: {t['content'][:300]}" for t in self.turns]
        return "\n".join(lines)

    def get_recent_query(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["content"]
        return None

    def clear(self):
        self.turns = []


class ChatMemoryManager:
    def __init__(self):
        self._sessions: Dict[str, ChatMemory] = {}
        self._load_from_disk()

    def get_session(self, session_id: str) -> ChatMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMemory(session_id)
        return self._sessions[session_id]

    def add_exchange(self, session_id: str, user_query: str, assistant_answer: str):
        session = self.get_session(session_id)
        session.add_turn("user", user_query)
        session.add_turn("assistant", assistant_answer)
        self._save_to_disk()

    def build_contextual_query(self, session_id: str, current_query: str) -> str:
        session = self.get_session(session_id)
        history_text = session.get_history_text()

        if not history_text:
            return current_query

        return f"""Previous conversation:
{history_text}

Current follow-up question: {current_query}"""

    def get_history(self, session_id: str) -> List[Dict]:
        return self.get_session(session_id).turns

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].clear()
        self._save_to_disk()

    def _save_to_disk(self):
        data = {
            sid: mem.turns for sid, mem in self._sessions.items()
        }
        try:
            with open(MEMORY_STORE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_from_disk(self):
        if not os.path.exists(MEMORY_STORE_FILE):
            return
        try:
            with open(MEMORY_STORE_FILE) as f:
                data = json.load(f)
            for sid, turns in data.items():
                mem = ChatMemory(sid)
                mem.turns = turns
                self._sessions[sid] = mem
        except Exception:
            pass


chat_memory_manager = ChatMemoryManager()
