"""
tracing.py — Full Request-Level Tracing
"""

import time
import uuid
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

TRACE_LOG_FILE = "request_traces.json"


class RequestTrace:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())[:8]
        self.query = query
        self.started_at = time.time()
        self.steps: List[Dict[str, Any]] = []
        self._step_start: Optional[float] = None

    def start_step(self, name: str):
        self._step_start = time.time()
        self._current_step_name = name

    def end_step(self, detail: Optional[Dict] = None):
        if self._step_start is None:
            return
        duration = round(time.time() - self._step_start, 4)
        self.steps.append({
            "step": self._current_step_name,
            "duration_seconds": duration,
            "detail": detail or {}
        })
        self._step_start = None

    def step(self, name: str, fn, *args, **kwargs):
        self.start_step(name)
        result = fn(*args, **kwargs)
        self.end_step()
        return result

    def finish(self, final_answer: str = "", flags: Optional[Dict] = None) -> Dict:
        total_duration = round(time.time() - self.started_at, 4)
        record = {
            "trace_id": self.trace_id,
            "query": self.query,
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "steps": self.steps,
            "final_answer_preview": final_answer[:200],
            "flags": flags or {}
        }
        self._save(record)
        return record

    def _save(self, record: Dict):
        traces = []
        if os.path.exists(TRACE_LOG_FILE):
            try:
                with open(TRACE_LOG_FILE) as f:
                    traces = json.load(f)
            except Exception:
                traces = []

        traces.append(record)
        traces = traces[-200:]

        try:
            with open(TRACE_LOG_FILE, "w") as f:
                json.dump(traces, f, indent=2)
        except Exception:
            pass


def get_trace(trace_id: str) -> Optional[Dict]:
    if not os.path.exists(TRACE_LOG_FILE):
        return None
    try:
        with open(TRACE_LOG_FILE) as f:
            traces = json.load(f)
        for t in traces:
            if t["trace_id"] == trace_id:
                return t
    except Exception:
        pass
    return None


def get_recent_traces(limit: int = 20) -> List[Dict]:
    if not os.path.exists(TRACE_LOG_FILE):
        return []
    try:
        with open(TRACE_LOG_FILE) as f:
            traces = json.load(f)
        recent = traces[-limit:]
        return [
            {
                "trace_id": t["trace_id"],
                "query": t["query"],
                "timestamp": t["timestamp"],
                "total_duration_seconds": t["total_duration_seconds"],
                "num_steps": len(t["steps"]),
                "flags": t.get("flags", {})
            }
            for t in reversed(recent)
        ]
    except Exception:
        return []
