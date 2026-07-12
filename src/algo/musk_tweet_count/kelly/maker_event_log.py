"""
Durable, structured event log for maker mode.

Every maker event — quote placed, gate open/close, would-fill (shadow),
forward markout sample, unfilled expiry — is appended as one JSON object
per line to a per-event JSONL file. This is the machine-readable record
the markout study and coverage analysis are built from; the human-facing
logger lines remain for live monitoring.

Writes are flushed per line so the record survives a hard process exit,
and the log is a no-op (no file opened) when no directory is configured,
keeping unit tests that construct a QuoteManager hermetic by default.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MakerEventLog:
    def __init__(self, directory: Optional[str], event_name: str, mode: str):
        self.event_name = event_name
        self.mode = mode
        self._fh = None
        self._path: Optional[Path] = None
        if not directory:
            return
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_name).strip("_") or "event"
        self._path = Path(directory) / f"{safe}.jsonl"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")
        except OSError as e:
            logger.error(
                f"[{event_name}][MAKER] could not open event log {self._path}: {e}"
            )
            self._fh = None

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def emit(self, type: str, now: Optional[float] = None, **fields) -> None:
        """Append one event record. Never raises."""
        if self._fh is None:
            return
        record = {
            "ts": now if now is not None else time.time(),
            "event": self.event_name,
            "mode": self.mode,
            "type": type,
        }
        record.update(fields)
        try:
            self._fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            self._fh.flush()
        except (OSError, TypeError) as e:
            logger.error(f"[{self.event_name}][MAKER] event log write failed ({type}): {e}")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
