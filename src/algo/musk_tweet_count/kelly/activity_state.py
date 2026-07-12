"""
Activity-state tracking for maker-quote gating.

Classifies the current moment as quiet / active / storm from the
realtime tracker's dual stream (countable posts AND non-countable reply
activity). The states mirror the 2026-07 markout study, which measured
conditional bin-price drift by exactly these definitions: quiet/earlier
drift is far below half-spread, active runs ~1.6x quiet, storm ~2.2x,
so resting quotes are only safe in quiet state.
"""

import time
from collections import deque
from typing import Deque, Optional


class ActivityStateTracker:
    """Rolling window of post timestamps -> quiet/active/storm state.

    Fail-closed by design:
    - At startup the last-event time is seeded to "now", so the state
      cannot be quiet until a full quiet window has elapsed under
      observation (a restart mid-session must not quote immediately).
    - If tracker polls stop arriving, seconds_since_poll() grows and the
      maker gate must treat the state as unknown (not quiet).
    """

    def __init__(
        self,
        quiet_window_seconds: float = 1200.0,
        storm_window_seconds: float = 1800.0,
        storm_count: int = 5,
    ):
        self.quiet_window_seconds = quiet_window_seconds
        self.storm_window_seconds = storm_window_seconds
        self.storm_count = storm_count

        self._events: Deque[float] = deque()
        self._last_event_ts: float = time.time()  # conservative seed
        self._last_poll_ts: Optional[float] = None

    def note_event(self, ts: float) -> None:
        """Record a post (countable or reply-activity) at unix time ts."""
        self._events.append(ts)
        if ts > self._last_event_ts:
            self._last_event_ts = ts
        self._prune()

    def note_poll(self, now: Optional[float] = None) -> None:
        """Record that the realtime tracker completed a successful poll."""
        self._last_poll_ts = now if now is not None else time.time()

    def seconds_since_poll(self, now: Optional[float] = None) -> Optional[float]:
        """Age of the last successful poll, or None if never polled."""
        if self._last_poll_ts is None:
            return None
        now = now if now is not None else time.time()
        return now - self._last_poll_ts

    def last_event_age(self, now: Optional[float] = None) -> float:
        """Seconds since the most recent observed post (or startup seed)."""
        now = now if now is not None else time.time()
        return now - self._last_event_ts

    def state(self, now: Optional[float] = None) -> str:
        """Current activity state: "quiet", "active", or "storm"."""
        now = now if now is not None else time.time()
        self._prune(now)
        recent = sum(1 for ts in self._events if ts >= now - self.storm_window_seconds)
        if recent >= self.storm_count:
            return "storm"
        if now - self._last_event_ts < self.quiet_window_seconds:
            return "active"
        return "quiet"

    def _prune(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        cutoff = now - self.storm_window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def to_summary(self) -> dict:
        now = time.time()
        return {
            "state": self.state(now),
            "last_event_age_seconds": round(self.last_event_age(now), 1),
            "seconds_since_poll": (
                round(self.seconds_since_poll(now), 1)
                if self._last_poll_ts is not None
                else None
            ),
            "events_in_storm_window": len(self._events),
        }
