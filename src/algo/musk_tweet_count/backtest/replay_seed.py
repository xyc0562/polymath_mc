"""
Helpers for seeded backtest replays.

These loaders are opt-in and let the unified backtest start from an existing
portfolio state instead of a flat portfolio.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class ReplaySeedPosition:
    """Seed position for a single bin."""

    yes_shares: float = 0.0
    yes_avg_cost: float = 0.0
    no_shares: float = 0.0
    no_avg_cost: float = 0.0

    @property
    def collateral_used(self) -> float:
        """Collateral implied by the seeded position."""
        return (self.yes_shares * self.yes_avg_cost) + (self.no_shares * self.no_avg_cost)


@dataclass
class ReplaySeedState:
    """Seed state for resuming a backtest from a live snapshot."""

    available_capital: Optional[float] = None
    event_budget: Optional[float] = None
    positions: Dict[int, ReplaySeedPosition] = field(default_factory=dict)
    snapshot_timestamp: Optional[int] = None
    source: str = ""


def load_seed_state(path: Path) -> ReplaySeedState:
    """
    Load replay seed state from a JSON file.

    Supported formats:
    - {"available_capital": 123.4, "positions": {"13": {"yes_shares": 10, ...}}}
    - {"positions": [{"bin_index": 13, "yes_shares": 10, ...}, ...]}
    """
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    positions: Dict[int, ReplaySeedPosition] = {}
    raw_positions = payload.get("positions", {})

    if isinstance(raw_positions, dict):
        items = []
        for raw_bin, raw_pos in raw_positions.items():
            items.append({"bin_index": raw_bin, **raw_pos})
    elif isinstance(raw_positions, list):
        items = raw_positions
    else:
        raise ValueError(f"Unsupported positions format in {path}")

    for item in items:
        if "bin_index" not in item:
            raise ValueError(f"Missing bin_index in replay seed position from {path}")
        bin_index = int(item["bin_index"])
        positions[bin_index] = ReplaySeedPosition(
            yes_shares=float(item.get("yes_shares", 0.0)),
            yes_avg_cost=float(item.get("yes_avg_cost", 0.0)),
            no_shares=float(item.get("no_shares", 0.0)),
            no_avg_cost=float(item.get("no_avg_cost", 0.0)),
        )

    snapshot_timestamp = payload.get("snapshot_timestamp")
    if snapshot_timestamp is not None:
        snapshot_timestamp = int(snapshot_timestamp)

    return ReplaySeedState(
        available_capital=_optional_float(payload.get("available_capital")),
        event_budget=_optional_float(payload.get("event_budget")),
        positions=positions,
        snapshot_timestamp=snapshot_timestamp,
        source=str(path),
    )


def extract_seed_state_from_log(
    log_path: Path,
    event_name: str,
    snapshot_timestamp: int,
) -> ReplaySeedState:
    """
    Extract a replay seed state from a production log snapshot.

    This parser targets the structured Kelly sync block in the production log:
    - "[Event] Event capital: budget=$..., invested=$..., available=$..."
    - "[Event][KELLY] iter=... Synced from API: ..."
    - "  Bin X: YES=... NO=... cost=$..."
    """
    target_prefix = datetime.fromtimestamp(snapshot_timestamp, tz=timezone.utc).strftime("%y-%m-%d %H:%M:%S")
    capital_re = re.compile(
        r"\[(?P<event>.+?)\] Event capital: budget=\$(?P<budget>[0-9.]+), "
        r"invested=\$(?P<invested>[0-9.]+), available=\$(?P<available>[0-9.]+)"
    )
    position_re = re.compile(
        r"\s+Bin (?P<bin>\d+): YES=(?P<yes>[0-9.]+) NO=(?P<no>[0-9.]+) cost=\$(?P<cost>[0-9.]+)"
    )

    available_capital: Optional[float] = None
    event_budget: Optional[float] = None
    positions: Dict[int, ReplaySeedPosition] = {}
    capture_positions = False
    saw_target_timestamp = False

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.startswith(target_prefix):
                if saw_target_timestamp and capture_positions:
                    break
                continue

            saw_target_timestamp = True
            content = _strip_log_prefix(raw_line)

            capital_match = capital_re.search(content)
            if capital_match and capital_match.group("event") == event_name:
                available_capital = float(capital_match.group("available"))
                event_budget = float(capital_match.group("budget"))
                continue

            if f"[{event_name}][KELLY]" in content and "Synced from API" in content:
                capture_positions = True
                continue

            if not capture_positions:
                continue

            position_match = position_re.match(content)
            if position_match:
                bin_index = int(position_match.group("bin"))
                yes_shares = float(position_match.group("yes"))
                no_shares = float(position_match.group("no"))
                cost = float(position_match.group("cost"))
                positions[bin_index] = _position_from_log_snapshot(
                    yes_shares=yes_shares,
                    no_shares=no_shares,
                    cost=cost,
                    bin_index=bin_index,
                    event_name=event_name,
                )
                continue

            if content == "Multi-bin Kelly state:":
                continue
            if content.startswith("  Capital:"):
                continue
            if content.startswith("  Top model probability bins:"):
                break
            if content.startswith("Candidate rejection reasons:"):
                break
            if not content.startswith("  "):
                break

    if available_capital is None and not positions:
        raise ValueError(
            f"Could not find replay seed block for event '{event_name}' at "
            f"{target_prefix} in {log_path}"
        )

    return ReplaySeedState(
        available_capital=available_capital,
        event_budget=event_budget,
        positions=positions,
        snapshot_timestamp=snapshot_timestamp,
        source=f"{log_path} @ {target_prefix} [{event_name}]",
    )


def _optional_float(value) -> Optional[float]:
    """Convert value to float unless it is missing."""
    if value is None:
        return None
    return float(value)


def _strip_log_prefix(line: str) -> str:
    """Remove the timestamp/log-level prefix from a log line."""
    parts = line.split("]: ", 1)
    if len(parts) == 2:
        return parts[1].rstrip("\n")
    return line.rstrip("\n")


def _position_from_log_snapshot(
    yes_shares: float,
    no_shares: float,
    cost: float,
    bin_index: int,
    event_name: str,
) -> ReplaySeedPosition:
    """Infer a one-sided seed position from the production Kelly state log."""
    if yes_shares > 0 and no_shares > 0:
        raise ValueError(
            f"Cannot infer separate YES/NO cost bases for bin {bin_index} in "
            f"'{event_name}' from a combined log cost=${cost:.2f}"
        )

    if yes_shares > 0:
        return ReplaySeedPosition(
            yes_shares=yes_shares,
            yes_avg_cost=(cost / yes_shares) if yes_shares > 0 else 0.0,
        )

    if no_shares > 0:
        return ReplaySeedPosition(
            no_shares=no_shares,
            no_avg_cost=(cost / no_shares) if no_shares > 0 else 0.0,
        )

    logger.debug("Ignoring empty seeded position for bin %s in %s", bin_index, event_name)
    return ReplaySeedPosition()
