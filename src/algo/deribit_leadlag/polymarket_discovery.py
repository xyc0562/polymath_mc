"""Discover and parse Polymarket BTC binary threshold events."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import requests

from src.const import GAMMA_API_URL

logger = logging.getLogger(__name__)

# Match "Bitcoin above ___" events (noon ET daily resolution)
# The title uses "___" as placeholder; strikes are in individual market titles
_ABOVE_EVENT_RE = re.compile(
    r"bitcoin\s+above\s+.+\s+on\s+\w+\s+\d+\??$", re.IGNORECASE
)
_PRICE_EXTRACT_RE = re.compile(r"[\$]?([\d,]+)")


@dataclass
class ThresholdMarket:
    """A single Polymarket binary threshold market ("Bitcoin above $X on date Y")."""

    condition_id: str
    question: str  # Full question text
    strike: float  # e.g., 66000.0
    expiry_date: date  # Date portion of resolution
    resolution_time_utc: datetime  # Exact resolution time in UTC
    yes_token_id: str
    no_token_id: str
    yes_price: float  # Last/indicative YES price [0, 1]
    no_price: float  # Last/indicative NO price [0, 1]
    event_id: str
    volume: float  # Total volume in USD


def _extract_strike_from_title(title: str) -> Optional[float]:
    """
    Extract a dollar-denominated strike price from a market title.

    Handles: "$66,000", "66000", "$66,000.00"
    """
    m = _PRICE_EXTRACT_RE.search(title)
    if m:
        price_str = m.group(1).replace(",", "")
        try:
            return float(price_str)
        except ValueError:
            return None
    return None


def _parse_resolution_time(end_date_str: str) -> Optional[datetime]:
    """Parse Gamma API endDate (ISO 8601) to UTC datetime."""
    if not end_date_str:
        return None
    try:
        # Handle "2026-04-06T16:00:00Z" and similar
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _is_btc_above_event(title: str) -> bool:
    """
    Check if an event title is a BTC binary threshold event.

    Matches: "Bitcoin above ___ on April 9?"
    Does NOT match: "Bitcoin above ___ on April 4, 10AM ET?" (hourly variants)
    """
    t = title.lower().strip().rstrip("?")
    # Must contain "bitcoin above" and "on <month> <day>" without a time component
    if "bitcoin above" not in t:
        return False
    # Exclude hourly variants that contain time like "10AM", "9AM", "7AM"
    if re.search(r"\d+\s*(?:am|pm)\s", t, re.IGNORECASE):
        return False
    # Must have "on <word> <digit>" pattern (e.g., "on April 9")
    if not re.search(r"on\s+\w+\s+\d+", t):
        return False
    return True


def _parse_outcome_prices(market: dict) -> tuple:
    """Extract YES/NO prices from market data."""
    outcome_prices = market.get("outcomePrices")
    if outcome_prices:
        if isinstance(outcome_prices, str):
            import json
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = None

    yes_price = 0.0
    no_price = 0.0
    if outcome_prices and len(outcome_prices) >= 2:
        yes_price = float(outcome_prices[0])
        no_price = float(outcome_prices[1])

    return yes_price, no_price


def _fetch_event_by_id(event_id: int) -> Optional[dict]:
    """Fetch a single event by ID from Gamma API."""
    try:
        resp = requests.get(f"{GAMMA_API_URL}/events/{event_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _parse_event_markets(event: dict) -> List[ThresholdMarket]:
    """Parse markets from a single event dict."""
    title = event.get("title", "")
    if not _is_btc_above_event(title):
        return []

    event_id = str(event.get("id", ""))
    end_date_str = event.get("endDate", "")
    resolution_time = _parse_resolution_time(end_date_str)
    if resolution_time is None:
        return []

    # Skip closed events
    if event.get("closed"):
        return []

    expiry = resolution_time.date()
    markets = []

    for mkt in event.get("markets", []):
        question = mkt.get("question", "") or mkt.get("groupItemTitle", "")
        strike = _extract_strike_from_title(question)

        if strike is None:
            group_title = mkt.get("groupItemTitle", "")
            strike = _extract_strike_from_title(group_title)

        if strike is None:
            continue

        tokens = mkt.get("clobTokenIds")
        if not tokens:
            continue

        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except (json.JSONDecodeError, TypeError):
                continue

        if len(tokens) < 2:
            continue

        yes_price, no_price = _parse_outcome_prices(mkt)

        markets.append(
            ThresholdMarket(
                condition_id=mkt.get("conditionId", ""),
                question=question,
                strike=strike,
                expiry_date=expiry,
                resolution_time_utc=resolution_time,
                yes_token_id=tokens[0],
                no_token_id=tokens[1],
                yes_price=yes_price,
                no_price=no_price,
                event_id=event_id,
                volume=float(mkt.get("volume", 0) or 0),
            )
        )

    return markets


def discover_btc_threshold_markets(
    seed_event_ids: Optional[List[int]] = None,
    scan_range: int = 200,
) -> List[ThresholdMarket]:
    """
    Discover active Polymarket BTC binary threshold events.

    Strategy: Start from seed event IDs (known recent BTC events),
    then scan nearby IDs to find more. This is necessary because the
    Gamma API pagination doesn't reliably reach high-ID events.

    Args:
        seed_event_ids: Known BTC event IDs to start from. If None, uses defaults.
        scan_range: How many IDs above/below seeds to scan.

    Returns:
        List of ThresholdMarket objects, one per strike per date.
    """
    # Default seeds: known BTC "above" event IDs from April 2026
    if seed_event_ids is None:
        seed_event_ids = [
            336429,  # Bitcoin above ___ on April 9
            340006,  # Bitcoin above ___ on April 10
        ]

    # Collect all event IDs to check
    ids_to_check = set()

    # Add seeds
    for sid in seed_event_ids:
        ids_to_check.add(sid)

    # Scan around seeds to find nearby events (new dates appear with close IDs)
    if seed_event_ids:
        max_seed = max(seed_event_ids)
        # Scan forward (future dates) and a bit backward
        for offset in range(-50, scan_range):
            ids_to_check.add(max_seed + offset)

    # Also try pagination (descending order) for any we might miss
    try:
        resp = requests.get(
            f"{GAMMA_API_URL}/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": 0,
                "order": "id",
                "ascending": "false",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            for event in resp.json():
                title = event.get("title", "")
                if _is_btc_above_event(title):
                    ids_to_check.add(event["id"])
    except Exception as e:
        logger.warning(f"Pagination fetch failed: {e}")

    # Fetch and parse each candidate event
    all_markets = []
    seen_event_ids = set()

    for eid in sorted(ids_to_check):
        event = _fetch_event_by_id(eid)
        if event is None:
            continue

        actual_id = event.get("id")
        if actual_id in seen_event_ids:
            continue
        seen_event_ids.add(actual_id)

        markets = _parse_event_markets(event)
        all_markets.extend(markets)

    # Deduplicate by condition_id
    seen_conditions = set()
    unique_markets = []
    for m in all_markets:
        if m.condition_id not in seen_conditions:
            seen_conditions.add(m.condition_id)
            unique_markets.append(m)

    logger.info(
        f"Discovered {len(unique_markets)} BTC threshold markets "
        f"across {len(set(m.expiry_date for m in unique_markets))} dates "
        f"(scanned {len(ids_to_check)} event IDs)"
    )
    return unique_markets


def build_target_strikes(
    markets: List[ThresholdMarket],
) -> Dict[tuple, datetime]:
    """
    Build target strikes map for implied probability computation.

    Returns:
        Dict mapping (expiry_date, strike) -> Polymarket resolution time (UTC).
    """
    targets = {}
    for m in markets:
        key = (m.expiry_date, m.strike)
        targets[key] = m.resolution_time_utc
    return targets
