"""Discover and parse Polymarket BTC binary threshold events."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import requests

from src.const import GAMMA_API_URL

from .settlement import SettlementMeta, parse_settlement_meta

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
    description: str = ""  # Raw description text from Gamma API
    settlement: Optional[SettlementMeta] = None  # Parsed settlement metadata


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


BTC_TAG_ID = 235


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
        description = mkt.get("description", "")

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
                description=description,
                settlement=parse_settlement_meta(description, resolution_time) if description else None,
            )
        )

    return markets


def discover_btc_threshold_markets() -> List[ThresholdMarket]:
    """
    Discover active Polymarket BTC binary threshold events.

    Uses tag_id=235 ("Bitcoin") to fetch all BTC events in one call,
    then filters to daily "Bitcoin above ___" events.
    """
    all_markets = []
    offset = 0
    limit = 100

    while True:
        try:
            resp = requests.get(
                f"{GAMMA_API_URL}/events",
                params={
                    "tag_id": BTC_TAG_ID,
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "offset": offset,
                },
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            logger.error(f"Gamma API error at offset {offset}: {e}")
            break

        if not events:
            break

        for event in events:
            all_markets.extend(_parse_event_markets(event))

        if len(events) < limit:
            break
        offset += limit

    # Deduplicate by condition_id
    seen = set()
    unique = []
    for m in all_markets:
        if m.condition_id not in seen:
            seen.add(m.condition_id)
            unique.append(m)

    logger.info(
        f"Discovered {len(unique)} BTC threshold markets "
        f"across {len(set(m.expiry_date for m in unique))} dates"
    )
    return unique


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
