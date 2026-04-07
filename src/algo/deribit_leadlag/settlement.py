"""
Settlement compatibility classification for Deribit-Polymarket pairs.

Deribit BTC options settle at 08:00 UTC via 30-min TWAP of multi-exchange index.
Polymarket BTC threshold events settle at 16:00 UTC via Binance BTC/USDT 1-min candle Close.

These are NOT the same terminal event. This module classifies compatibility
and gates trading accordingly.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Deribit always settles at 08:00 UTC
DERIBIT_SETTLE_HOUR_UTC = 8
# Expected Polymarket daily BTC resolution time: 16:00 UTC (noon ET)
EXPECTED_POLY_SETTLE_HOUR_UTC = 16


@dataclass
class SettlementMeta:
    """Parsed settlement metadata from Polymarket market description."""

    resolution_time_utc: datetime
    source_exchange: Optional[str]  # "binance" or None if unparseable
    source_symbol: Optional[str]  # "BTC/USDT" or None
    source_timeframe: Optional[str]  # "1m", "1h", or None
    source_price_field: Optional[str]  # "close" or None
    comparator: Optional[str]  # ">" or ">=" or None
    parse_success: bool  # True if all fields extracted without ambiguity


class CompatibilityClass(Enum):
    EXACT_MATCH = "exact_match"  # Same terminal event (none known currently)
    TIME_ADJUSTED = "time_adjusted"  # Timestamp mismatch modeled, source mismatch haircutted
    REJECT = "reject"  # Not eligible for trading


# Regex patterns for parsing Polymarket description
_EXCHANGE_RE = re.compile(r'(?:the\s+)?(\w+)\s+\d+\s*(?:minute|hour)\s+candle', re.IGNORECASE)
_SYMBOL_RE = re.compile(r'(BTC/USDT)', re.IGNORECASE)
_TIMEFRAME_RE = re.compile(r'(\d+)\s*(minute|hour)\s+candle', re.IGNORECASE)
_PRICE_FIELD_RE = re.compile(r'"(Close|Open|High|Low)"', re.IGNORECASE)
_COMPARATOR_HIGHER_RE = re.compile(r'higher\s+than', re.IGNORECASE)
_COMPARATOR_GREATER_RE = re.compile(r'greater\s+than', re.IGNORECASE)
_COMPARATOR_ATLEAST_RE = re.compile(r'at\s+least', re.IGNORECASE)
_HOURLY_RE = re.compile(r'\d+\s*(?:am|pm)\s', re.IGNORECASE)


def parse_settlement_meta(
    description: str,
    resolution_time_utc: datetime,
) -> SettlementMeta:
    """
    Parse settlement metadata from a Polymarket market description.

    Returns SettlementMeta with parse_success=True only if ALL fields
    were extracted without ambiguity.
    """
    if not description:
        return SettlementMeta(
            resolution_time_utc=resolution_time_utc,
            source_exchange=None,
            source_symbol=None,
            source_timeframe=None,
            source_price_field=None,
            comparator=None,
            parse_success=False,
        )

    # Exchange
    exchange_match = _EXCHANGE_RE.search(description)
    source_exchange = exchange_match.group(1).lower() if exchange_match else None

    # Symbol
    symbol_match = _SYMBOL_RE.search(description)
    source_symbol = symbol_match.group(1).upper() if symbol_match else None

    # Timeframe
    tf_match = _TIMEFRAME_RE.search(description)
    source_timeframe = None
    if tf_match:
        num = tf_match.group(1)
        unit = tf_match.group(2).lower()
        if unit == "minute":
            source_timeframe = f"{num}m"
        elif unit == "hour":
            source_timeframe = f"{num}h"

    # Price field
    pf_match = _PRICE_FIELD_RE.search(description)
    source_price_field = pf_match.group(1).lower() if pf_match else None

    # Comparator
    comparator = None
    if _COMPARATOR_HIGHER_RE.search(description):
        comparator = ">"
    elif _COMPARATOR_GREATER_RE.search(description):
        comparator = ">"
    elif _COMPARATOR_ATLEAST_RE.search(description):
        comparator = ">="

    # Parse success requires all fields present
    parse_success = all([
        source_exchange is not None,
        source_symbol is not None,
        source_timeframe is not None,
        source_price_field is not None,
        comparator is not None,
    ])

    return SettlementMeta(
        resolution_time_utc=resolution_time_utc,
        source_exchange=source_exchange,
        source_symbol=source_symbol,
        source_timeframe=source_timeframe,
        source_price_field=source_price_field,
        comparator=comparator,
        parse_success=parse_success,
    )


def classify_compatibility(
    deribit_expiry: date,
    poly_settlement: SettlementMeta,
) -> Tuple[CompatibilityClass, str]:
    """
    Classify settlement compatibility between a Deribit expiry and Polymarket market.

    A market is `time_adjusted` ONLY if ALL of the following hold:
    - Polymarket resolution time is exactly 16:00:00 UTC
    - Deribit expiry is 08:00:00 UTC (always true by convention)
    - Same calendar date
    - Source exchange = Binance
    - Source symbol = BTC/USDT
    - Source timeframe = 1m (1-minute candle)
    - Source price field = close
    - Comparator = strict >
    - Not an hourly market variant
    - Description parsing succeeds with no ambiguity

    Everything else is `reject`.

    Returns:
        (CompatibilityClass, reason_string)
    """
    # Gate 1: parsing must have succeeded
    if not poly_settlement.parse_success:
        return CompatibilityClass.REJECT, "description parsing failed or ambiguous"

    # Gate 2: resolution time must be exactly 16:00:00 UTC
    rt = poly_settlement.resolution_time_utc
    if rt.hour != EXPECTED_POLY_SETTLE_HOUR_UTC or rt.minute != 0 or rt.second != 0:
        return CompatibilityClass.REJECT, f"resolution time {rt.strftime('%H:%M:%S')} UTC != 16:00:00 UTC"

    # Gate 3: same calendar date
    poly_date = rt.date()
    if deribit_expiry != poly_date:
        return CompatibilityClass.REJECT, f"date mismatch: Deribit {deribit_expiry} vs Poly {poly_date}"

    # Gate 4: source exchange must be Binance
    if poly_settlement.source_exchange != "binance":
        return CompatibilityClass.REJECT, f"source exchange '{poly_settlement.source_exchange}' != 'binance'"

    # Gate 5: source symbol must be BTC/USDT
    if poly_settlement.source_symbol != "BTC/USDT":
        return CompatibilityClass.REJECT, f"source symbol '{poly_settlement.source_symbol}' != 'BTC/USDT'"

    # Gate 6: timeframe must be 1m (not hourly)
    if poly_settlement.source_timeframe != "1m":
        return CompatibilityClass.REJECT, f"timeframe '{poly_settlement.source_timeframe}' != '1m'"

    # Gate 7: price field must be close
    if poly_settlement.source_price_field != "close":
        return CompatibilityClass.REJECT, f"price field '{poly_settlement.source_price_field}' != 'close'"

    # Gate 8: comparator must be strict >
    if poly_settlement.comparator != ">":
        return CompatibilityClass.REJECT, f"comparator '{poly_settlement.comparator}' != '>'"

    # All gates passed → time_adjusted
    return CompatibilityClass.TIME_ADJUSTED, (
        "same date, Deribit 08:00 UTC -> Poly 16:00 UTC, "
        "Binance BTC/USDT 1m Close, strict >"
    )
