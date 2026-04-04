"""Fetch BTC option data from Deribit public API."""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Month abbreviation map for Deribit instrument names (e.g., "5APR26")
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Regex for instrument name: BTC-5APR26-85000-C
_INSTRUMENT_RE = re.compile(
    r"^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-(C|P)$"
)


@dataclass
class DeribitOption:
    instrument_name: str
    expiry_date: date
    strike: float
    option_type: str  # "C" or "P"
    mark_iv: float  # Annualized IV as decimal (e.g., 0.65 for 65%)
    underlying_price: float  # Forward price in USD
    mark_price_btc: float  # Mark price in BTC
    bid_price_btc: Optional[float]  # Best bid in BTC (None if no bids)
    ask_price_btc: Optional[float]  # Best ask in BTC (None if no asks)
    bid_iv: Optional[float]  # Bid IV as decimal
    ask_iv: Optional[float]  # Ask IV as decimal
    volume_24h: float
    open_interest: float

    @property
    def mark_price_usd(self) -> float:
        return self.mark_price_btc * self.underlying_price

    @property
    def bid_price_usd(self) -> Optional[float]:
        if self.bid_price_btc is not None:
            return self.bid_price_btc * self.underlying_price
        return None

    @property
    def ask_price_usd(self) -> Optional[float]:
        if self.ask_price_btc is not None:
            return self.ask_price_btc * self.underlying_price
        return None

    @property
    def mid_price_usd(self) -> Optional[float]:
        if self.bid_price_usd is not None and self.ask_price_usd is not None:
            return (self.bid_price_usd + self.ask_price_usd) / 2.0
        return self.mark_price_usd


def parse_instrument_name(name: str) -> Optional[Tuple[date, float, str]]:
    """
    Parse Deribit instrument name like 'BTC-5APR26-85000-C'.

    Returns (expiry_date, strike, option_type) or None if unparseable.
    """
    m = _INSTRUMENT_RE.match(name)
    if not m:
        return None

    day = int(m.group(1))
    month_str = m.group(2)
    year_short = int(m.group(3))
    strike = float(m.group(4))
    opt_type = m.group(5)

    month = _MONTH_MAP.get(month_str)
    if month is None:
        return None

    year = 2000 + year_short
    try:
        expiry = date(year, month, day)
    except ValueError:
        return None

    return expiry, strike, opt_type


def fetch_btc_options_summary(
    base_url: str = "https://www.deribit.com/api/v2/public",
) -> List[DeribitOption]:
    """
    Fetch all BTC option summaries from Deribit in a single API call.

    Returns parsed DeribitOption objects for all active BTC options.
    """
    url = f"{base_url}/get_book_summary_by_currency"
    params = {"currency": "BTC", "kind": "option"}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "result" not in data:
        logger.error(f"Deribit API error: {data}")
        return []

    options = []
    for item in data["result"]:
        parsed = parse_instrument_name(item["instrument_name"])
        if parsed is None:
            continue

        expiry_date, strike, opt_type = parsed

        # IV: Deribit returns as percentage (65.0 for 65%), convert to decimal
        mark_iv_raw = item.get("mark_iv")
        bid_iv_raw = item.get("bid_iv")
        ask_iv_raw = item.get("ask_iv")

        mark_iv = mark_iv_raw / 100.0 if mark_iv_raw and mark_iv_raw > 0 else 0.0
        bid_iv = bid_iv_raw / 100.0 if bid_iv_raw and bid_iv_raw > 0 else None
        ask_iv = ask_iv_raw / 100.0 if ask_iv_raw and ask_iv_raw > 0 else None

        # Prices: in BTC
        mark_price = item.get("mark_price", 0.0) or 0.0
        bid_price = item.get("bid_price")
        ask_price = item.get("ask_price")

        options.append(
            DeribitOption(
                instrument_name=item["instrument_name"],
                expiry_date=expiry_date,
                strike=strike,
                option_type=opt_type,
                mark_iv=mark_iv,
                underlying_price=item.get("underlying_price", 0.0),
                mark_price_btc=mark_price,
                bid_price_btc=bid_price if bid_price and bid_price > 0 else None,
                ask_price_btc=ask_price if ask_price and ask_price > 0 else None,
                bid_iv=bid_iv,
                ask_iv=ask_iv,
                volume_24h=item.get("volume", 0.0) or 0.0,
                open_interest=item.get("open_interest", 0.0) or 0.0,
            )
        )

    logger.info(f"Fetched {len(options)} BTC options from Deribit")
    return options


def fetch_spot_price(
    base_url: str = "https://www.deribit.com/api/v2/public",
) -> Optional[float]:
    """Fetch current BTC spot (index) price from Deribit."""
    url = f"{base_url}/get_index_price"
    params = {"index_name": "btc_usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "result" in data:
        return data["result"].get("index_price")
    return None


def filter_calls_by_expiry(
    options: List[DeribitOption],
    min_date: date,
    max_date: date,
) -> List[DeribitOption]:
    """Filter to call options within the given date range (inclusive)."""
    return [
        opt
        for opt in options
        if opt.option_type == "C"
        and min_date <= opt.expiry_date <= max_date
    ]
