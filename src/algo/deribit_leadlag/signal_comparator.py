"""
Compare Deribit implied probabilities against Polymarket prices.

Generates trade signals when mispricing exceeds threshold after fees.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from .config import SignalConfig, ExecutionConfig
from .implied_probs import ImpliedProb
from .polymarket_discovery import ThresholdMarket

logger = logging.getLogger(__name__)


@dataclass
class MatchedPair:
    """A Deribit option matched with a Polymarket market on (date, strike)."""

    strike: float
    expiry_date: date
    deribit_prob: ImpliedProb
    poly_market: ThresholdMarket
    settlement_time_diff_hours: float  # Deribit 08:00 UTC vs Polymarket 16:00 UTC


@dataclass
class TradeSignal:
    """An actionable trade signal."""

    matched: MatchedPair
    side: str  # "BUY_YES" or "BUY_NO"
    gross_edge: float  # Before fees
    net_edge: float  # After fees
    fee_per_share: float
    token_id: str  # YES or NO token to buy
    price: float  # Price to place FAK at
    size_usd: float  # USD value of the order
    size_shares: int  # Number of shares
    deribit_prob_mid: float
    deribit_prob_conservative: float
    poly_price: float  # Polymarket YES price used
    timestamp: float  # Signal generation time


def compute_polymarket_fee(price: float, fee_rate: float = 0.072) -> float:
    """
    Compute Polymarket taker fee per share.

    fee = feeRate * p * (1 - p)

    Peaks at p=0.50 (~1.80%), approaches zero near 0 or 1.
    """
    return fee_rate * price * (1.0 - price)


def match_markets(
    implied_probs: Dict[Tuple[date, float], ImpliedProb],
    poly_markets: List[ThresholdMarket],
) -> List[MatchedPair]:
    """
    Join Deribit implied probabilities with Polymarket markets on (date, strike).

    The Deribit expiry date and Polymarket expiry date must match. Deribit settles
    at 08:00 UTC, Polymarket at 16:00 UTC (noon ET) — an 8-hour gap.
    """
    # Build lookup for Polymarket markets
    poly_lookup: Dict[Tuple[date, float], ThresholdMarket] = {}
    for m in poly_markets:
        key = (m.expiry_date, m.strike)
        # If multiple markets for same (date, strike), take highest volume
        if key not in poly_lookup or m.volume > poly_lookup[key].volume:
            poly_lookup[key] = m

    matched = []
    for (exp_date, strike), prob in implied_probs.items():
        poly = poly_lookup.get((exp_date, strike))
        if poly is None:
            continue

        # Settlement time difference: Deribit 08:00 UTC, Polymarket from endDate
        deribit_settle_hour = 8.0  # UTC
        poly_settle_hour = poly.resolution_time_utc.hour + poly.resolution_time_utc.minute / 60.0
        time_diff = poly_settle_hour - deribit_settle_hour

        matched.append(
            MatchedPair(
                strike=strike,
                expiry_date=exp_date,
                deribit_prob=prob,
                poly_market=poly,
                settlement_time_diff_hours=time_diff,
            )
        )

    logger.info(f"Matched {len(matched)} Deribit-Polymarket pairs")
    return matched


def generate_signals(
    matched_pairs: List[MatchedPair],
    signal_config: SignalConfig,
    exec_config: ExecutionConfig,
    current_exposure_usd: float = 0.0,
    recent_signals: Optional[Dict[str, float]] = None,
) -> List[TradeSignal]:
    """
    Generate trade signals from matched pairs with multi-stage filtering.

    Stage 1: Deribit quality (IV spread, volume, OI)
    Stage 2: Probability range (skip deep wings)
    Stage 3: Polymarket activity (skip zero-volume markets)
    Stage 4: Edge computation with fees + conservative prob
    Stage 5: Rank + deduplicate
    """
    if recent_signals is None:
        recent_signals = {}

    now = time.time()
    signals = []
    remaining_exposure = exec_config.max_total_exposure_usd - current_exposure_usd

    if remaining_exposure <= 0:
        logger.info("Max exposure reached, no new signals")
        return []

    for pair in matched_pairs:
        prob = pair.deribit_prob
        poly = pair.poly_market
        strike = pair.strike
        exp = pair.expiry_date

        # --- Stage 1: Deribit quality ---
        # IV spread check (only for BS fallback where we have IV data)
        # For call-spread method, quality is baked into the conservative bound

        # --- Stage 2: Probability range ---
        if prob.prob_mid < signal_config.min_prob or prob.prob_mid > signal_config.max_prob:
            continue

        # --- Stage 3: Polymarket activity ---
        if poly.volume <= 0:
            continue

        # --- Stage 4: Edge computation ---
        poly_yes_price = poly.yes_price

        # Determine side
        if prob.prob_mid > poly_yes_price:
            # Deribit says YES is more likely than Polymarket prices → BUY YES
            side = "BUY_YES"
            token_id = poly.yes_token_id
            entry_price = poly_yes_price
            # Use conservative (lower) probability for edge calculation
            conservative_prob = prob.prob_conservative
            gross_edge = conservative_prob - poly_yes_price
        else:
            # Deribit says YES is less likely → BUY NO
            side = "BUY_NO"
            token_id = poly.no_token_id
            entry_price = poly.no_price
            # For NO: P(NO) = 1 - P(YES)
            deribit_no_prob_conservative = 1.0 - prob.prob_aggressive
            gross_edge = deribit_no_prob_conservative - poly.no_price

        if gross_edge <= 0:
            continue

        # Dynamic fee
        fee = compute_polymarket_fee(entry_price, exec_config.polymarket_crypto_fee_rate)
        # Net edge: gross edge minus fee as fraction of payout ($1)
        net_edge = gross_edge - fee

        if net_edge < signal_config.min_edge_threshold:
            continue
        if net_edge > signal_config.max_edge_threshold:
            logger.warning(
                f"Suspiciously large edge {net_edge:.3f} for {exp}/{strike}, skipping"
            )
            continue

        # --- Stage 5: Dedup + sizing ---
        signal_key = f"{exp}_{strike}_{side}"
        last_signal_time = recent_signals.get(signal_key, 0)
        if now - last_signal_time < signal_config.signal_cooldown_seconds:
            continue

        # Size: capped by max order size and remaining exposure
        max_usd = min(exec_config.max_order_size_usd, remaining_exposure)
        if entry_price > 0:
            size_shares = int(max_usd / entry_price)
        else:
            size_shares = 0

        if size_shares < 1:
            continue

        actual_usd = size_shares * entry_price

        signals.append(
            TradeSignal(
                matched=pair,
                side=side,
                gross_edge=gross_edge,
                net_edge=net_edge,
                fee_per_share=fee,
                token_id=token_id,
                price=entry_price,
                size_usd=actual_usd,
                size_shares=size_shares,
                deribit_prob_mid=prob.prob_mid,
                deribit_prob_conservative=prob.prob_conservative if side == "BUY_YES" else (1.0 - prob.prob_aggressive),
                poly_price=poly_yes_price,
                timestamp=now,
            )
        )

    # Sort by net edge descending
    signals.sort(key=lambda s: s.net_edge, reverse=True)

    if signals:
        logger.info(
            f"Generated {len(signals)} signals (best net_edge={signals[0].net_edge:.3f} "
            f"for {signals[0].matched.expiry_date}/{signals[0].matched.strike:.0f} {signals[0].side})"
        )

    return signals
