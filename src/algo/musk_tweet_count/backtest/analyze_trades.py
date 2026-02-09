"""
Analyze backtest trades by days-to-settlement and buy price.

Runs the backtest with the same configuration as run_backtest.py, then performs
per-trade P&L analysis using FIFO matching to determine whether each buy trade
was closed early or held to settlement.

Results are bucketed by:
  1. Days before settlement when the trade was placed
  2. Buy price range at time of purchase

Usage:
    # Quick analysis with asymmetric+bucket (same config as run_backtest.py)
    python -m src.algo.musk_tweet_count.backtest.analyze_trades \
        --unified --quick --projection asymmetric --intraday-mode bucket

    # With date filter
    python -m src.algo.musk_tweet_count.backtest.analyze_trades \
        --unified --quick --projection asymmetric --intraday-mode bucket \
        --start-date 2026-01-01

    # Full (non-quick) analysis
    python -m src.algo.musk_tweet_count.backtest.analyze_trades \
        --unified --projection gamma --intraday-mode ridge
"""

import argparse
import logging
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .unified_runner import (
    UnifiedBacktestRunner,
    UnifiedBacktestConfig,
    Trade,
    BacktestResult,
)
from .run_backtest import (
    filter_events_by_date,
    filter_events_by_duration,
    run_multiple_backtests,
)
from ..kelly.config import (
    KellyConfig,
    EdgeBufferConfig,
    AdaptiveDeltaConfig,
    RateLimitConfig,
    CollateralConfig,
    EventTradingRulesConfig,
)


# Setup logging
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


# ── Price buckets ──────────────────────────────────────────────────────────────

PRICE_BUCKETS = [
    (0.00, 0.03),
    (0.03, 0.05),
    (0.05, 0.10),
    (0.10, 0.30),
    (0.30, 0.50),
    (0.50, 0.70),
    (0.70, 0.90),
    (0.90, 1.00),
]


def get_price_bucket(price: float) -> int:
    """Return index into PRICE_BUCKETS for a given price."""
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        if i == 0 and price <= hi:
            return i
        if lo < price <= hi:
            return i
    return len(PRICE_BUCKETS) - 1


def price_bucket_label(idx: int) -> str:
    lo, hi = PRICE_BUCKETS[idx]
    return f"{lo:.2f}-{hi:.2f}"


# ── Resolved trade ─────────────────────────────────────────────────────────────

@dataclass
class ResolvedTrade:
    """A buy trade matched with its resolution (sell or settlement)."""
    event_name: str
    buy_datetime: datetime
    buy_price: float
    buy_size: float
    buy_side: str        # "YES" or "NO"
    bin_index: int
    days_to_settlement: int
    price_bucket: int
    resolution: str      # "settlement_win", "settlement_loss", "closed_early"
    pnl: float
    sell_price: Optional[float] = None


# ── FIFO matching ──────────────────────────────────────────────────────────────

def analyze_event_trades(
    result: BacktestResult,
    settlement_date: date,
) -> List[ResolvedTrade]:
    """
    Analyze trades from a single event using FIFO matching.

    For each BUY trade, determines whether it was closed early (matched to a
    subsequent SELL) or held to settlement (win or loss based on winner bin).
    """
    resolved: List[ResolvedTrade] = []
    est_tz = ZoneInfo("America/New_York")

    # FIFO queues keyed by (bin_index, side)
    open_buys: Dict[Tuple[int, str], deque] = defaultdict(deque)

    sorted_trades = sorted(result.trades, key=lambda t: t.timestamp)

    for trade in sorted_trades:
        bin_idx = trade.bin_index
        size = trade.size
        price = trade.price
        dt = trade.datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        days_to_settle = max(0, (settlement_date - dt.astimezone(est_tz).date()).days)

        if trade.side in ("BUY_YES", "BUY_NO"):
            side = "YES" if trade.side == "BUY_YES" else "NO"
            open_buys[(bin_idx, side)].append({
                "event_name": result.event_name,
                "datetime": dt,
                "price": price,
                "remaining": size,
                "side": side,
                "bin_index": bin_idx,
                "days_to_settlement": days_to_settle,
                "price_bucket": get_price_bucket(price),
            })

        elif trade.side in ("SELL_YES", "SELL_NO"):
            side = "YES" if trade.side == "SELL_YES" else "NO"
            key = (bin_idx, side)
            sell_remaining = size

            while sell_remaining > 1e-9 and open_buys[key]:
                buy = open_buys[key][0]
                match_size = min(sell_remaining, buy["remaining"])
                pnl = (price - buy["price"]) * match_size

                resolved.append(ResolvedTrade(
                    event_name=buy["event_name"],
                    buy_datetime=buy["datetime"],
                    buy_price=buy["price"],
                    buy_size=match_size,
                    buy_side=buy["side"],
                    bin_index=buy["bin_index"],
                    days_to_settlement=buy["days_to_settlement"],
                    price_bucket=buy["price_bucket"],
                    resolution="closed_early",
                    pnl=pnl,
                    sell_price=price,
                ))

                buy["remaining"] -= match_size
                sell_remaining -= match_size
                if buy["remaining"] <= 1e-9:
                    open_buys[key].popleft()

    # Settle remaining open positions
    winner_bin = result.winner_bin
    for (bin_idx, side), queue in open_buys.items():
        for buy in queue:
            remaining = buy["remaining"]
            if remaining <= 1e-9:
                continue

            if side == "YES":
                won = bin_idx == winner_bin
            else:
                won = bin_idx != winner_bin

            if won:
                pnl = (1.0 - buy["price"]) * remaining
                resolution = "settlement_win"
            else:
                pnl = -buy["price"] * remaining
                resolution = "settlement_loss"

            resolved.append(ResolvedTrade(
                event_name=buy["event_name"],
                buy_datetime=buy["datetime"],
                buy_price=buy["price"],
                buy_size=remaining,
                buy_side=buy["side"],
                bin_index=buy["bin_index"],
                days_to_settlement=buy["days_to_settlement"],
                price_bucket=buy["price_bucket"],
                resolution=resolution,
                pnl=pnl,
            ))

    return resolved


# ── Bucket statistics ──────────────────────────────────────────────────────────

@dataclass
class BucketStats:
    settle_win_count: int = 0
    settle_win_size: float = 0.0
    settle_win_cost: float = 0.0
    settle_win_pnl: float = 0.0

    settle_loss_count: int = 0
    settle_loss_size: float = 0.0
    settle_loss_cost: float = 0.0
    settle_loss_pnl: float = 0.0

    closed_count: int = 0
    closed_size: float = 0.0
    closed_cost: float = 0.0
    closed_pnl: float = 0.0
    closed_wins: int = 0  # closed with positive P&L

    @property
    def total_count(self):
        return self.settle_win_count + self.settle_loss_count + self.closed_count

    @property
    def total_cost(self):
        return self.settle_win_cost + self.settle_loss_cost + self.closed_cost

    @property
    def total_pnl(self):
        return self.settle_win_pnl + self.settle_loss_pnl + self.closed_pnl

    @property
    def total_wins(self):
        return self.settle_win_count + self.closed_wins

    def add(self, trade: ResolvedTrade):
        cost = trade.buy_price * trade.buy_size
        if trade.resolution == "settlement_win":
            self.settle_win_count += 1
            self.settle_win_size += trade.buy_size
            self.settle_win_cost += cost
            self.settle_win_pnl += trade.pnl
        elif trade.resolution == "settlement_loss":
            self.settle_loss_count += 1
            self.settle_loss_size += trade.buy_size
            self.settle_loss_cost += cost
            self.settle_loss_pnl += trade.pnl
        elif trade.resolution == "closed_early":
            self.closed_count += 1
            self.closed_size += trade.buy_size
            self.closed_cost += cost
            self.closed_pnl += trade.pnl
            if trade.pnl > 0:
                self.closed_wins += 1


def _roi(pnl: float, cost: float) -> str:
    if cost <= 0:
        return "   n/a"
    return f"{pnl / cost * 100:>+6.1f}%"


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_by_day(trades: List[ResolvedTrade]):
    """Table 1: Overall P&L by days-to-settlement."""
    buckets: Dict[int, BucketStats] = defaultdict(BucketStats)
    for t in trades:
        buckets[min(t.days_to_settlement, 10)].add(t)

    print("=" * 115)
    print("TABLE 1  Overall P&L by Days-to-Settlement")
    print("=" * 115)
    hdr = f"{'Days':>5} {'#Trades':>8} {'Tot Cost':>10} {'Tot P&L':>10} {'Avg P&L':>10} {'ROI':>8} {'Win%':>6}"
    print(hdr)
    print("-" * 115)

    g_count, g_cost, g_pnl = 0, 0.0, 0.0
    for d in sorted(buckets, reverse=True):
        b = buckets[d]
        if b.total_count == 0:
            continue
        g_count += b.total_count
        g_cost += b.total_cost
        g_pnl += b.total_pnl
        wr = b.total_wins / b.total_count * 100
        label = f"{d}d" if d > 0 else "0d"
        print(f"{label:>5} {b.total_count:>8} ${b.total_cost:>9.2f} "
              f"${b.total_pnl:>+9.2f} ${b.total_pnl / b.total_count:>+9.2f} "
              f"{_roi(b.total_pnl, b.total_cost):>8} {wr:>5.0f}%")

    print("-" * 115)
    print(f"{'TOTAL':>5} {g_count:>8} ${g_cost:>9.2f} ${g_pnl:>+9.2f} "
          f"${g_pnl / max(1, g_count):>+9.2f} {_roi(g_pnl, g_cost):>8}")
    print()


def print_by_day_resolution(trades: List[ResolvedTrade]):
    """Table 2: Breakdown by resolution type (held-win, held-loss, closed)."""
    buckets: Dict[int, BucketStats] = defaultdict(BucketStats)
    for t in trades:
        buckets[min(t.days_to_settlement, 10)].add(t)

    print("=" * 135)
    print("TABLE 2  Breakdown by Resolution Type")
    print("=" * 135)
    print(f"{'Days':>5} {'--- Held→Win ---':>30} {'--- Held→Loss ---':>30} "
          f"{'--- Closed Early ---':>30} {'Net P&L':>12}")
    print(f"{'':>5} {'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}  "
          f"{'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}  "
          f"{'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}")
    print("-" * 135)

    for d in sorted(buckets, reverse=True):
        b = buckets[d]
        if b.total_count == 0:
            continue
        label = f"{d}d" if d > 0 else "0d"
        print(
            f"{label:>5} "
            f"{b.settle_win_count:>5} ${b.settle_win_cost:>8.2f} ${b.settle_win_pnl:>+8.2f} {_roi(b.settle_win_pnl, b.settle_win_cost):>7}  "
            f"{b.settle_loss_count:>5} ${b.settle_loss_cost:>8.2f} ${b.settle_loss_pnl:>+8.2f} {_roi(b.settle_loss_pnl, b.settle_loss_cost):>7}  "
            f"{b.closed_count:>5} ${b.closed_cost:>8.2f} ${b.closed_pnl:>+8.2f} {_roi(b.closed_pnl, b.closed_cost):>7} "
            f"${b.total_pnl:>+10.2f}"
        )
    print("-" * 135)
    print()


def print_by_price(trades: List[ResolvedTrade]):
    """Table 3: Overall P&L by buy-price bucket."""
    buckets: Dict[int, BucketStats] = defaultdict(BucketStats)
    for t in trades:
        buckets[t.price_bucket].add(t)

    print("=" * 115)
    print("TABLE 3  Overall P&L by Buy Price")
    print("=" * 115)
    print(f"{'Price':>10} {'#Trades':>8} {'Tot Cost':>10} {'Tot P&L':>10} "
          f"{'Avg P&L':>10} {'ROI':>8} {'Win%':>6}")
    print("-" * 115)

    g_count, g_cost, g_pnl = 0, 0.0, 0.0
    for i in range(len(PRICE_BUCKETS)):
        b = buckets.get(i)
        if not b or b.total_count == 0:
            continue
        g_count += b.total_count
        g_cost += b.total_cost
        g_pnl += b.total_pnl
        wr = b.total_wins / b.total_count * 100
        label = price_bucket_label(i)
        print(f"{label:>10} {b.total_count:>8} ${b.total_cost:>9.2f} "
              f"${b.total_pnl:>+9.2f} ${b.total_pnl / b.total_count:>+9.2f} "
              f"{_roi(b.total_pnl, b.total_cost):>8} {wr:>5.0f}%")

    print("-" * 115)
    print(f"{'TOTAL':>10} {g_count:>8} ${g_cost:>9.2f} ${g_pnl:>+9.2f} "
          f"${g_pnl / max(1, g_count):>+9.2f} {_roi(g_pnl, g_cost):>8}")
    print()


def print_by_price_resolution(trades: List[ResolvedTrade]):
    """Table 4: Price bucket breakdown by resolution type."""
    buckets: Dict[int, BucketStats] = defaultdict(BucketStats)
    for t in trades:
        buckets[t.price_bucket].add(t)

    print("=" * 135)
    print("TABLE 4  Buy Price Breakdown by Resolution Type")
    print("=" * 135)
    print(f"{'Price':>10} {'--- Held→Win ---':>30} {'--- Held→Loss ---':>30} "
          f"{'--- Closed Early ---':>30} {'Net P&L':>12}")
    print(f"{'':>10} {'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}  "
          f"{'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}  "
          f"{'#':>5} {'Cost':>9} {'P&L':>9} {'ROI':>7}")
    print("-" * 135)

    for i in range(len(PRICE_BUCKETS)):
        b = buckets.get(i)
        if not b or b.total_count == 0:
            continue
        label = price_bucket_label(i)
        print(
            f"{label:>10} "
            f"{b.settle_win_count:>5} ${b.settle_win_cost:>8.2f} ${b.settle_win_pnl:>+8.2f} {_roi(b.settle_win_pnl, b.settle_win_cost):>7}  "
            f"{b.settle_loss_count:>5} ${b.settle_loss_cost:>8.2f} ${b.settle_loss_pnl:>+8.2f} {_roi(b.settle_loss_pnl, b.settle_loss_cost):>7}  "
            f"{b.closed_count:>5} ${b.closed_cost:>8.2f} ${b.closed_pnl:>+8.2f} {_roi(b.closed_pnl, b.closed_cost):>7} "
            f"${b.total_pnl:>+10.2f}"
        )
    print("-" * 135)
    print()


def print_cross_table(trades: List[ResolvedTrade]):
    """Table 5: Days-to-settlement × price bucket cross-table (ROI heatmap)."""
    # Two-key bucket: (days, price_bucket) -> BucketStats
    cross: Dict[Tuple[int, int], BucketStats] = defaultdict(BucketStats)
    for t in trades:
        key = (min(t.days_to_settlement, 10), t.price_bucket)
        cross[key].add(t)

    # Determine which price buckets have data
    active_pb = sorted({pb for _, pb in cross})
    day_keys = sorted({d for d, _ in cross}, reverse=True)

    print("=" * (18 + 13 * len(active_pb)))
    print("TABLE 5  ROI (%) by Days-to-Settlement × Buy Price  [count in brackets]")
    print("=" * (18 + 13 * len(active_pb)))

    # Header
    hdr = f"{'Days':>5} "
    for pb in active_pb:
        hdr += f"{price_bucket_label(pb):>12} "
    hdr += f"{'ALL':>12}"
    print(hdr)
    print("-" * (18 + 13 * len(active_pb)))

    for d in day_keys:
        label = f"{d}d" if d > 0 else "0d"
        row = f"{label:>5} "
        row_total = BucketStats()
        for pb in active_pb:
            b = cross.get((d, pb))
            if b and b.total_count > 0:
                roi_val = b.total_pnl / b.total_cost * 100 if b.total_cost > 0 else 0
                row += f"{roi_val:>+6.0f}%[{b.total_count:>3}] "
                # Accumulate
                row_total.settle_win_pnl += b.settle_win_pnl
                row_total.settle_win_cost += b.settle_win_cost
                row_total.settle_win_count += b.settle_win_count
                row_total.settle_loss_pnl += b.settle_loss_pnl
                row_total.settle_loss_cost += b.settle_loss_cost
                row_total.settle_loss_count += b.settle_loss_count
                row_total.closed_pnl += b.closed_pnl
                row_total.closed_cost += b.closed_cost
                row_total.closed_count += b.closed_count
            else:
                row += f"{'—':>12} "

        # Row total
        if row_total.total_count > 0:
            rtot_roi = row_total.total_pnl / row_total.total_cost * 100 if row_total.total_cost > 0 else 0
            row += f"{rtot_roi:>+6.0f}%[{row_total.total_count:>3}]"
        print(row)

    # Column totals
    print("-" * (18 + 13 * len(active_pb)))
    tot_row = f"{'ALL':>5} "
    for pb in active_pb:
        col = BucketStats()
        for d in day_keys:
            b = cross.get((d, pb))
            if b:
                col.settle_win_pnl += b.settle_win_pnl
                col.settle_win_cost += b.settle_win_cost
                col.settle_win_count += b.settle_win_count
                col.settle_loss_pnl += b.settle_loss_pnl
                col.settle_loss_cost += b.settle_loss_cost
                col.settle_loss_count += b.settle_loss_count
                col.closed_pnl += b.closed_pnl
                col.closed_cost += b.closed_cost
                col.closed_count += b.closed_count
        if col.total_count > 0:
            croi = col.total_pnl / col.total_cost * 100 if col.total_cost > 0 else 0
            tot_row += f"{croi:>+6.0f}%[{col.total_count:>3}] "
        else:
            tot_row += f"{'—':>12} "
    print(tot_row)
    print()


def print_held_settlement_cross(trades: List[ResolvedTrade]):
    """Table 6: Same cross-table but only for held-to-settlement trades."""
    held = [t for t in trades if t.resolution != "closed_early"]
    if not held:
        return

    cross: Dict[Tuple[int, int], BucketStats] = defaultdict(BucketStats)
    for t in held:
        key = (min(t.days_to_settlement, 10), t.price_bucket)
        cross[key].add(t)

    active_pb = sorted({pb for _, pb in cross})
    day_keys = sorted({d for d, _ in cross}, reverse=True)

    print("=" * (18 + 13 * len(active_pb)))
    print("TABLE 6  Held-to-Settlement ROI (%) by Days × Price  [count]")
    print("=" * (18 + 13 * len(active_pb)))

    hdr = f"{'Days':>5} "
    for pb in active_pb:
        hdr += f"{price_bucket_label(pb):>12} "
    hdr += f"{'ALL':>12}"
    print(hdr)
    print("-" * (18 + 13 * len(active_pb)))

    for d in day_keys:
        label = f"{d}d" if d > 0 else "0d"
        row = f"{label:>5} "
        row_pnl, row_cost, row_cnt = 0.0, 0.0, 0
        for pb in active_pb:
            b = cross.get((d, pb))
            if b and b.total_count > 0:
                roi_val = b.total_pnl / b.total_cost * 100 if b.total_cost > 0 else 0
                row += f"{roi_val:>+6.0f}%[{b.total_count:>3}] "
                row_pnl += b.total_pnl
                row_cost += b.total_cost
                row_cnt += b.total_count
            else:
                row += f"{'—':>12} "
        if row_cnt > 0:
            row += f"{row_pnl / row_cost * 100 if row_cost > 0 else 0:>+6.0f}%[{row_cnt:>3}]"
        print(row)

    print("-" * (18 + 13 * len(active_pb)))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with same config options as run_backtest.py."""
    parser = argparse.ArgumentParser(
        description="Analyze backtest trades by days-to-settlement and buy price"
    )

    # Event selection
    parser.add_argument("--event", type=str, help="Specific event directory")
    parser.add_argument("--start-date", type=str, help="Events ending after (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Events ending before (YYYY-MM-DD)")
    parser.add_argument("--duration", type=int,
                        help="Only events with this counting window duration in days")

    # Trading parameters
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--spread", type=float, default=0.02)
    parser.add_argument("--slippage", type=float, default=0.005)
    parser.add_argument("--roi", type=float, default=EdgeBufferConfig.required_roi)
    parser.add_argument("--exit-hours", type=float, default=0.0)

    # Paths
    parser.add_argument("--price-data", type=str, default="data/price_history")
    parser.add_argument("--cache-dir", type=str, default="data/backtest_cache")

    # Logging
    parser.add_argument("-v", "--verbose", action="store_true")

    # Unified mode
    parser.add_argument("--unified", action="store_true",
                        help="Use unified Kelly trading logic (required)")

    # Kelly parameters
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--kelly-fraction", type=float, default=1.0)
    parser.add_argument("--min-utility", type=float, default=KellyConfig.min_utility)
    parser.add_argument("--t-stop", type=float, default=None)
    parser.add_argument("--c-bin-max-ratio", type=float, default=0.15)
    parser.add_argument("--min-perceived-prob", type=float,
                        default=EdgeBufferConfig.min_perceived_prob)
    parser.add_argument("--min-market-price", type=float,
                        default=EdgeBufferConfig.min_market_price)
    parser.add_argument("--max-spread-ratio", type=float, default=2.0)
    parser.add_argument("--no-require-two-sided", action="store_true")

    _adaptive_defaults = AdaptiveDeltaConfig()
    parser.add_argument("--base-delta-ratio", type=float,
                        default=_adaptive_defaults.base_delta_ratio)
    parser.add_argument("--max-orders", type=int, default=1000)

    parser.add_argument("--quick", action="store_true",
                        help="Deprecated, no-op.")

    parser.add_argument("--projection", type=str, default="asymmetric",
                        choices=["asymmetric", "normal", "skew_normal", "gamma"])
    parser.add_argument("--intraday-mode", type=str, default="ridge",
                        choices=["ridge", "bucket"])
    parser.add_argument("--event-rules", type=str, default=None)

    return parser


def build_runner(args) -> UnifiedBacktestRunner:
    """Create a UnifiedBacktestRunner from parsed CLI args."""
    base_delta_ratio = args.base_delta_ratio
    max_orders = args.max_orders

    event_trading_rules = None
    if args.event_rules:
        event_trading_rules = EventTradingRulesConfig.from_yaml(args.event_rules)

    _default_kelly = KellyConfig()
    trading_config = KellyConfig(
        kappa=args.kappa,
        kelly_fraction=args.kelly_fraction,
        min_utility=args.min_utility,
        t_stop_hours=args.t_stop if args.t_stop is not None else _default_kelly.t_stop_hours,
        edge_buffer=EdgeBufferConfig(
            required_roi=args.roi,
            friction_mid=0.015,
            friction_tail=0.03,
            tail_threshold=0.09,
            min_perceived_prob=args.min_perceived_prob,
            min_market_price=args.min_market_price,
            max_spread_ratio=args.max_spread_ratio,
            require_two_sided_liquidity=not args.no_require_two_sided,
        ),
        adaptive_delta=AdaptiveDeltaConfig(
            base_delta_ratio=base_delta_ratio,
            max_depth_fraction=0.10,
        ),
        rate_limit=RateLimitConfig(
            max_orders_per_tick=max_orders,
            min_order_delay_seconds=0.0,
            max_orders_per_minute=1000,
        ),
        collateral=CollateralConfig(
            c_bin_max_ratio=args.c_bin_max_ratio,
        ),
        max_iters_per_tick=1000,
    )

    unified_config = UnifiedBacktestConfig(
        initial_capital=args.capital,
        spread=args.spread,
        slippage=args.slippage,
        trading=trading_config,
        exit_hours_before_settlement=args.exit_hours,
        verbose=False,
        projection_model=args.projection,
        intraday_mode=args.intraday_mode,
        event_trading_rules=event_trading_rules,
    )

    return UnifiedBacktestRunner(
        config=unified_config,
        price_data_dir=Path(args.price_data),
        cache_dir=Path(args.cache_dir),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.unified:
        logger.error("Trade analysis requires --unified mode")
        sys.exit(1)

    runner = build_runner(args)
    price_data_dir = Path(args.price_data)

    # Determine events
    if args.event:
        event_dirs = [args.event]
    else:
        all_events = runner.price_provider.list_available_events()
        start_date = date.fromisoformat(args.start_date) if args.start_date else None
        end_date = date.fromisoformat(args.end_date) if args.end_date else None
        event_dirs = filter_events_by_date(all_events, start_date, end_date)
        if args.duration:
            event_dirs = filter_events_by_duration(event_dirs, price_data_dir, args.duration)

    if not event_dirs:
        logger.error("No events to analyze")
        sys.exit(1)

    logger.info(f"Will analyze {len(event_dirs)} event(s)")
    logger.info(f"Projection: {args.projection}, Intraday: {args.intraday_mode}")

    # Pre-load settlement dates
    settlement_dates: Dict[str, date] = {}
    for event_dir in event_dirs:
        event = runner.price_provider.load_event(event_dir)
        if event and event.counting_end_date:
            settlement_dates[event.short_name] = event.counting_end_date

    # Run backtests
    results = run_multiple_backtests(runner, event_dirs)
    logger.info(f"Completed {len(results)} events")

    # Analyze trades
    all_resolved: List[ResolvedTrade] = []
    for result in results:
        settlement = settlement_dates.get(result.event_name)
        if not settlement:
            logger.warning(f"No settlement date for {result.event_name}, skipping")
            continue
        resolved = analyze_event_trades(result, settlement)
        all_resolved.extend(resolved)

    if not all_resolved:
        print("No trades to analyze.")
        sys.exit(0)

    # Verify P&L reconciliation
    total_resolved_pnl = sum(t.pnl for t in all_resolved)
    total_backtest_pnl = sum(r.total_pnl for r in results)
    print()
    print(f"Total resolved trades: {len(all_resolved)}")
    print(f"Resolved P&L: ${total_resolved_pnl:+.2f}  |  Backtest P&L: ${total_backtest_pnl:+.2f}  "
          f"|  Diff: ${abs(total_resolved_pnl - total_backtest_pnl):.2f}")
    print()

    # Print all tables
    print_by_day(all_resolved)
    print_by_day_resolution(all_resolved)
    print_by_price(all_resolved)
    print_by_price_resolution(all_resolved)
    print_cross_table(all_resolved)
    print_held_settlement_cross(all_resolved)


if __name__ == "__main__":
    main()
