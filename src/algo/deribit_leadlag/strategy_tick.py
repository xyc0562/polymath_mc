"""
Single strategy tick — shared between live (`main.LeadLagBot`) and backtest.

Pure logic: takes a snapshot of inputs (options, markets, orderbooks) plus a
PositionManager and an order gateway, runs one decision pass, and returns the
execution report. No I/O, no globals, no time of its own — `now` is a parameter.

The order gateway is duck-typed: it only needs `execute_actions(actions)` and
returns an object with `canceled_order_ids`, `posted_actions`, `raw_results`
attributes (matching `OrderManager.ExecutionReport`).
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from .config import SignalConfig
from .deribit_client import DeribitOption
from .polymarket_discovery import ThresholdMarket
from .position_manager import BinKey, PositionManager
from .settlement import CompatibilityClass
from .signal_comparator import build_adjusted_prob_map

logger = logging.getLogger(__name__)


def run_strategy_tick(
    *,
    now: datetime,
    options: List[DeribitOption],
    markets: List[ThresholdMarket],
    compat_map: Dict[BinKey, CompatibilityClass],
    orderbooks_by_token: Dict[str, object],
    signal_config: SignalConfig,
    position_mgr: PositionManager,
    order_gateway,
) -> Optional[object]:
    """Run one decision pass. Returns the execution report or None if no-op.

    Same path the live bot exercises in `LeadLagBot._strategy_tick`.
    """
    if not options:
        return None

    adjusted_probs = build_adjusted_prob_map(
        options=options,
        now=now,
        markets=markets,
        signal_config=signal_config,
        compatibility_map=compat_map,
    )
    if not adjusted_probs:
        return None

    position_mgr.compute_targets(adjusted_probs, orderbooks_by_token)
    actions = position_mgr.compute_deltas()
    if not actions:
        return None

    report = order_gateway.execute_actions(actions)
    position_mgr.apply_execution_report(report)
    if getattr(report, "posted_actions", None):
        logger.info(f"Executed {len(report.posted_actions)} orders this tick")
    return report
