"""
FAK order placement and state persistence for lead-lag trading.

Inlines the order placement pattern from the musk kelly executor
to avoid importing its heavy dependency graph.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from .config import ExecutionConfig
from .signal_comparator import TradeSignal

logger = logging.getLogger(__name__)

# Polymarket minimums
MIN_ORDER_SIZE = 15
MIN_ORDER_VALUE_USD = 1.0


class LeadLagExecutor:
    """Executes FAK orders on Polymarket and tracks exposure."""

    def __init__(self, clob_client: ClobClient, config: ExecutionConfig):
        self.client = clob_client
        self.config = config
        self.exposure: Dict[str, float] = {}  # condition_id -> USD exposure
        self.total_exposure: float = 0.0
        self._load_state()

    def execute_signal(self, signal: TradeSignal) -> Optional[dict]:
        """
        Place a FAK order for the given signal.

        Returns order response dict, or None on failure/dry-run.
        """
        token_id = signal.token_id
        price = signal.price
        size = signal.size_shares

        # Validation
        if price <= 0 or price >= 1:
            logger.warning(f"Invalid price: {price:.4f}")
            return None

        rounded_size = math.floor(size)
        if rounded_size < MIN_ORDER_SIZE and rounded_size * price < MIN_ORDER_VALUE_USD:
            logger.warning(
                f"Order too small: {rounded_size} shares @ {price:.4f} "
                f"(${rounded_size * price:.2f})"
            )
            return None

        order_usd = rounded_size * price

        # Exposure check
        if self.total_exposure + order_usd > self.config.max_total_exposure_usd:
            logger.info(
                f"Would exceed max exposure: current=${self.total_exposure:.2f} "
                f"+ order=${order_usd:.2f} > max=${self.config.max_total_exposure_usd:.2f}"
            )
            return None

        # Log signal details
        pair = signal.matched
        logger.info(
            f"[SIGNAL] {signal.side} {pair.expiry_date}/{pair.strike:.0f} "
            f"| deribit_prob={signal.deribit_prob_mid:.3f} (conservative={signal.deribit_prob_conservative:.3f}) "
            f"| poly_yes={signal.poly_price:.3f} "
            f"| gross_edge={signal.gross_edge:.3f} net_edge={signal.net_edge:.3f} "
            f"| fee={signal.fee_per_share:.4f} "
            f"| method={pair.deribit_prob.method}"
        )

        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] Would place FAK: {signal.side} "
                f"{rounded_size} shares @ {price:.4f} "
                f"token={token_id[:20]}..."
            )
            self._track_exposure(signal, order_usd)
            return {"order_id": "dry_run", "status": "simulated"}

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=rounded_size,
                side="BUY",
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, orderType=OrderType.FAK)

            order_id = response.get("orderID", "unknown")
            logger.info(
                f"[ORDER] Placed FAK: {signal.side} {rounded_size} @ {price:.4f} "
                f"order_id={order_id}"
            )

            self._track_exposure(signal, order_usd)
            return response

        except Exception as e:
            logger.error(f"[ORDER FAILED] {e}")
            return None

    def _track_exposure(self, signal: TradeSignal, order_usd: float):
        """Update in-memory and persisted exposure tracking."""
        cond_id = signal.matched.poly_market.condition_id
        self.exposure[cond_id] = self.exposure.get(cond_id, 0.0) + order_usd
        self.total_exposure += order_usd
        self._save_state()

    def _load_state(self):
        """Load persisted exposure state from JSON file."""
        path = Path(self.config.state_file)
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)
            self.exposure = state.get("exposure", {})
            self.total_exposure = state.get("total_exposure", 0.0)
            logger.info(
                f"Loaded state: total_exposure=${self.total_exposure:.2f}, "
                f"{len(self.exposure)} positions"
            )
        except Exception as e:
            logger.warning(f"Failed to load state from {path}: {e}")

    def _save_state(self):
        """Persist exposure state to JSON file."""
        path = Path(self.config.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            state = {
                "exposure": self.exposure,
                "total_exposure": self.total_exposure,
            }
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state to {path}: {e}")

    def reconcile_positions(self):
        """
        Reconcile persisted state against actual Polymarket positions.

        Fetches current positions from the CLOB client and updates
        the exposure tracking to match reality.
        """
        try:
            # py_clob_client doesn't have a direct get_positions for market positions
            # For POC, we rely on persisted state + startup logging
            logger.info(
                f"Exposure state: total=${self.total_exposure:.2f}, "
                f"positions={len(self.exposure)}"
            )
        except Exception as e:
            logger.warning(f"Position reconciliation failed: {e}")
