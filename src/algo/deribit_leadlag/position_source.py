"""
Live position/order source: pulls inventory from the Polymarket data API and
open orders from the CLOB client, then pushes the resulting snapshots into a
PositionManager.

Kept separate from PositionManager so the manager itself stays pure logic and
the backtest runner can feed it from an in-memory ledger using the same
setter API.
"""

import logging
import time
from typing import Dict, List

import requests

from .position_manager import BinKey, OpenOrder, PositionManager

logger = logging.getLogger(__name__)

POLYMARKET_DATA_API = "https://data-api.polymarket.com"


class LivePositionSource:
    """Live wiring around PositionManager. All I/O happens here."""

    def __init__(self, wallet_address: str):
        self._wallet = wallet_address.lower() if wallet_address else ""

    def fetch_positions(self, position_mgr: PositionManager) -> None:
        """Fetch holdings via Polymarket data API and push into the manager.

        With no wallet address we cannot query the API. Leave the manager's
        existing inventory untouched — wiping it would erase positions learned
        from WebSocket fills and cause `compute_deltas` to trade as if flat.
        """
        if not self._wallet:
            logger.warning("Cannot fetch positions without wallet address")
            return

        bins = position_mgr.get_bins()
        yes_tokens = {bs.poly_market.yes_token_id for bs in bins.values()}
        no_tokens = {bs.poly_market.no_token_id for bs in bins.values()}

        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/positions",
                params={"user": self._wallet, "sizeThreshold": 0},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(f"Failed to fetch positions snapshot: {exc}")
            return

        if isinstance(payload, dict):
            positions = payload.get("positions", payload.get("data", []))
        else:
            positions = payload

        yes_by_token: Dict[str, float] = {}
        no_by_token: Dict[str, float] = {}
        for pos in positions:
            asset = pos.get("asset")
            if isinstance(asset, str):
                token_id = asset
            elif isinstance(asset, dict):
                token_id = asset.get("id")
            else:
                token_id = pos.get("token_id") or pos.get("asset_id")
            shares = float(pos.get("shares") or pos.get("size") or 0)
            if shares <= 0 or not token_id:
                continue
            if token_id in yes_tokens:
                yes_by_token[token_id] = shares
            elif token_id in no_tokens:
                no_by_token[token_id] = shares

        position_mgr.set_position_snapshot(yes_by_token, no_by_token)
        logger.info(
            "Fetched positions: %d bins, total YES=%.0f NO=%.0f",
            len(bins),
            sum(yes_by_token.values()),
            sum(no_by_token.values()),
        )

    def fetch_open_orders(self, position_mgr: PositionManager, clob_client) -> None:
        """Fetch live orders via the CLOB client and push into the manager.

        Clears the manager's open-order view before calling the CLOB so that a
        transient `get_orders()` failure leaves us with no stale orders rather
        than the pre-cancel snapshot. Match the legacy ordering: the original
        `PositionManager.fetch_open_orders` cleared first, then queried.
        """
        bins = position_mgr.get_bins()
        token_to_bin: Dict[str, BinKey] = {}
        for bs in bins.values():
            token_to_bin[bs.poly_market.yes_token_id] = bs.bin_key
            token_to_bin[bs.poly_market.no_token_id] = bs.bin_key

        position_mgr.set_open_orders({})

        try:
            orders = clob_client.get_orders()
            live_orders = [o for o in orders if o.get("status") == "LIVE"]
        except Exception as exc:
            logger.warning(f"Failed to fetch open orders: {exc}")
            return

        orders_by_bin: Dict[BinKey, List[OpenOrder]] = {}
        for order in live_orders:
            token_id = order.get("asset_id", "")
            bin_key = token_to_bin.get(token_id)
            if bin_key is None:
                continue
            order_type = order.get("type") or order.get("order_type") or ""
            orders_by_bin.setdefault(bin_key, []).append(
                OpenOrder(
                    order_id=order.get("id", ""),
                    bin_key=bin_key,
                    side=order.get("side", "BUY"),
                    token_id=token_id,
                    price=float(order.get("price", 0) or 0),
                    size=float(order.get("original_size", 0) or order.get("size", 0) or 0),
                    posted_at=time.time(),
                    edge_at_post=0.0,
                    is_maker=order_type in ("GTC", "GTD"),
                )
            )

        position_mgr.set_open_orders(orders_by_bin)
        total = sum(len(v) for v in orders_by_bin.values())
        logger.info("Mapped %d live orders across bins", total)
