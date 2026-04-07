"""
Polymarket WebSocket wrapper for lead-lag trading.

Wraps musk OrderbookWebSocket and UserStreamClient for orderbook streaming
and fill confirmations. Does NOT modify musk code.
"""

import logging
from typing import Callable, Dict, List, Optional

from src.algo.musk_tweet_count.kelly.websocket_client import (
    OrderbookWebSocket,
    WebSocketConfig,
)
from src.algo.musk_tweet_count.kelly.user_stream import (
    UserStreamClient,
    FillEvent,
    OrderEvent,
)
from src.algo.musk_tweet_count.kelly.orderbook import UnifiedOrderbook

from .config import PolyWSConfig
from .polymarket_discovery import ThresholdMarket

logger = logging.getLogger(__name__)


class PolymarketStreams:
    """
    Manages Polymarket WebSocket connections for the lead-lag bot.

    - Orderbook WS: real-time orderbook updates for all tracked markets
    - User stream: authenticated channel for fill and order confirmations
    """

    def __init__(
        self,
        config: PolyWSConfig,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        on_fill: Optional[Callable[[FillEvent], None]] = None,
        on_order: Optional[Callable[[OrderEvent], None]] = None,
    ):
        ws_config = WebSocketConfig(
            enabled=True,
            reconnect_delay_seconds=5.0,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            ws_url=config.orderbook_ws_url,
        )
        self._orderbook_ws = OrderbookWebSocket(config=ws_config)

        self._user_stream = UserStreamClient(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        self._user_stream.on_fill = on_fill
        self._user_stream.on_order = on_order

        self._tracked_token_ids: List[str] = []

    async def start(self) -> None:
        """Start both WebSocket connections."""
        logger.info("Starting Polymarket streams...")
        await self._user_stream.start()
        await self._orderbook_ws.connect()
        logger.info("Polymarket streams started")

    async def stop(self) -> None:
        """Stop both WebSocket connections."""
        logger.info("Stopping Polymarket streams...")
        await self._orderbook_ws.disconnect()
        await self._user_stream.stop()
        logger.info("Polymarket streams stopped")

    async def subscribe_markets(self, markets: List[ThresholdMarket]) -> None:
        """Subscribe to orderbook updates for all market tokens."""
        token_ids = []
        condition_ids = set()
        for m in markets:
            token_ids.append(m.yes_token_id)
            # NO token shares the same orderbook as YES in Polymarket
            condition_ids.add(m.condition_id)

        self._tracked_token_ids = token_ids

        # Subscribe orderbook WS to YES token IDs
        await self._orderbook_ws.subscribe(token_ids)

        # Subscribe user stream to market (condition) IDs
        await self._user_stream.set_markets(condition_ids)

        logger.info(
            f"Subscribed to {len(token_ids)} orderbooks, "
            f"{len(condition_ids)} user stream markets"
        )

    def get_orderbook(self, token_id: str) -> Optional[UnifiedOrderbook]:
        """Get current orderbook for a token."""
        return self._orderbook_ws.get_orderbook(token_id)

    def get_all_orderbooks(self) -> Dict[str, UnifiedOrderbook]:
        """Get all cached orderbooks."""
        result = {}
        for tid in self._tracked_token_ids:
            ob = self._orderbook_ws.get_orderbook(tid)
            if ob is not None:
                result[tid] = ob
        return result
