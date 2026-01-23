"""
WebSocket client for real-time Polymarket orderbook streaming.

Connects to Polymarket's WebSocket API to receive orderbook updates
in real-time, avoiding the need for frequent polling.

Endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/market
"""

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from .config import WebSocketConfig
from .orderbook import UnifiedOrderbook, OrderbookLevel

logger = logging.getLogger(__name__)


class OrderbookWebSocket:
    """
    WebSocket client for real-time orderbook streaming from Polymarket.

    Features:
    - Automatic reconnection on disconnect
    - Dynamic subscription management
    - Heartbeat to keep connection alive
    - Thread-safe orderbook access
    """

    def __init__(
        self,
        config: WebSocketConfig,
        on_orderbook_update: Optional[Callable[[str, UnifiedOrderbook], None]] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            config: WebSocket configuration
            on_orderbook_update: Optional callback for orderbook updates.
                                 Called with (token_id, orderbook).
        """
        self.config = config
        self.on_orderbook_update = on_orderbook_update

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = False

        # Subscriptions
        self._subscribed_tokens: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()

        # Orderbook cache
        self._orderbooks: Dict[str, UnifiedOrderbook] = {}
        self._token_to_bin: Dict[str, int] = {}  # Map token_id -> bin_index

        # Tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._running:
            return

        self._running = True
        logger.info(f"Connecting to WebSocket: {self.config.ws_url}")

        try:
            self._ws = await websockets.connect(
                self.config.ws_url,
                ping_interval=self.config.heartbeat_interval_seconds,
                ping_timeout=10,
            )
            self._connected = True
            logger.info("WebSocket connected successfully")

            # Resubscribe to any pending tokens
            if self._pending_subscriptions:
                await self._subscribe(list(self._pending_subscriptions))
                self._pending_subscriptions.clear()

            # Start listener and heartbeat tasks
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        self._running = False
        self._connected = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("WebSocket disconnected")

    async def subscribe(
        self,
        token_ids: List[str],
        bin_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Subscribe to orderbook updates for tokens.

        Args:
            token_ids: List of YES token IDs to subscribe to
            bin_indices: Optional corresponding bin indices for mapping
        """
        # Store bin mappings
        if bin_indices:
            for token_id, bin_idx in zip(token_ids, bin_indices):
                self._token_to_bin[token_id] = bin_idx

        new_tokens = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_tokens:
            return

        if self.is_connected:
            await self._subscribe(new_tokens)
        else:
            # Queue for subscription after connection
            self._pending_subscriptions.update(new_tokens)

    async def _subscribe(self, token_ids: List[str]) -> None:
        """Send subscription message to WebSocket."""
        if not self._ws:
            return

        msg = {
            "type": "MARKET",
            "assets_ids": token_ids,
            "auth": {},
        }

        try:
            await self._ws.send(json.dumps(msg))
            self._subscribed_tokens.update(token_ids)
            logger.info(f"Subscribed to {len(token_ids)} tokens")
        except Exception as e:
            logger.error(f"Subscription failed: {e}")

    async def unsubscribe(self, token_ids: List[str]) -> None:
        """Unsubscribe from token updates."""
        if not self._ws or not token_ids:
            return

        msg = {
            "assets_ids": token_ids,
            "operation": "unsubscribe",
        }

        try:
            await self._ws.send(json.dumps(msg))
            self._subscribed_tokens -= set(token_ids)
            logger.info(f"Unsubscribed from {len(token_ids)} tokens")
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")

    async def _listen_loop(self) -> None:
        """Main loop to receive and process messages."""
        while self._running:
            try:
                if not self._ws:
                    await asyncio.sleep(1)
                    continue

                message = await self._ws.recv()
                data = json.loads(message)
                self._handle_message(data)

            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self._connected = False
                if self._running:
                    await self._reconnect()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                if self._ws and self._connected:
                    # websockets library handles ping/pong automatically,
                    # but we can send a custom heartbeat if needed
                    pass

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect after disconnect."""
        logger.info("Attempting to reconnect...")

        # Store current subscriptions for resubscription
        tokens_to_resubscribe = list(self._subscribed_tokens)
        self._subscribed_tokens.clear()
        self._pending_subscriptions.update(tokens_to_resubscribe)

        while self._running:
            try:
                await asyncio.sleep(self.config.reconnect_delay_seconds)

                self._ws = await websockets.connect(
                    self.config.ws_url,
                    ping_interval=self.config.heartbeat_interval_seconds,
                    ping_timeout=10,
                )
                self._connected = True
                logger.info("WebSocket reconnected")

                # Resubscribe
                if self._pending_subscriptions:
                    await self._subscribe(list(self._pending_subscriptions))
                    self._pending_subscriptions.clear()

                break

            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

    def _handle_message(self, data: dict) -> None:
        """Process incoming WebSocket message."""
        event_type = data.get("event_type")

        if event_type == "book":
            self._handle_book_update(data)
        elif event_type == "price_change":
            self._handle_price_change(data)
        elif event_type == "last_trade_price":
            # Trade occurred - can be used for tracking
            pass
        else:
            # Unknown event type
            logger.debug(f"Unknown event type: {event_type}")

    def _handle_book_update(self, data: dict) -> None:
        """Handle full orderbook snapshot."""
        token_id = data.get("asset_id")
        if not token_id:
            return

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        bin_index = self._token_to_bin.get(token_id, 0)

        orderbook = UnifiedOrderbook.from_api_response(
            bin_index=bin_index,
            yes_token_id=token_id,
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

        self._orderbooks[token_id] = orderbook

        # Call update callback if provided
        if self.on_orderbook_update:
            self.on_orderbook_update(token_id, orderbook)

    def _handle_price_change(self, data: dict) -> None:
        """Handle incremental price update."""
        token_id = data.get("asset_id")
        if not token_id or token_id not in self._orderbooks:
            return

        # For price changes, we may need to update specific levels
        # For simplicity, we treat it as requiring a full book refresh
        # In production, implement incremental updates for efficiency

        # Get changes
        changes = data.get("changes", [])
        for change in changes:
            side = change.get("side")  # "BUY" or "SELL"
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))

            orderbook = self._orderbooks[token_id]

            if side == "BUY":
                # Update bid
                self._update_level(orderbook.yes_bids, price, size, reverse=True)
            elif side == "SELL":
                # Update ask
                self._update_level(orderbook.yes_asks, price, size, reverse=False)

        orderbook.last_updated = time.time()

        if self.on_orderbook_update:
            self.on_orderbook_update(token_id, orderbook)

    def _update_level(
        self,
        levels: List[OrderbookLevel],
        price: float,
        size: float,
        reverse: bool,
    ) -> None:
        """Update a specific price level in the orderbook."""
        # Find existing level
        for i, level in enumerate(levels):
            if abs(level.price - price) < 0.0001:
                if size <= 0:
                    # Remove level
                    levels.pop(i)
                else:
                    # Update size
                    level.size = size
                return

        # Add new level if size > 0
        if size > 0:
            levels.append(OrderbookLevel(price=price, size=size))
            # Re-sort
            levels.sort(key=lambda x: x.price, reverse=reverse)

    def get_orderbook(self, token_id: str) -> Optional[UnifiedOrderbook]:
        """Get current orderbook for a token."""
        return self._orderbooks.get(token_id)

    def get_all_orderbooks(self) -> Dict[str, UnifiedOrderbook]:
        """Get all cached orderbooks."""
        return self._orderbooks.copy()


class OrderbookManager:
    """
    High-level manager for orderbook data.

    Provides both WebSocket streaming and REST API fallback.
    """

    def __init__(
        self,
        config: WebSocketConfig,
        clob_client=None,  # ClobClient for fallback
    ):
        """
        Initialize orderbook manager.

        Args:
            config: WebSocket configuration
            clob_client: Optional ClobClient for REST API fallback
        """
        self.config = config
        self.clob_client = clob_client

        self._ws_client: Optional[OrderbookWebSocket] = None
        self._orderbooks: Dict[str, UnifiedOrderbook] = {}
        self._token_to_bin: Dict[str, int] = {}

    async def start(self) -> None:
        """Start the orderbook manager."""
        if self.config.enabled:
            self._ws_client = OrderbookWebSocket(
                config=self.config,
                on_orderbook_update=self._on_ws_update,
            )
            await self._ws_client.connect()
        else:
            logger.info("WebSocket disabled, using REST API polling")

    async def stop(self) -> None:
        """Stop the orderbook manager."""
        if self._ws_client:
            await self._ws_client.disconnect()

    def _on_ws_update(self, token_id: str, orderbook: UnifiedOrderbook) -> None:
        """Handle orderbook update from WebSocket."""
        self._orderbooks[token_id] = orderbook

    async def subscribe_bins(
        self,
        bins: List[dict],  # List of {"token_id": str, "bin_index": int}
    ) -> None:
        """Subscribe to orderbooks for multiple bins."""
        token_ids = [b["token_id"] for b in bins]
        bin_indices = [b["bin_index"] for b in bins]

        for token_id, bin_idx in zip(token_ids, bin_indices):
            self._token_to_bin[token_id] = bin_idx

        if self._ws_client:
            await self._ws_client.subscribe(token_ids, bin_indices)

    def get_orderbook(self, token_id: str) -> Optional[UnifiedOrderbook]:
        """
        Get orderbook for a token.

        Returns cached WebSocket data or fetches via REST if needed.
        """
        # Try WebSocket cache first
        if self._ws_client:
            ob = self._ws_client.get_orderbook(token_id)
            if ob:
                return ob

        # Return local cache
        return self._orderbooks.get(token_id)

    async def fetch_orderbook(
        self,
        token_id: str,
        bin_index: int = 0,
    ) -> Optional[UnifiedOrderbook]:
        """
        Fetch orderbook via REST API.

        Used as fallback when WebSocket is not available or stale.
        """
        if not self.clob_client:
            return None

        try:
            response = self.clob_client.get_order_book(token_id)

            orderbook = UnifiedOrderbook.from_api_response(
                bin_index=bin_index,
                yes_token_id=token_id,
                bids=response.get("bids", []),
                asks=response.get("asks", []),
                timestamp=time.time(),
            )

            self._orderbooks[token_id] = orderbook
            return orderbook

        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return None
