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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from .orderbook import UnifiedOrderbook, OrderbookLevel

logger = logging.getLogger(__name__)
APP_PING_MESSAGE = "PING"
APP_PONG_MESSAGE = "PONG"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket orderbook streaming."""

    # Enable WebSocket streaming (vs polling)
    enabled: bool = True

    # Delay between reconnection attempts
    reconnect_delay_seconds: float = 5.0

    # Polymarket expects app-level PING messages roughly every 10 seconds.
    heartbeat_interval_seconds: float = 10.0

    # WebSocket endpoint
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

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
                                 For backward compat; auto-registered as a callback.
        """
        self.config = config

        # Multi-callback support: each OrderbookManager registers its own callback
        self._update_callbacks: List[Callable[[str, UnifiedOrderbook], None]] = []
        if on_orderbook_update:
            self._update_callbacks.append(on_orderbook_update)

        # Connection state
        self._ws: Optional[ClientConnection] = None
        self._running = False
        self._connected = False

        # Reconnect state with exponential backoff
        self._reconnect_delay = config.reconnect_delay_seconds
        self._max_reconnect_delay = 60.0

        # Liveness tracking for dead connection detection
        self._last_pong_time: float = time.time()
        self._last_message_time: float = time.time()

        # Subscriptions
        self._subscribed_tokens: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        self._subscribe_sent: bool = False  # True after first subscribe on this connection

        # Orderbook cache
        self._orderbooks: Dict[str, UnifiedOrderbook] = {}
        self._token_to_bin: Dict[str, int] = {}  # Map token_id -> bin_index

        # Tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

    def register_callback(self, cb: Callable[[str, UnifiedOrderbook], None]) -> None:
        """Register an orderbook update callback."""
        if cb not in self._update_callbacks:
            self._update_callbacks.append(cb)

    def unregister_callback(self, cb: Callable[[str, UnifiedOrderbook], None]) -> None:
        """Unregister an orderbook update callback."""
        try:
            self._update_callbacks.remove(cb)
        except ValueError:
            pass

    def get_connection_status(self) -> dict:
        """Get connection health status."""
        return {
            "enabled": self.config.enabled,
            "connected": self.is_connected,
            "subscribed_tokens": len(self._subscribed_tokens),
            "cached_orderbooks": len(self._orderbooks),
            "last_message_age_seconds": round(time.time() - self._last_message_time, 1),
        }

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._is_ws_open()

    def _is_ws_open(self) -> bool:
        """Return True when the websocket is fully open."""
        if self._ws is None:
            return False

        state = getattr(self._ws, "state", None)
        if state is not None:
            return state == State.OPEN

        closed = getattr(self._ws, "closed", None)
        if closed is not None:
            return not closed

        return False

    @staticmethod
    def _format_close_details(exc: ConnectionClosed) -> str:
        details = f"code={getattr(exc, 'code', 'unknown')}"
        reason = getattr(exc, "reason", "") or ""
        if reason:
            details += f", reason={reason}"
        return details

    async def _send_message(self, payload: str) -> None:
        """Serialize websocket writes to avoid concurrent send races."""
        if not self._is_ws_open():
            return

        async with self._send_lock:
            if self._ws is not None:
                await self._ws.send(payload)

    async def _send_json(self, payload: dict) -> None:
        """Send a JSON payload over the websocket."""
        await self._send_message(json.dumps(payload))

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if not self.config.enabled:
            logger.info("WebSocket disabled, skipping connect")
            return

        if self._running:
            return

        self._running = True
        logger.info(f"Connecting to WebSocket: {self.config.ws_url}")

        try:
            self._ws = await websockets.connect(
                self.config.ws_url,
                open_timeout=20,
                close_timeout=10,
                ping_interval=None,
                ping_timeout=None,
            )
            self._connected = True
            self._subscribe_sent = False
            self._reconnect_delay = self.config.reconnect_delay_seconds
            self._last_pong_time = time.time()
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
        """Send subscription message to WebSocket.

        The Polymarket orderbook WS only accepts a single subscribe message
        per connection.  If a subscribe has already been sent, the new tokens
        are queued and a reconnect is triggered so they are included in the
        fresh subscribe on the new connection.
        """
        if not self._ws:
            return

        self._subscribed_tokens.update(token_ids)

        if self._subscribe_sent:
            # Can't send another subscribe — queue and reconnect
            logger.info(
                f"Subscribe already sent on this connection, "
                f"queuing {len(token_ids)} new tokens and reconnecting"
            )
            await self.force_resubscribe()
            return

        all_tokens = list(self._subscribed_tokens)
        msg = {
            "type": "MARKET",
            "assets_ids": all_tokens,
            "auth": {},
        }

        try:
            await self._send_json(msg)
            self._subscribe_sent = True
            logger.info(f"Subscribed to {len(all_tokens)} tokens (added {len(token_ids)} new)")
        except Exception as e:
            logger.error(f"Subscription failed: {e}")

    async def unsubscribe(self, token_ids: List[str]) -> None:
        """Unsubscribe from token updates (local state only).

        The Polymarket WS does not support per-token unsubscribe messages,
        so we only clean up local state here.  Call ``force_resubscribe()``
        afterwards to reconnect with the pruned token set (e.g. after an
        event expires).
        """
        if not token_ids:
            return

        token_set = set(token_ids)

        # Drop local subscription state so dead tokens are not resubscribed
        # on the next reconnect.
        self._pending_subscriptions -= token_set
        self._subscribed_tokens -= token_set
        for token_id in token_ids:
            self._token_to_bin.pop(token_id, None)
            self._orderbooks.pop(token_id, None)

        logger.info(f"Unsubscribed locally from {len(token_ids)} tokens, "
                     f"{len(self._subscribed_tokens)} remaining")

    async def force_resubscribe(self) -> None:
        """Force a reconnect so the server only sends updates for current tokens.

        Should be called after removing tokens via ``unsubscribe()`` when you
        want the server-side subscription to reflect the change (e.g. after
        event expiry).  Skipped if there are no remaining subscriptions.
        """
        if not self._subscribed_tokens:
            logger.info("No tokens left after unsubscribe, skipping resubscribe")
            return
        if not self._ws:
            return

        logger.info(f"Force-resubscribing with {len(self._subscribed_tokens)} tokens")
        try:
            await self._ws.close()
        except Exception:
            pass
        # _reconnect is triggered by the listen loop when the WS closes

    async def _listen_loop(self) -> None:
        """Main loop to receive and process messages."""
        while self._running:
            try:
                if not self._ws:
                    await asyncio.sleep(1)
                    continue

                message = await self._ws.recv()
                self._last_message_time = time.time()
                if isinstance(message, str):
                    raw = message.strip()
                    if raw == APP_PONG_MESSAGE:
                        self._last_pong_time = time.time()
                        continue
                    if raw == APP_PING_MESSAGE:
                        continue
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message from server: %s", raw[:120])
                        continue
                    self._process_data(data)

            except ConnectionClosed as e:
                logger.warning(
                    "WebSocket connection closed: %s",
                    self._format_close_details(e),
                )
                self._connected = False
                if self._running:
                    await self._reconnect()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_loop(self) -> None:
        """Monitor connection liveness by checking for prolonged silence.

        The Polymarket orderbook WS does not support application-level PING;
        sending one causes an immediate server-side disconnect.  Instead we
        track the time of the last received message and force-close if the
        connection has been silent for too long.
        """
        silence_limit = 60.0  # seconds of silence before we consider the connection dead
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                if self.is_connected:
                    silence = time.time() - self._last_message_time
                    if silence > silence_limit:
                        logger.warning(
                            f"No messages received for {silence:.0f}s, forcing reconnect"
                        )
                        if self._ws:
                            await self._ws.close()

            except asyncio.CancelledError:
                break

            except ConnectionClosed as e:
                logger.warning(f"Heartbeat error: {self._format_close_details(e)}")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect after disconnect with exponential backoff."""
        logger.info("Attempting to reconnect...")

        # Store current subscriptions for resubscription
        tokens_to_resubscribe = list(self._subscribed_tokens)
        self._subscribed_tokens.clear()
        self._pending_subscriptions.update(tokens_to_resubscribe)

        # Clear stale orderbook cache — incremental price_change events may have
        # been lost during the disconnect, so the cached data could be wrong.
        # The next full 'book' snapshot on resubscribe will repopulate correctly.
        self._orderbooks.clear()

        while self._running:
            try:
                logger.info(f"Reconnecting in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)

                self._ws = await websockets.connect(
                    self.config.ws_url,
                    open_timeout=20,
                    close_timeout=10,
                    ping_interval=None,
                    ping_timeout=None,
                )
                self._connected = True
                self._subscribe_sent = False
                self._reconnect_delay = self.config.reconnect_delay_seconds
                self._last_pong_time = time.time()
                logger.info("WebSocket reconnected")

                # Resubscribe with all pending tokens
                if self._pending_subscriptions:
                    await self._subscribe(list(self._pending_subscriptions))
                    self._pending_subscriptions.clear()

                break

            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                # Exponential backoff
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    def _process_data(self, data) -> None:
        """Recursively process data, handling nested lists."""
        if isinstance(data, list):
            for item in data:
                self._process_data(item)
        elif isinstance(data, dict):
            self._handle_message(data)
        # Skip primitives (strings, numbers, None)

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

        # Notify all registered callbacks
        for cb in self._update_callbacks:
            try:
                cb(token_id, orderbook)
            except Exception as e:
                logger.error(f"Error in orderbook update callback: {e}")

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

        # Notify all registered callbacks
        for cb in self._update_callbacks:
            try:
                cb(token_id, orderbook)
            except Exception as e:
                logger.error(f"Error in orderbook update callback: {e}")

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
        on_significant_update: Optional[Callable[[str, int, UnifiedOrderbook], None]] = None,
        ws_client: Optional[OrderbookWebSocket] = None,
    ):
        """
        Initialize orderbook manager.

        Args:
            config: WebSocket configuration
            clob_client: Optional ClobClient for REST API fallback
            on_significant_update: Optional callback for significant orderbook changes.
                                   Called with (token_id, bin_index, orderbook).
                                   Use this to trigger trading logic on WS updates.
            ws_client: Optional external OrderbookWebSocket (shared connection).
                      If provided, this manager won't create/destroy its own WS.
        """
        self.config = config
        self.clob_client = clob_client
        self.on_significant_update = on_significant_update

        # Track whether we own the WS (and should connect/disconnect it)
        self._ws_client: Optional[OrderbookWebSocket] = ws_client
        self._owns_ws = ws_client is None

        self._orderbooks: Dict[str, UnifiedOrderbook] = {}
        self._token_to_bin: Dict[str, int] = {}

        # Tracking for significant change detection
        self._last_best_prices: Dict[str, tuple] = {}  # token_id -> (best_bid, best_ask)

    async def start(self) -> None:
        """Start the orderbook manager."""
        if not self.config.enabled:
            logger.info("WebSocket disabled, using REST API polling")
            return

        if self._owns_ws:
            # Create and connect our own WS
            self._ws_client = OrderbookWebSocket(
                config=self.config,
                on_orderbook_update=self._on_ws_update,
            )
            await self._ws_client.connect()
        else:
            # External WS: just register our callback
            self._ws_client.register_callback(self._on_ws_update)

    async def stop(self) -> None:
        """Stop the orderbook manager."""
        if not self._ws_client:
            return

        if self._owns_ws:
            await self._ws_client.disconnect()
        else:
            # External WS: unregister callback + unsubscribe our tokens
            self._ws_client.unregister_callback(self._on_ws_update)
            own_tokens = list(self._token_to_bin.keys())
            if own_tokens:
                await self._ws_client.unsubscribe(own_tokens)
        self._orderbooks.clear()
        self._last_best_prices.clear()
        self._token_to_bin.clear()

    def _on_ws_update(self, token_id: str, orderbook: UnifiedOrderbook) -> None:
        """Handle orderbook update from WebSocket."""
        # Token filter: only process tokens this manager subscribed to
        if token_id not in self._token_to_bin:
            return

        old_orderbook = self._orderbooks.get(token_id)
        self._orderbooks[token_id] = orderbook

        # Check for significant change (best price changed)
        if self.on_significant_update:
            old_prices = self._last_best_prices.get(token_id)
            new_prices = (orderbook.best_yes_bid, orderbook.best_yes_ask)

            is_significant = (
                old_prices is None or
                old_prices != new_prices
            )

            if is_significant:
                self._last_best_prices[token_id] = new_prices
                bin_index = self._token_to_bin.get(token_id, 0)
                try:
                    self.on_significant_update(token_id, bin_index, orderbook)
                except Exception as e:
                    logger.error(f"Error in on_significant_update callback: {e}")

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

            # Handle both dict and object responses (OrderBookSummary)
            if hasattr(response, 'bids'):
                bids = response.bids or []
                asks = response.asks or []
            else:
                bids = response.get("bids", [])
                asks = response.get("asks", [])

            orderbook = UnifiedOrderbook.from_api_response(
                bin_index=bin_index,
                yes_token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )

            # Update both manager cache and WS cache so get_orderbook()
            # returns the fresh data (it prefers the WS cache).
            self._orderbooks[token_id] = orderbook
            if self._ws_client:
                self._ws_client._orderbooks[token_id] = orderbook
            return orderbook

        except Exception as e:
            error_str = str(e)
            # 404 means no orderbook exists - this is normal for settled/delisted bins
            if "404" in error_str or "No orderbook exists" in error_str:
                logger.debug(f"No orderbook for token {token_id[:16]}...")
            else:
                logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return None
