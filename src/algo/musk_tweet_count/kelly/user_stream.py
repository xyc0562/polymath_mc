"""
WebSocket client for Polymarket User Channel.

Subscribes to authenticated user events:
- Trade fills (order matched)
- Order updates (placement, cancellation)

Used to:
- Confirm order fills before updating portfolio
- Track open orders
- Detect order cancellations
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

# Polymarket User WebSocket endpoint
USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


class OrderStatus(Enum):
    """Order status from WebSocket events."""
    PENDING = "PENDING"
    MATCHED = "MATCHED"
    MINED = "MINED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


@dataclass
class PendingOrder:
    """Tracks a pending order awaiting fill confirmation."""
    order_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    bin_index: int
    created_at: float = field(default_factory=time.time)
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class FillEvent:
    """Represents a confirmed fill from WebSocket."""
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    status: OrderStatus
    timestamp: datetime


@dataclass
class OrderEvent:
    """Represents an order update from WebSocket."""
    order_id: str
    event_type: str  # "PLACEMENT", "CANCELLATION", etc.
    status: str
    size_matched: float
    original_size: float


class UserStreamClient:
    """
    WebSocket client for Polymarket User Channel.

    Provides real-time updates on:
    - Trade fills
    - Order status changes

    Usage:
        client = UserStreamClient(api_key, api_secret, api_passphrase)
        client.on_fill = my_fill_callback
        client.on_order = my_order_callback
        await client.start()
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
    ):
        """
        Initialize user stream client.

        Args:
            api_key: Polymarket API key
            api_secret: Polymarket API secret
            api_passphrase: Polymarket API passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase

        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

        # Callbacks
        self.on_fill: Optional[Callable[[FillEvent], None]] = None
        self.on_order: Optional[Callable[[OrderEvent], None]] = None
        self.on_connected: Optional[Callable[[], None]] = None
        self.on_disconnected: Optional[Callable[[], None]] = None

        # Pending orders tracking
        self._pending_orders: Dict[str, PendingOrder] = {}
        self._pending_lock = asyncio.Lock()

        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._stale_order_task: Optional[asyncio.Task] = None

        # Config
        self.stale_order_timeout_seconds: float = 30.0
        self.stale_order_check_interval: float = 5.0
        self.heartbeat_interval_seconds: float = 300.0  # 5 minutes

        # Heartbeat tracking
        self._connected_at: Optional[datetime] = None
        self._last_message_at: Optional[datetime] = None
        self._message_count: int = 0
        self._fill_count: int = 0
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _generate_auth_object(self) -> Dict[str, str]:
        """Generate authentication object for WebSocket subscription message."""
        return {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "passphrase": self.api_passphrase,
        }

    async def start(self) -> None:
        """Start the user stream client."""
        if self._running:
            return

        self._running = True

        # Start listener task
        self._listen_task = asyncio.create_task(
            self._listen_loop(),
            name="user_stream_listener"
        )

        # Start stale order cleanup task
        self._stale_order_task = asyncio.create_task(
            self._stale_order_loop(),
            name="stale_order_cleanup"
        )

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="user_stream_heartbeat"
        )

        logger.info("User stream client started")

    async def stop(self) -> None:
        """Stop the user stream client."""
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._stale_order_task:
            self._stale_order_task.cancel()
            try:
                await self._stale_order_task
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

        logger.info("User stream client stopped")

    async def _listen_loop(self) -> None:
        """Main WebSocket listen loop with reconnection."""
        # Brief delay on first connection to avoid contention with orderbook WS
        await asyncio.sleep(1.0)

        while self._running:
            try:
                await self._connect_and_listen()
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                # Reconnect with exponential backoff
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and listen for events."""
        logger.info(f"[UserWS] Connecting to {USER_WS_URL}...")

        async with websockets.connect(
            USER_WS_URL,
            open_timeout=20,  # Allow more time for initial handshake
            ping_interval=30,
            ping_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connect
            self._connected_at = datetime.now()
            self._message_count = 0
            self._fill_count = 0

            logger.info("[UserWS] Connected to Polymarket User WebSocket")

            if self.on_connected:
                self.on_connected()

            # Subscribe to user channel with authentication in the message
            # Polymarket expects auth in the subscription message, not HTTP headers
            subscribe_msg = json.dumps({
                "type": "user",
                "auth": self._generate_auth_object(),
            })
            logger.info("[UserWS] Sending subscription message...")
            await ws.send(subscribe_msg)

            # Listen for messages
            async for message in ws:
                await self._handle_message(message)

        if self.on_disconnected:
            self.on_disconnected()

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            self._last_message_at = datetime.now()
            self._message_count += 1

            data = json.loads(message)
            await self._process_data(data)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _process_data(self, data: Any) -> None:
        """Recursively process data, handling nested lists."""
        if isinstance(data, list):
            # Recursively process each item in the list
            for item in data:
                await self._process_data(item)
        elif isinstance(data, dict):
            await self._handle_single_message(data)
        # Skip primitives (strings, numbers, None) - not meaningful events

    async def _handle_single_message(self, data: Dict[str, Any]) -> None:
        """Handle a single message dict."""
        event_type = data.get("event_type")

        if event_type == "trade":
            await self._handle_trade_event(data)
        elif event_type == "order":
            await self._handle_order_event(data)
        elif data.get("type") == "subscribed":
            logger.info("[UserWS] Subscription CONFIRMED - ready to receive fill events")
        elif data.get("type") == "pong":
            pass  # Heartbeat response
        else:
            logger.debug(f"[UserWS] Unknown event: {data}")

    async def _handle_trade_event(self, data: Dict[str, Any]) -> None:
        """Handle trade fill event."""
        try:
            order_id = data.get("taker_order_id") or data.get("order_id")
            status_str = data.get("status", "")

            fill = FillEvent(
                order_id=order_id,
                token_id=data.get("asset_id", ""),
                side=data.get("side", ""),
                price=float(data.get("price", 0)),
                size=float(data.get("size", 0)),
                status=OrderStatus(status_str) if status_str else OrderStatus.MATCHED,
                timestamp=datetime.utcnow(),
            )

            # Log prominently so fills are visible in logs
            logger.info(
                f"[FILL RECEIVED] order={order_id[:16] if order_id else 'N/A'}..., "
                f"side={fill.side}, size={fill.size:.2f} @ {fill.price:.4f}, "
                f"status={fill.status.value}, token={fill.token_id[:16] if fill.token_id else 'N/A'}..."
            )

            # Update pending order
            # NOTE: Same fill may arrive multiple times with different statuses (MATCHED -> MINED -> CONFIRMED)
            # Each callback reports the SAME fill.size, not incremental, so we must not double-count
            async with self._pending_lock:
                if order_id and order_id in self._pending_orders:
                    pending = self._pending_orders[order_id]

                    # Only add to filled_size on first fill callback (MATCHED)
                    # Subsequent callbacks (MINED, CONFIRMED) are status updates only
                    if pending.status == OrderStatus.PENDING or pending.filled_size == 0:
                        # First fill notification - add the size
                        pending.filled_size = fill.size  # Use SET not ADD to be safe
                        logger.debug(f"First fill for order {order_id[:16]}...: {fill.size:.2f} shares")
                    else:
                        # Status update only - already counted this fill
                        logger.debug(f"Status update for order {order_id[:16]}...: {pending.status.value} -> {fill.status.value}")

                    pending.status = fill.status

                    # Remove if fully filled or terminal status
                    if (pending.filled_size >= pending.size or
                        fill.status in (OrderStatus.CONFIRMED, OrderStatus.FAILED)):
                        del self._pending_orders[order_id]

            # Callback
            if self.on_fill:
                self._fill_count += 1
                logger.debug(f"[FILL ROUTING] Invoking on_fill callback for order {order_id[:16] if order_id else 'N/A'}...")
                self.on_fill(fill)
            else:
                logger.warning(
                    f"[FILL DROPPED] No on_fill callback set! Fill for order {order_id[:16] if order_id else 'N/A'}... "
                    f"will not be processed. This indicates a configuration issue."
                )

        except Exception as e:
            logger.error(f"Error handling trade event: {e}", exc_info=True)

    async def _handle_order_event(self, data: Dict[str, Any]) -> None:
        """Handle order update event."""
        try:
            order_id = data.get("order_id", "")

            event = OrderEvent(
                order_id=order_id,
                event_type=data.get("type", ""),
                status=data.get("status", ""),
                size_matched=float(data.get("size_matched", 0)),
                original_size=float(data.get("original_size", 0)),
            )

            logger.debug(
                f"Order event: order={order_id[:16]}..., "
                f"type={event.event_type}, status={event.status}"
            )

            # Handle cancellation
            if event.event_type == "CANCELLATION" or event.status == "CANCELLED":
                async with self._pending_lock:
                    if order_id in self._pending_orders:
                        del self._pending_orders[order_id]
                        logger.info(f"Order {order_id[:16]}... cancelled")

            # Callback
            if self.on_order:
                self.on_order(event)

        except Exception as e:
            logger.error(f"Error handling order event: {e}")

    async def _stale_order_loop(self) -> None:
        """
        Background loop to detect and cancel stale orders.

        Non-blocking: runs independently of main trading loop.
        """
        while self._running:
            try:
                await asyncio.sleep(self.stale_order_check_interval)
                await self._check_stale_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stale order loop: {e}")

    async def _check_stale_orders(self) -> None:
        """Check for and handle stale orders."""
        now = time.time()
        stale_orders: List[PendingOrder] = []

        async with self._pending_lock:
            for order_id, pending in list(self._pending_orders.items()):
                age = now - pending.created_at
                if age > self.stale_order_timeout_seconds:
                    stale_orders.append(pending)

        for pending in stale_orders:
            logger.warning(
                f"Stale order detected: {pending.order_id[:16]}..., "
                f"age={now - pending.created_at:.1f}s, "
                f"filled={pending.filled_size}/{pending.size}"
            )

            # Emit callback for stale order handling
            # The executor should cancel this order
            if self.on_stale_order:
                await self.on_stale_order(pending)

    # Callback for stale orders (set by executor)
    on_stale_order: Optional[Callable[["PendingOrder"], Any]] = None

    async def _heartbeat_loop(self) -> None:
        """
        Background loop to log heartbeat status every 5 minutes.

        Shows that the user WebSocket is still connected and functioning.
        """
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval_seconds)

                if not self._connected_at:
                    continue

                # Calculate uptime
                uptime = datetime.now() - self._connected_at
                uptime_mins = uptime.total_seconds() / 60

                # Last message age
                if self._last_message_at:
                    last_msg_age = (datetime.now() - self._last_message_at).total_seconds()
                    last_msg_str = f"{last_msg_age:.0f}s ago"
                else:
                    last_msg_str = "none"

                # Connection status (use getattr for compatibility with different websocket libs)
                connected = self._ws is not None and not getattr(self._ws, 'closed', True)

                # Pending orders
                pending_count = len(self._pending_orders)

                logger.info(
                    f"[UserWS] Heartbeat: connected={connected}, "
                    f"uptime={uptime_mins:.1f}m, msgs={self._message_count}, "
                    f"fills={self._fill_count}, pending={pending_count}, "
                    f"last_msg={last_msg_str}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def add_pending_order(self, order: PendingOrder) -> None:
        """Add an order to pending tracking."""
        async with self._pending_lock:
            self._pending_orders[order.order_id] = order
            logger.debug(f"Added pending order: {order.order_id[:16]}...")

    async def remove_pending_order(self, order_id: str) -> Optional[PendingOrder]:
        """Remove an order from pending tracking."""
        async with self._pending_lock:
            return self._pending_orders.pop(order_id, None)

    async def get_pending_order(self, order_id: str) -> Optional[PendingOrder]:
        """Get a pending order by ID."""
        async with self._pending_lock:
            return self._pending_orders.get(order_id)

    def get_pending_orders_count(self) -> int:
        """Get count of pending orders."""
        return len(self._pending_orders)

    def get_pending_collateral(self) -> float:
        """Get total collateral locked in pending orders."""
        total = 0.0
        for order in self._pending_orders.values():
            if order.side == "BUY":
                # Buy orders lock collateral = price * remaining_size
                remaining = order.size - order.filled_size
                total += order.price * remaining
        return total

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status for debugging.

        Returns dict with:
        - connected: bool, whether WebSocket is connected
        - connected_at: datetime or None
        - uptime_seconds: float, seconds since connection
        - last_message_at: datetime or None
        - last_message_age_seconds: float, seconds since last message
        - message_count: int, total messages received
        - fill_count: int, total fills received
        - pending_orders: int, number of pending orders
        - on_fill_callback_set: bool, whether callback is configured
        """
        now = datetime.now()
        connected = self._ws is not None and not getattr(self._ws, 'closed', True)

        uptime_seconds = 0.0
        if self._connected_at:
            uptime_seconds = (now - self._connected_at).total_seconds()

        last_message_age = float('inf')
        if self._last_message_at:
            last_message_age = (now - self._last_message_at).total_seconds()

        return {
            "connected": connected,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "uptime_seconds": round(uptime_seconds, 1),
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "last_message_age_seconds": round(last_message_age, 1) if last_message_age != float('inf') else None,
            "message_count": self._message_count,
            "fill_count": self._fill_count,
            "pending_orders": len(self._pending_orders),
            "on_fill_callback_set": self.on_fill is not None,
        }

    def log_connection_status(self) -> None:
        """Log current connection status for debugging."""
        status = self.get_connection_status()
        logger.info(
            f"[UserWS STATUS] connected={status['connected']}, "
            f"uptime={status['uptime_seconds']}s, "
            f"msgs={status['message_count']}, "
            f"fills={status['fill_count']}, "
            f"pending={status['pending_orders']}, "
            f"callback_set={status['on_fill_callback_set']}, "
            f"last_msg_age={status['last_message_age_seconds']}s"
        )
