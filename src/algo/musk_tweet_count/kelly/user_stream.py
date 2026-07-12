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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

logger = logging.getLogger(__name__)

# Polymarket User WebSocket endpoint
USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
APP_PING_MESSAGE = "PING"
APP_PONG_MESSAGE = "PONG"
DEFAULT_KEEPALIVE_INTERVAL_SECONDS = 10.0


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
    condition_id: str = ""  # Market condition ID for event isolation
    created_at: float = field(default_factory=time.time)
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    # Per-order staleness override in seconds. Resting maker (GTD) orders
    # legitimately outlive the FAK default; set this to TTL + grace so the
    # stale janitor only fires after exchange-side expiration.
    stale_after: Optional[float] = None
    # True for resting (GTD maker) orders: a CONFIRMED partial fill must
    # not stop tracking, because the remainder is still live on the book
    # and later fills need to stay attributable.
    resting: bool = False
    _seen_fills: set = field(default_factory=set, repr=False)  # Dedup keys for partial fills


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
    match_id: str = ""  # Unique trade/match ID from Polymarket for dedup


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
        self._ws: Optional[ClientConnection] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._send_lock = asyncio.Lock()
        self._market_ids: Set[str] = set()
        self._subscribed_market_ids: Set[str] = set()

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
        self._keepalive_task: Optional[asyncio.Task] = None

        # App-level heartbeat required by Polymarket's WS docs.
        self.keepalive_interval_seconds: float = DEFAULT_KEEPALIVE_INTERVAL_SECONDS

    def _generate_auth_object(self) -> Dict[str, str]:
        """Generate authentication object for WebSocket subscription message."""
        return {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "passphrase": self.api_passphrase,
        }

    def _build_subscription_message(self) -> str:
        """Build the authenticated user-channel subscription payload."""
        return json.dumps({
            "type": "user",
            "auth": self._generate_auth_object(),
            "markets": sorted(self._market_ids),
        })

    @staticmethod
    def _build_market_update_message(
        market_ids: Iterable[str],
        operation: str,
    ) -> str:
        """Build a dynamic subscription update for an open user-channel socket."""
        return json.dumps({
            "markets": sorted({market_id for market_id in market_ids if market_id}),
            "operation": operation,
        })

    def _is_ws_open(self) -> bool:
        """Return True when the websocket connection is fully open."""
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

    async def _send_message(self, message: str) -> None:
        """Serialize websocket writes to avoid concurrent send races."""
        if not self._is_ws_open():
            return

        async with self._send_lock:
            if self._ws is not None:
                await self._ws.send(message)

    async def _send_subscription(self) -> None:
        """Send the current user-channel subscription payload."""
        logger.info(
            "[UserWS] Sending subscription message for %d market(s)...",
            len(self._market_ids),
        )
        await self._send_message(self._build_subscription_message())
        self._subscribed_market_ids = set(self._market_ids)

    async def _send_market_update(
        self,
        market_ids: Iterable[str],
        operation: str,
    ) -> None:
        """Send a documented subscribe/unsubscribe delta for an open socket."""
        normalized = sorted({market_id for market_id in market_ids if market_id})
        if not normalized:
            return

        logger.info(
            "[UserWS] Sending %s update for %d market(s)...",
            operation,
            len(normalized),
        )
        await self._send_message(
            self._build_market_update_message(normalized, operation)
        )

    async def set_markets(self, market_ids: Iterable[str]) -> None:
        """
        Update the user-channel market filter.

        Polymarket supports dynamic subscribe/unsubscribe updates on open sockets.
        """
        normalized = {market_id for market_id in market_ids if market_id}
        if normalized == self._market_ids:
            return

        previous_markets = set(self._market_ids)
        self._market_ids = normalized
        logger.info("[UserWS] Tracking %d market(s) for user events", len(self._market_ids))

        if self._is_ws_open():
            added_markets = self._market_ids - self._subscribed_market_ids
            removed_markets = self._subscribed_market_ids - self._market_ids

            # Subscribe first so newly tracked markets start streaming immediately.
            await self._send_market_update(added_markets, "subscribe")
            await self._send_market_update(removed_markets, "unsubscribe")
            self._subscribed_market_ids = set(self._market_ids)
        else:
            self._subscribed_market_ids = set(previous_markets)

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

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="user_stream_keepalive"
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

        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
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
                logger.warning(
                    "[UserWS] WebSocket connection closed: %s",
                    self._format_close_details(e),
                )
            except Exception as e:
                logger.error(f"[UserWS] WebSocket error: {e}")

            if self._running:
                # Reconnect with exponential backoff
                logger.info(f"[UserWS] Reconnecting in {self._reconnect_delay}s...")
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
            close_timeout=10,
            ping_interval=None,
            ping_timeout=None,
        ) as ws:
            self._ws = ws
            self._subscribed_market_ids = set()
            self._reconnect_delay = 1.0  # Reset on successful connect
            self._connected_at = datetime.now(timezone.utc)
            self._last_message_at = None
            self._message_count = 0
            self._fill_count = 0

            logger.info("[UserWS] Connected to Polymarket User WebSocket")

            if self.on_connected:
                self.on_connected()

            await self._send_subscription()

            # Listen for messages
            async for message in ws:
                await self._handle_message(message)

            if self._running:
                logger.warning(
                    "[UserWS] WebSocket closed by server (code=%s, reason=%s)",
                    getattr(ws, "close_code", None),
                    getattr(ws, "close_reason", None) or "none",
                )

        self._ws = None
        self._subscribed_market_ids = set()
        if self.on_disconnected:
            self.on_disconnected()

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            self._last_message_at = datetime.now(timezone.utc)
            self._message_count += 1

            raw = message.strip() if isinstance(message, str) else message
            if raw == APP_PONG_MESSAGE:
                return
            if raw == APP_PING_MESSAGE:
                logger.debug("[UserWS] Received unexpected PING from server")
                return
            if not raw:
                return

            data = json.loads(raw)
            await self._process_data(data)

        except json.JSONDecodeError:
            logger.warning(f"[UserWS] Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"[UserWS] Error handling message: {e}")

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
            logger.info(
                "[UserWS] Subscription confirmed for %d market(s)",
                len(self._market_ids),
            )
        elif data.get("type") == "pong":
            pass  # Heartbeat response
        elif data.get("type") == "error":
            logger.warning("[UserWS] Server error: %s", data)
        else:
            logger.debug(f"[UserWS] Unknown event: {data}")

    async def _handle_trade_event(self, data: Dict[str, Any]) -> None:
        """Handle trade fill event.

        A trade event describes the TAKER's trade at the top level; when one
        of OUR resting (maker) orders is on the passive side, our order id
        and matched portion live in the maker_orders entries. One FillEvent
        is dispatched per involved order of ours: the taker fill exactly as
        before, plus one per recognized maker entry.
        """
        try:
            status_str = data.get("status", "")
            status = OrderStatus(status_str) if status_str else OrderStatus.MATCHED
            timestamp = datetime.now(timezone.utc)

            # Extract unique match/trade ID for deduplication
            # Polymarket sends id/match_id on trade events
            trade_id = data.get("id") or data.get("match_id") or ""

            taker_order_id = data.get("taker_order_id") or data.get("order_id")
            fills = [
                FillEvent(
                    order_id=taker_order_id,
                    token_id=data.get("asset_id", ""),
                    side=data.get("side", ""),
                    price=float(data.get("price", 0)),
                    size=float(data.get("size", 0)),
                    status=status,
                    timestamp=timestamp,
                    match_id=trade_id,
                )
            ]

            # Maker-side attribution: the top-level size/price describe the
            # taker's trade (possibly spanning several makers). Our portion
            # is the maker entry's matched amount. Ownership check = the
            # order is tracked as pending; counterparty entries are skipped.
            for entry in data.get("maker_orders") or []:
                if not isinstance(entry, dict):
                    continue
                maker_order_id = entry.get("order_id") or ""
                if not maker_order_id or maker_order_id == taker_order_id:
                    continue
                async with self._pending_lock:
                    pending = self._pending_orders.get(maker_order_id)
                if pending is None:
                    logger.debug(
                        f"Skipping counterparty maker entry {maker_order_id[:16]}..."
                    )
                    continue
                matched = float(entry.get("matched_amount") or entry.get("size") or 0)
                if matched <= 0:
                    logger.warning(
                        f"Maker entry for our order {maker_order_id[:16]}... has no "
                        f"matched amount; skipping (payload keys: {sorted(entry.keys())})"
                    )
                    continue
                fills.append(
                    FillEvent(
                        order_id=maker_order_id,
                        token_id=entry.get("asset_id") or pending.token_id,
                        side=pending.side,
                        price=float(entry.get("price") or data.get("price") or 0),
                        size=matched,
                        status=status,
                        timestamp=timestamp,
                        # The trade id is shared by every fill in this trade;
                        # suffix our order id so two of our orders filled by
                        # one taker sweep don't dedup each other away.
                        match_id=f"{trade_id}:{maker_order_id}" if trade_id else "",
                    )
                )

            for fill in fills:
                await self._dispatch_fill(fill)

        except Exception as e:
            logger.error(f"Error handling trade event: {e}", exc_info=True)

    async def _dispatch_fill(self, fill: FillEvent) -> None:
        """Update pending-order tracking for a fill and invoke the callback."""
        order_id = fill.order_id

        # Log prominently so fills are visible in logs
        logger.info(
            f"[FILL RECEIVED] order={order_id[:16] if order_id else 'N/A'}..., "
            f"side={fill.side}, size={fill.size:.2f} @ {fill.price:.4f}, "
            f"status={fill.status.value}, token={fill.token_id[:16] if fill.token_id else 'N/A'}..."
        )

        # Update pending order
        # Fill deduplication: same fill arrives multiple times with escalating statuses
        # (MATCHED -> MINED -> CONFIRMED). We also handle genuine partial fills where
        # different chunks fill at different times (distinct match_id or size).
        async with self._pending_lock:
            if order_id and order_id in self._pending_orders:
                pending = self._pending_orders[order_id]

                # Token ID validation: ensure fill belongs to this order's market
                if fill.token_id and pending.token_id and fill.token_id != pending.token_id:
                    logger.warning(
                        f"Token mismatch for order {order_id[:16]}...: "
                        f"fill token={fill.token_id[:16]}... != pending token={pending.token_id[:16]}..."
                    )
                    # Still update status but don't count fill size
                    pending.status = fill.status
                else:
                    # Dedup key: use match_id if available, fall back to (size, price) tuple
                    if fill.match_id:
                        dedup_key = fill.match_id
                    else:
                        dedup_key = (fill.size, fill.price)

                    if dedup_key not in pending._seen_fills:
                        # Genuinely new fill — accumulate
                        pending._seen_fills.add(dedup_key)
                        pending.filled_size += fill.size
                        logger.debug(
                            f"New fill for order {order_id[:16]}...: +{fill.size:.2f} shares "
                            f"(total filled: {pending.filled_size:.2f}/{pending.size:.2f})"
                        )
                    else:
                        # Status escalation of already-counted fill
                        logger.debug(
                            f"Status update for order {order_id[:16]}...: "
                            f"{pending.status.value} -> {fill.status.value}"
                        )

                    pending.status = fill.status

                # Remove if fully filled or terminal. A resting (GTD maker)
                # order survives a CONFIRMED partial fill: the remainder is
                # still live at the exchange, and later fills must still be
                # attributable via this registry.
                if (pending.filled_size >= pending.size or
                    fill.status == OrderStatus.FAILED or
                    (fill.status == OrderStatus.CONFIRMED and not pending.resting)):
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
                timeout = (
                    pending.stale_after
                    if pending.stale_after is not None
                    else self.stale_order_timeout_seconds
                )
                if age > timeout:
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

            # Always remove from pending tracking after handling.
            # For FAK orders the unfilled remainder is already killed by the exchange,
            # so keeping it in _pending_orders just causes infinite stale warnings.
            async with self._pending_lock:
                self._pending_orders.pop(pending.order_id, None)

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
                uptime = datetime.now(timezone.utc) - self._connected_at
                uptime_mins = uptime.total_seconds() / 60

                # Last message age
                if self._last_message_at:
                    last_msg_age = (datetime.now(timezone.utc) - self._last_message_at).total_seconds()
                    last_msg_str = f"{last_msg_age:.0f}s ago"
                else:
                    last_msg_str = "none"

                connected = self._is_ws_open()

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

    async def _keepalive_loop(self) -> None:
        """Send Polymarket-required app-level PING frames on the user socket."""
        while self._running:
            try:
                await asyncio.sleep(self.keepalive_interval_seconds)

                if not self._is_ws_open():
                    continue

                await self._send_message(APP_PING_MESSAGE)

            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                logger.warning(
                    "[UserWS] Keepalive ping failed: %s",
                    self._format_close_details(e),
                )
            except Exception as e:
                logger.warning(f"[UserWS] Keepalive ping failed: {e}")

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
        now = datetime.now(timezone.utc)
        connected = self._is_ws_open()

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
