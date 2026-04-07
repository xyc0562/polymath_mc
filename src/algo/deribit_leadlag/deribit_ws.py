"""
Async Deribit WebSocket client and in-memory price store.

Subscribes to `ticker.{instrument}.100ms` channels (public, no auth required)
for real-time option price updates. PriceStore converts ticker snapshots
into DeribitOption objects for the existing implied probability pipeline.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Set, Tuple

import websockets

from .deribit_client import DeribitOption, parse_instrument_name

logger = logging.getLogger(__name__)


@dataclass
class TickerSnapshot:
    """A single ticker update from Deribit WS."""

    instrument_name: str
    best_bid_price: Optional[float]  # BTC-denominated
    best_bid_amount: Optional[float]
    best_ask_price: Optional[float]
    best_ask_amount: Optional[float]
    mark_price: float
    mark_iv: float  # Percentage (e.g., 65.0 for 65%)
    underlying_price: float  # Forward price in USD
    open_interest: float
    index_price: float
    last_price: Optional[float]
    timestamp_ms: int


class PriceStore:
    """
    In-memory store for Deribit ticker snapshots.

    Thread-safe for single-writer (WS listener) / multi-reader (strategy tick) pattern
    within asyncio (no true threads, but guards against mid-update reads).
    """

    def __init__(self):
        self._tickers: Dict[str, TickerSnapshot] = {}
        self._last_update_time: float = 0.0

    def update(self, ticker: TickerSnapshot) -> None:
        self._tickers[ticker.instrument_name] = ticker
        self._last_update_time = time.monotonic()

    def seconds_since_last_update(self) -> float:
        if self._last_update_time == 0.0:
            return float("inf")
        return time.monotonic() - self._last_update_time

    def load_initial(self, options: List[DeribitOption]) -> None:
        """Seed store from REST API snapshot at startup."""
        for opt in options:
            self._tickers[opt.instrument_name] = TickerSnapshot(
                instrument_name=opt.instrument_name,
                best_bid_price=opt.bid_price_btc,
                best_bid_amount=None,
                best_ask_price=opt.ask_price_btc,
                best_ask_amount=None,
                mark_price=opt.mark_price_btc,
                mark_iv=opt.mark_iv * 100.0,  # Back to percentage
                underlying_price=opt.underlying_price,
                open_interest=opt.open_interest,
                index_price=opt.underlying_price,
                last_price=None,
                timestamp_ms=int(time.time() * 1000),
            )
        self._last_update_time = time.monotonic()
        logger.info(f"PriceStore seeded with {len(options)} options from REST")

    def build_deribit_options(self) -> List[DeribitOption]:
        """Convert current ticker snapshots back into DeribitOption objects."""
        options = []
        for name, t in self._tickers.items():
            parsed = parse_instrument_name(name)
            if parsed is None:
                continue
            expiry_date, strike, opt_type = parsed

            mark_iv = t.mark_iv / 100.0 if t.mark_iv > 0 else 0.0

            options.append(
                DeribitOption(
                    instrument_name=name,
                    expiry_date=expiry_date,
                    strike=strike,
                    option_type=opt_type,
                    mark_iv=mark_iv,
                    underlying_price=t.underlying_price,
                    mark_price_btc=t.mark_price,
                    bid_price_btc=t.best_bid_price if t.best_bid_price and t.best_bid_price > 0 else None,
                    ask_price_btc=t.best_ask_price if t.best_ask_price and t.best_ask_price > 0 else None,
                    bid_iv=None,  # Not available from ticker.100ms
                    ask_iv=None,
                    volume_24h=0.0,  # Not tracked per-instrument via WS
                    open_interest=t.open_interest,
                )
            )
        return options

    @property
    def instrument_count(self) -> int:
        return len(self._tickers)


def _parse_ticker_data(data: dict) -> Optional[TickerSnapshot]:
    """Parse a Deribit ticker notification into TickerSnapshot."""
    try:
        return TickerSnapshot(
            instrument_name=data["instrument_name"],
            best_bid_price=data.get("best_bid_price"),
            best_bid_amount=data.get("best_bid_amount"),
            best_ask_price=data.get("best_ask_price"),
            best_ask_amount=data.get("best_ask_amount"),
            mark_price=data.get("mark_price", 0.0),
            mark_iv=data.get("mark_iv", 0.0),
            underlying_price=data.get("underlying_price", 0.0),
            open_interest=data.get("open_interest", 0.0),
            index_price=data.get("index_price", 0.0),
            last_price=data.get("last_price"),
            timestamp_ms=data.get("timestamp", int(time.time() * 1000)),
        )
    except (KeyError, TypeError) as e:
        logger.warning(f"Failed to parse ticker data: {e}")
        return None


class DeribitWebSocket:
    """
    Async Deribit WebSocket client for real-time option ticker updates.

    Uses `ticker.{instrument}.100ms` channels (public, no auth).
    Implements heartbeat via `public/set_heartbeat` and auto-reconnect
    with exponential backoff.
    """

    def __init__(
        self,
        price_store: PriceStore,
        on_emergency: Callable,
        ws_url: str = "wss://www.deribit.com/ws/api/v2",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
        stale_threshold: float = 5.0,
        heartbeat_interval: int = 10,
    ):
        self._price_store = price_store
        self._on_emergency = on_emergency
        self._ws_url = ws_url
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._stale_threshold = stale_threshold
        self._heartbeat_interval = heartbeat_interval

        self._ws = None
        self._subscribed_channels: Set[str] = set()
        self._target_instruments: List[str] = []
        self._running = False
        self._msg_id = 0
        self._listen_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def connect(self) -> None:
        """Start the WebSocket connection and background tasks."""
        self._running = True
        self._listen_task = asyncio.create_task(self._connection_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def disconnect(self) -> None:
        """Gracefully shut down."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def update_subscriptions(self, instruments: List[str]) -> None:
        """Update the set of instruments to subscribe to."""
        self._target_instruments = instruments
        if self._ws is not None:
            await self._sync_subscriptions()

    async def _connection_loop(self) -> None:
        """Main connection loop with exponential backoff reconnect."""
        delay = self._reconnect_delay
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=None,  # We handle heartbeat ourselves
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    delay = self._reconnect_delay  # Reset backoff on success
                    logger.info("Deribit WS connected")

                    # Enable heartbeat
                    await self._send_json({
                        "jsonrpc": "2.0",
                        "id": self._next_id(),
                        "method": "public/set_heartbeat",
                        "params": {"interval": self._heartbeat_interval},
                    })

                    # Subscribe to target instruments
                    await self._sync_subscriptions()

                    # Listen for messages
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        await self._handle_message(raw_msg)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Deribit WS error: {e}")
                self._ws = None
                self._subscribed_channels.clear()

            if not self._running:
                break

            logger.info(f"Deribit WS reconnecting in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)

    async def _handle_message(self, raw_msg: str) -> None:
        """Process a single WS message."""
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        # Heartbeat: test_request -> respond with public/test
        if msg.get("method") == "heartbeat":
            if msg.get("params", {}).get("type") == "test_request":
                await self._send_json({
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "public/test",
                    "params": {},
                })
            return

        # Subscription notification
        if msg.get("method") == "subscription":
            params = msg.get("params", {})
            data = params.get("data")
            if data:
                ticker = _parse_ticker_data(data)
                if ticker:
                    self._price_store.update(ticker)
            return

    async def _sync_subscriptions(self) -> None:
        """Synchronize WS subscriptions with target instruments."""
        target_channels = {
            f"ticker.{inst}.100ms" for inst in self._target_instruments
        }

        # Unsubscribe from channels we no longer need
        to_unsub = self._subscribed_channels - target_channels
        if to_unsub:
            await self._send_json({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "public/unsubscribe",
                "params": {"channels": list(to_unsub)},
            })
            self._subscribed_channels -= to_unsub

        # Subscribe to new channels
        to_sub = target_channels - self._subscribed_channels
        if to_sub:
            await self._send_json({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "public/subscribe",
                "params": {"channels": list(to_sub)},
            })
            self._subscribed_channels |= to_sub

        logger.info(
            f"Deribit WS subscriptions: {len(self._subscribed_channels)} channels "
            f"({len(to_sub)} added, {len(to_unsub)} removed)"
        )

    async def _send_json(self, payload: dict) -> None:
        """Send a JSON message to the WS."""
        if self._ws is not None:
            await self._ws.send(json.dumps(payload))

    async def _watchdog_loop(self) -> None:
        """Monitor for stale data and trigger emergency if needed."""
        while self._running:
            await asyncio.sleep(1.0)
            stale_seconds = self._price_store.seconds_since_last_update()
            if stale_seconds > self._stale_threshold and self._price_store.instrument_count > 0:
                logger.error(
                    f"Deribit data stale for {stale_seconds:.1f}s "
                    f"(threshold={self._stale_threshold}s) — triggering emergency"
                )
                try:
                    await self._on_emergency()
                except Exception as e:
                    logger.error(f"Emergency callback failed: {e}")
                # Wait before next emergency trigger
                await asyncio.sleep(self._stale_threshold)
