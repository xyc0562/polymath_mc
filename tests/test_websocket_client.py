"""Tests for OrderbookWebSocket and OrderbookManager shared-connection changes."""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

from websockets.protocol import State

from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.websocket_client import (
    APP_PING_MESSAGE,
    APP_PONG_MESSAGE,
    OrderbookManager,
    OrderbookWebSocket,
    WebSocketConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyWS:
    """Minimal mock WebSocket connection."""

    def __init__(self, state: State = State.OPEN):
        self.state = state
        self.sent: list[str] = []
        self._closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self._closed = True
        self.state = State.CLOSED

    async def recv(self) -> str:
        await asyncio.sleep(999)


def _make_ws(config: WebSocketConfig | None = None) -> OrderbookWebSocket:
    return OrderbookWebSocket(config=config or WebSocketConfig(enabled=True))


def _make_book_data(token_id: str = "tok_abc") -> dict:
    return {
        "event_type": "book",
        "asset_id": token_id,
        "bids": [{"price": "0.40", "size": "100"}],
        "asks": [{"price": "0.60", "size": "200"}],
    }


def _make_price_change_data(token_id: str = "tok_abc") -> dict:
    return {
        "event_type": "price_change",
        "asset_id": token_id,
        "changes": [{"side": "BUY", "price": "0.42", "size": "50"}],
    }


# ===========================================================================
# OrderbookWebSocket tests
# ===========================================================================


class TestRegisterUnregisterCallback:
    def test_register_two_callbacks_both_called(self):
        ws = _make_ws()
        ws._token_to_bin["tok_abc"] = 0
        results_a, results_b = [], []
        cb_a = lambda tid, ob: results_a.append(tid)
        cb_b = lambda tid, ob: results_b.append(tid)

        ws.register_callback(cb_a)
        ws.register_callback(cb_b)
        ws._handle_book_update(_make_book_data())

        assert results_a == ["tok_abc"]
        assert results_b == ["tok_abc"]

    def test_unregister_one_only_remaining_called(self):
        ws = _make_ws()
        ws._token_to_bin["tok_abc"] = 0
        results_a, results_b = [], []
        cb_a = lambda tid, ob: results_a.append(tid)
        cb_b = lambda tid, ob: results_b.append(tid)

        ws.register_callback(cb_a)
        ws.register_callback(cb_b)
        ws.unregister_callback(cb_a)
        ws._handle_book_update(_make_book_data())

        assert results_a == []
        assert results_b == ["tok_abc"]

    def test_unregister_nonexistent_is_noop(self):
        ws = _make_ws()
        ws.unregister_callback(lambda t, o: None)  # Should not raise

    def test_backward_compat_constructor_callback(self):
        results = []
        cb = lambda tid, ob: results.append(tid)
        ws = OrderbookWebSocket(config=WebSocketConfig(), on_orderbook_update=cb)
        ws._token_to_bin["tok_abc"] = 0
        ws._handle_book_update(_make_book_data())
        assert results == ["tok_abc"]


class TestSilenceBasedLiveness:
    def test_stale_message_reported_in_health(self):
        ws = _make_ws()
        ws._last_message_time = time.time() - 60.0
        status = ws.get_connection_status()
        assert status["last_message_age_seconds"] >= 59.0

    def test_message_updates_timestamp(self):
        ws = _make_ws()
        ws._last_message_time = time.time() - 60.0
        ws._last_message_time = time.time()
        status = ws.get_connection_status()
        assert status["last_message_age_seconds"] < 2.0

    def test_silence_timeout_triggers_ws_close(self):
        """Heartbeat loop should close WS when no messages received for too long."""
        config = WebSocketConfig(heartbeat_interval_seconds=0.01)
        ws = OrderbookWebSocket(config=config)
        dummy = DummyWS(State.OPEN)
        ws._ws = dummy
        ws._connected = True
        ws._running = True
        ws._last_message_time = time.time() - 120.0

        async def _run():
            task = asyncio.create_task(ws._heartbeat_loop())
            await asyncio.sleep(0.1)
            ws._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        assert dummy._closed is True


class TestSubscribeUnsubscribeState:
    def test_subscribe_updates_state(self):
        ws = _make_ws()
        ws._ws = DummyWS()
        ws._connected = True

        asyncio.run(ws.subscribe(["tok_a", "tok_b"], bin_indices=[0, 1]))

        assert "tok_a" in ws._subscribed_tokens
        assert "tok_b" in ws._subscribed_tokens
        assert ws._token_to_bin["tok_a"] == 0
        assert ws._token_to_bin["tok_b"] == 1

    def test_unsubscribe_updates_state(self):
        ws = _make_ws()
        ws._ws = DummyWS()
        ws._connected = True

        asyncio.run(ws.subscribe(["tok_a", "tok_b"], bin_indices=[0, 1]))
        asyncio.run(ws.unsubscribe(["tok_a"]))

        assert "tok_a" not in ws._subscribed_tokens
        assert "tok_b" in ws._subscribed_tokens

    def test_unsubscribe_when_disconnected_clears_local_state(self):
        ws = _make_ws()
        ws._subscribed_tokens = {"tok_a", "tok_b"}
        ws._pending_subscriptions = {"tok_a"}
        ws._token_to_bin = {"tok_a": 0, "tok_b": 1}
        ws._orderbooks["tok_a"] = UnifiedOrderbook(bin_index=0, yes_token_id="tok_a")
        ws._connected = False
        ws._ws = None

        asyncio.run(ws.unsubscribe(["tok_a"]))

        assert "tok_a" not in ws._subscribed_tokens
        assert "tok_a" not in ws._pending_subscriptions
        assert "tok_a" not in ws._token_to_bin
        assert "tok_a" not in ws._orderbooks
        assert "tok_b" in ws._subscribed_tokens

    def test_subscribe_when_disconnected_queues(self):
        ws = _make_ws()
        ws._connected = False

        asyncio.run(ws.subscribe(["tok_a"]))

        assert "tok_a" in ws._pending_subscriptions
        assert "tok_a" not in ws._subscribed_tokens


class TestSingleSubscribeAll:
    def test_subscribe_sends_all_tokens_in_one_message(self):
        """_subscribe sends ALL subscribed tokens (existing + new) in a single message."""
        ws = _make_ws()
        dummy = DummyWS()
        ws._ws = dummy
        ws._connected = True

        # Pre-existing subscriptions
        ws._subscribed_tokens = {"tok_existing_1", "tok_existing_2"}

        new_tokens = ["tok_new_1", "tok_new_2", "tok_new_3"]
        asyncio.run(ws._subscribe(new_tokens))

        # Should send exactly 1 message containing all 5 tokens
        assert len(dummy.sent) == 1
        msg = json.loads(dummy.sent[0])
        assert msg["type"] == "MARKET"
        assert set(msg["assets_ids"]) == {
            "tok_existing_1", "tok_existing_2",
            "tok_new_1", "tok_new_2", "tok_new_3",
        }


    def test_second_subscribe_triggers_reconnect(self):
        """Second _subscribe call should queue tokens and close WS for reconnect."""
        ws = _make_ws()
        dummy = DummyWS()
        ws._ws = dummy
        ws._connected = True

        # First subscribe — goes through normally
        asyncio.run(ws._subscribe(["tok_a", "tok_b"]))
        assert ws._subscribe_sent is True
        assert len(dummy.sent) == 1

        # Second subscribe — should close WS instead of sending
        asyncio.run(ws._subscribe(["tok_c"]))
        assert dummy._closed is True
        # New tokens are added to _subscribed_tokens for reconnect
        assert "tok_c" in ws._subscribed_tokens


class TestForceResubscribe:
    def test_force_resubscribe_closes_ws(self):
        """force_resubscribe() should close the WS so the listen loop reconnects."""
        ws = _make_ws()
        dummy = DummyWS()
        ws._ws = dummy
        ws._connected = True
        ws._subscribed_tokens = {"tok_a", "tok_b"}

        asyncio.run(ws.force_resubscribe())
        assert dummy._closed is True

    def test_force_resubscribe_skipped_when_no_tokens(self):
        """force_resubscribe() is a no-op if no tokens remain."""
        ws = _make_ws()
        dummy = DummyWS()
        ws._ws = dummy
        ws._connected = True
        ws._subscribed_tokens = set()

        asyncio.run(ws.force_resubscribe())
        assert dummy._closed is False

    def test_unsubscribe_then_force_resubscribe(self):
        """Unsubscribe removes tokens locally, force_resubscribe closes WS."""
        ws = _make_ws()
        dummy = DummyWS()
        ws._ws = dummy
        ws._connected = True
        ws._subscribed_tokens = {"tok_a", "tok_b", "tok_c"}
        ws._token_to_bin = {"tok_a": 0, "tok_b": 1, "tok_c": 2}

        asyncio.run(ws.unsubscribe(["tok_a", "tok_b"]))
        assert ws._subscribed_tokens == {"tok_c"}

        asyncio.run(ws.force_resubscribe())
        assert dummy._closed is True


class TestReconnectDelay:
    def test_initial_delay_uses_config(self):
        config = WebSocketConfig(reconnect_delay_seconds=7.0)
        ws = OrderbookWebSocket(config=config)
        assert ws._reconnect_delay == 7.0

    def test_default_delay_is_5(self):
        ws = _make_ws()
        assert ws._reconnect_delay == 5.0


class TestCacheClearedOnReconnect:
    def test_reconnect_clears_orderbook_cache(self):
        ws = _make_ws()
        ws._running = True

        # Pre-populate cache
        ws._orderbooks["tok_a"] = UnifiedOrderbook(bin_index=0, yes_token_id="tok_a")
        ws._subscribed_tokens.add("tok_a")

        call_count = 0

        async def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("fail")
            return DummyWS()

        with patch("src.algo.musk_tweet_count.kelly.websocket_client.websockets.connect", side_effect=mock_connect):
            ws._reconnect_delay = 0.01
            ws._max_reconnect_delay = 0.02
            asyncio.run(ws._reconnect())

        assert ws._orderbooks == {}


class TestBookUpdateParsing:
    def test_book_update_creates_orderbook(self):
        ws = _make_ws()
        ws._token_to_bin["tok_abc"] = 3

        ws._handle_book_update(_make_book_data())

        ob = ws._orderbooks.get("tok_abc")
        assert ob is not None
        assert ob.bin_index == 3
        assert ob.yes_token_id == "tok_abc"
        assert len(ob.yes_bids) == 1
        assert ob.yes_bids[0].price == 0.40
        assert ob.yes_bids[0].size == 100.0
        assert len(ob.yes_asks) == 1
        assert ob.yes_asks[0].price == 0.60

    def test_book_update_ignores_missing_asset_id(self):
        ws = _make_ws()
        ws._handle_book_update({"event_type": "book"})
        assert len(ws._orderbooks) == 0


class TestPriceChangeParsing:
    def test_price_change_updates_existing_orderbook(self):
        ws = _make_ws()
        ws._token_to_bin["tok_abc"] = 0

        ws._handle_book_update(_make_book_data())
        ws._handle_price_change(_make_price_change_data())

        ob = ws._orderbooks["tok_abc"]
        bid_prices = [level.price for level in ob.yes_bids]
        assert 0.42 in bid_prices

    def test_price_change_remove_level(self):
        ws = _make_ws()
        ws._token_to_bin["tok_abc"] = 0
        ws._handle_book_update(_make_book_data())

        ws._handle_price_change({
            "event_type": "price_change",
            "asset_id": "tok_abc",
            "changes": [{"side": "BUY", "price": "0.40", "size": "0"}],
        })

        ob = ws._orderbooks["tok_abc"]
        assert len(ob.yes_bids) == 0

    def test_price_change_ignored_for_unknown_token(self):
        ws = _make_ws()
        ws._handle_price_change(_make_price_change_data())


class TestConnectionStatus:
    def test_status_when_disconnected(self):
        ws = _make_ws()
        status = ws.get_connection_status()
        assert status["connected"] is False
        assert status["subscribed_tokens"] == 0
        assert status["cached_orderbooks"] == 0

    def test_status_when_connected(self):
        ws = _make_ws()
        ws._ws = DummyWS(State.OPEN)
        ws._connected = True
        ws._subscribed_tokens = {"a", "b", "c"}
        ws._orderbooks = {"a": MagicMock(), "b": MagicMock()}

        status = ws.get_connection_status()
        assert status["connected"] is True
        assert status["subscribed_tokens"] == 3
        assert status["cached_orderbooks"] == 2


# ===========================================================================
# OrderbookManager tests
# ===========================================================================


class TestExternalWSNoOwnConnection:
    def test_external_ws_start_registers_callback(self):
        shared_ws = _make_ws()
        manager = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )

        asyncio.run(manager.start())

        assert manager._on_ws_update in shared_ws._update_callbacks
        assert not manager._owns_ws

    def test_external_ws_stop_unsubscribes_but_keeps_ws(self):
        shared_ws = _make_ws()
        shared_ws._ws = DummyWS()
        shared_ws._connected = True

        manager = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )

        async def _run():
            await manager.start()
            await manager.subscribe_bins([
                {"token_id": "tok_a", "bin_index": 0},
                {"token_id": "tok_b", "bin_index": 1},
            ])
            await manager.stop()

        asyncio.run(_run())

        assert manager._on_ws_update not in shared_ws._update_callbacks
        assert not shared_ws._ws._closed
        assert "tok_a" not in shared_ws._subscribed_tokens
        assert "tok_b" not in shared_ws._subscribed_tokens

    def test_external_ws_stop_while_disconnected_clears_global_subscription_state(self):
        shared_ws = _make_ws()
        shared_ws._connected = False
        shared_ws._ws = None
        shared_ws._subscribed_tokens = {"tok_a"}
        shared_ws._pending_subscriptions = {"tok_a"}
        shared_ws._token_to_bin = {"tok_a": 0}

        manager = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager._token_to_bin = {"tok_a": 0}

        async def _run():
            await manager.start()
            await manager.stop()

        asyncio.run(_run())

        assert manager._on_ws_update not in shared_ws._update_callbacks
        assert "tok_a" not in shared_ws._subscribed_tokens
        assert "tok_a" not in shared_ws._pending_subscriptions
        assert "tok_a" not in shared_ws._token_to_bin


class TestOwnedWSStopDisconnects:
    def test_owned_ws_stop_disconnects(self):
        """When manager owns the WS, stop() should disconnect."""
        config = WebSocketConfig(enabled=True)
        manager = OrderbookManager(config=config)

        dummy = DummyWS()
        ws = _make_ws()
        ws._ws = dummy
        ws._connected = True
        ws._running = True
        manager._ws_client = ws
        manager._owns_ws = True

        asyncio.run(manager.stop())

        assert ws._running is False


class TestTokenFilterInCallback:
    def test_only_own_tokens_processed(self):
        """Two managers sharing one WS; each only processes its own tokens."""
        shared_ws = _make_ws()
        shared_ws._token_to_bin["tok_a"] = 0
        shared_ws._token_to_bin["tok_b"] = 1

        results_a = []
        results_b = []

        manager_a = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager_a._token_to_bin = {"tok_a": 0}
        manager_a.on_significant_update = lambda tid, bi, ob: results_a.append(tid)

        manager_b = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager_b._token_to_bin = {"tok_b": 1}
        manager_b.on_significant_update = lambda tid, bi, ob: results_b.append(tid)

        shared_ws.register_callback(manager_a._on_ws_update)
        shared_ws.register_callback(manager_b._on_ws_update)

        shared_ws._handle_book_update(_make_book_data("tok_a"))
        assert results_a == ["tok_a"]
        assert results_b == []

        shared_ws._handle_book_update(_make_book_data("tok_b"))
        assert results_a == ["tok_a"]
        assert results_b == ["tok_b"]


class TestSignificantChangeDetection:
    def test_same_best_prices_not_significant(self):
        """Two updates with same best prices should fire callback only once."""
        shared_ws = _make_ws()
        shared_ws._token_to_bin["tok_abc"] = 0

        results = []
        manager = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager._token_to_bin = {"tok_abc": 0}
        manager.on_significant_update = lambda tid, bi, ob: results.append(tid)
        shared_ws.register_callback(manager._on_ws_update)

        shared_ws._handle_book_update(_make_book_data())
        assert len(results) == 1

        shared_ws._handle_book_update(_make_book_data())
        assert len(results) == 1


# ===========================================================================
# Integration-style test
# ===========================================================================


class TestSharedWSMultipleManagersLifecycle:
    def test_lifecycle(self):
        """
        Create shared WS, create 2 managers, subscribe, stop one,
        verify other still receives updates, stop second, verify WS alive.
        """
        shared_ws = _make_ws()
        shared_ws._ws = DummyWS()
        shared_ws._connected = True
        shared_ws._token_to_bin["tok_a"] = 0
        shared_ws._token_to_bin["tok_b"] = 1

        results_a = []
        results_b = []

        manager_a = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager_a.on_significant_update = lambda tid, bi, ob: results_a.append(tid)

        manager_b = OrderbookManager(
            config=WebSocketConfig(enabled=True),
            ws_client=shared_ws,
        )
        manager_b.on_significant_update = lambda tid, bi, ob: results_b.append(tid)

        async def _run():
            await manager_a.start()
            await manager_b.start()

            # Set up subscription state directly to avoid triggering the
            # single-subscribe-per-connection reconnect logic (tested elsewhere).
            manager_a._token_to_bin = {"tok_a": 0}
            manager_b._token_to_bin = {"tok_b": 1}
            shared_ws._subscribed_tokens = {"tok_a", "tok_b"}

            # Both receive their updates
            shared_ws._handle_book_update(_make_book_data("tok_a"))
            shared_ws._handle_book_update(_make_book_data("tok_b"))
            assert results_a == ["tok_a"]
            assert results_b == ["tok_b"]

            # Stop manager_a
            await manager_a.stop()

            # Manager_b still works
            results_b.clear()
            shared_ws._handle_book_update({
                "event_type": "book",
                "asset_id": "tok_b",
                "bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.65", "size": "200"}],
            })
            assert len(results_b) == 1  # Price changed => significant

            # tok_a updates no longer reach manager_a
            results_a.clear()
            shared_ws._handle_book_update(_make_book_data("tok_a"))
            assert results_a == []

            # Stop manager_b
            await manager_b.stop()

            # Shared WS still connected
            assert not shared_ws._ws._closed

        asyncio.run(_run())
