import asyncio
import json

from websockets.protocol import State

from src.algo.musk_tweet_count.kelly.user_stream import APP_PONG_MESSAGE, UserStreamClient
from src.algo.musk_tweet_count.kelly.websocket_client import OrderbookWebSocket, WebSocketConfig


class DummyConnection:
    def __init__(self, state: State = State.OPEN):
        self.state = state
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_user_stream_connection_status_uses_websocket_state():
    client = UserStreamClient("key", "secret", "pass")
    client._ws = DummyConnection(State.OPEN)

    assert client.get_connection_status()["connected"] is True

    client._ws = DummyConnection(State.CLOSED)
    assert client.get_connection_status()["connected"] is False


def test_user_stream_set_markets_resends_sorted_subscription():
    client = UserStreamClient("key", "secret", "pass")
    client._ws = DummyConnection(State.OPEN)

    asyncio.run(client.set_markets(["market-b", "market-a", "market-a"]))

    assert len(client._ws.sent) == 1
    assert json.loads(client._ws.sent[0]) == {
        "type": "user",
        "auth": {
            "apiKey": "key",
            "secret": "secret",
            "passphrase": "pass",
        },
        "markets": ["market-a", "market-b"],
    }


def test_user_stream_handles_plain_pong_message():
    client = UserStreamClient("key", "secret", "pass")

    asyncio.run(client._handle_message(APP_PONG_MESSAGE))

    assert client._message_count == 1
    assert client._last_message_at is not None


def test_orderbook_socket_connection_state_uses_websocket_state():
    ws_client = OrderbookWebSocket(WebSocketConfig())
    ws_client._connected = True
    ws_client._ws = DummyConnection(State.OPEN)

    assert ws_client.is_connected is True

    ws_client._ws = DummyConnection(State.CLOSED)
    assert ws_client.is_connected is False


def test_orderbook_subscribe_sends_expected_payload():
    ws_client = OrderbookWebSocket(WebSocketConfig())
    ws_client._connected = True
    ws_client._ws = DummyConnection(State.OPEN)

    asyncio.run(ws_client._subscribe(["token-1", "token-2"]))

    assert len(ws_client._ws.sent) == 1
    assert json.loads(ws_client._ws.sent[0]) == {
        "type": "MARKET",
        "assets_ids": ["token-1", "token-2"],
        "auth": {},
    }
