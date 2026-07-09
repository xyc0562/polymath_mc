import time

import pytest

from src.algo.musk_tweet_count.kelly import executor as executor_module
from src.algo.musk_tweet_count.kelly.executor import (
    ORDER_MANAGER_BREAKER,
    OrderExecutor,
    OrderManagerBreaker,
)


class Fake425Error(Exception):
    status_code = 425


class Fake503Error(Exception):
    status_code = 503


class FakeClient:
    """CLOB client stub; post_errors is a per-call sequence (None = success)."""

    def __init__(self, post_errors=None):
        self.post_errors = list(post_errors or [])
        self.post_calls = 0

    def get_tick_size(self, token_id):
        return "0.01"

    def create_order(self, order_args):
        return {"signed": True}

    def _next(self):
        self.post_calls += 1
        if self.post_errors:
            err = self.post_errors.pop(0)
            if err is not None:
                raise err

    def post_orders(self, signed_args):
        self._next()
        return [
            {"orderID": f"order-{self.post_calls}-{i}", "success": True}
            for i in range(len(signed_args))
        ]

    def post_order(self, signed_order, order_type=None):
        self._next()
        return {"orderID": f"order-{self.post_calls}"}


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch):
    ORDER_MANAGER_BREAKER.reset()
    monkeypatch.setattr(executor_module.time, "sleep", lambda s: None)
    yield
    ORDER_MANAGER_BREAKER.reset()


def _buy_order(size=50.0, price=0.5):
    return {
        "token_id": "tok-yes-0",
        "side": "BUY",
        "price": price,
        "size": size,
        "fak_resolved": True,
    }


def _make_executor(client) -> OrderExecutor:
    return OrderExecutor(clob_client=client, dry_run=False, event_name="test-event")


# ---------- breaker unit behavior ----------


def test_backoff_doubles_and_caps():
    breaker = OrderManagerBreaker()
    expected = [10.0, 20.0, 40.0, 80.0, 120.0, 120.0]
    for exp in expected:
        breaker.record_failure()
        assert breaker._backoff == exp
        assert breaker.is_open()
    breaker.record_success()
    assert not breaker.is_open()
    assert breaker._consecutive_failures == 0
    assert breaker._backoff == 0.0


def test_is_not_ready_error_detection():
    assert OrderManagerBreaker.is_not_ready_error(Fake425Error("boom")) is True
    assert OrderManagerBreaker.is_not_ready_error(
        Exception("PolyApiException[status_code=425, error_message=...]")
    ) is True
    assert OrderManagerBreaker.is_not_ready_error(Fake503Error("unavailable")) is False
    assert OrderManagerBreaker.is_not_ready_error(Exception("timeout")) is False


# ---------- batch path ----------


def test_batch_425_then_success_retries_once():
    client = FakeClient(post_errors=[Fake425Error("not ready"), None])
    executor = _make_executor(client)

    results = executor.place_batch_orders([_buy_order()])

    assert client.post_calls == 2
    assert results[0]["orderID"] is not None
    assert not ORDER_MANAGER_BREAKER.is_open()


def test_batch_double_425_opens_breaker_and_short_circuits():
    client = FakeClient(post_errors=[Fake425Error("not ready"), Fake425Error("still not ready")])
    executor = _make_executor(client)

    results = executor.place_batch_orders([_buy_order()])

    assert client.post_calls == 2
    assert results[0]["orderID"] is None
    assert ORDER_MANAGER_BREAKER.is_open()

    # While open: submissions never reach the client
    results2 = executor.place_batch_orders([_buy_order(), _buy_order()])
    assert client.post_calls == 2
    assert all(r["orderID"] is None for r in results2)
    assert all("not ready" in r["errorMsg"] for r in results2)
    assert executor._last_error == "order manager not ready (breaker open)"


def test_batch_non_425_error_single_attempt_no_breaker():
    client = FakeClient(post_errors=[Fake503Error("unavailable")])
    executor = _make_executor(client)

    results = executor.place_batch_orders([_buy_order()])

    assert client.post_calls == 1  # no retry for non-425
    assert results[0]["orderID"] is None
    assert not ORDER_MANAGER_BREAKER.is_open()
    assert "unavailable" in executor._last_error


def test_probe_after_expiry_closes_breaker():
    client = FakeClient(post_errors=[Fake425Error("a"), Fake425Error("b")])
    executor = _make_executor(client)
    executor.place_batch_orders([_buy_order()])
    assert ORDER_MANAGER_BREAKER.is_open()

    # Simulate backoff expiry
    ORDER_MANAGER_BREAKER._not_ready_until = time.monotonic() - 1

    results = executor.place_batch_orders([_buy_order()])

    assert results[0]["orderID"] is not None
    assert not ORDER_MANAGER_BREAKER.is_open()
    assert ORDER_MANAGER_BREAKER._consecutive_failures == 0
    assert ORDER_MANAGER_BREAKER._backoff == 0.0


def test_repeated_probe_failures_double_backoff():
    client = FakeClient(post_errors=[Fake425Error("a"), Fake425Error("b"),
                                     Fake425Error("c"), Fake425Error("d")])
    executor = _make_executor(client)
    executor.place_batch_orders([_buy_order()])
    first_backoff = ORDER_MANAGER_BREAKER._backoff

    ORDER_MANAGER_BREAKER._not_ready_until = time.monotonic() - 1
    executor.place_batch_orders([_buy_order()])

    assert ORDER_MANAGER_BREAKER._backoff == first_backoff * 2
    assert ORDER_MANAGER_BREAKER.is_open()


def test_dry_run_unaffected_by_open_breaker():
    client = FakeClient()
    executor = OrderExecutor(clob_client=client, dry_run=True, event_name="test-event")
    ORDER_MANAGER_BREAKER.record_failure()
    assert ORDER_MANAGER_BREAKER.is_open()

    results = executor.place_batch_orders([_buy_order()])

    assert results[0]["orderID"] == "dry_run_0"


# ---------- single-order path ----------


def test_limit_order_double_425_opens_breaker_and_short_circuits():
    client = FakeClient(post_errors=[Fake425Error("a"), Fake425Error("b")])
    executor = _make_executor(client)

    response = executor.place_limit_order("tok-yes-0", "BUY", 0.5, 50.0)

    assert response is None
    assert client.post_calls == 2
    assert ORDER_MANAGER_BREAKER.is_open()

    response2 = executor.place_limit_order("tok-yes-0", "BUY", 0.5, 50.0)
    assert response2 is None
    assert client.post_calls == 2  # short-circuited before the client


def test_limit_order_425_then_success():
    client = FakeClient(post_errors=[Fake425Error("not ready"), None])
    executor = _make_executor(client)

    response = executor.place_limit_order("tok-yes-0", "BUY", 0.5, 50.0)

    assert response is not None
    assert response["orderID"] == "order-2"
    assert not ORDER_MANAGER_BREAKER.is_open()


def test_breaker_status_shape():
    ORDER_MANAGER_BREAKER.record_failure()
    status = ORDER_MANAGER_BREAKER.status()

    assert status["open"] is True
    assert status["consecutive_failures"] == 1
    assert status["backoff_seconds"] == 10.0
    assert status["seconds_remaining"] > 0
