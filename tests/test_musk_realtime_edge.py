import asyncio
from datetime import date, timedelta
from types import SimpleNamespace

from src.algo.musk_tweet_count.forecaster.config import ForecasterConfig
from src.algo.musk_tweet_count.forecaster.data import ContractDayUtils, EventStore, TweetEvent
from src.algo.musk_tweet_count.forecaster.trading_bot import (
    GASKellyTradingBot,
    TradingBotConfig,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import TickResult


class DummyXTrackerClient:
    def fetch_historical(self, n_days, contract_utils, handle="elonmusk"):
        return {}


class DummyKellyBot:
    def __init__(self):
        self.orderbook_manager = None
        self.portfolio = None

    async def run_tick(self, **kwargs):
        return TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
            executions=[],
            elapsed_seconds=0.0,
        )


def _make_store() -> tuple[ContractDayUtils, EventStore]:
    contract_utils = ContractDayUtils()
    event_store = EventStore(contract_utils, DummyXTrackerClient())
    return contract_utils, event_store


def _make_event(
    contract_utils: ContractDayUtils,
    contract_day: date,
    hours_after_noon: int,
    event_id: str,
    *,
    source: str,
) -> TweetEvent:
    start_dt, _ = contract_utils.get_contract_day_bounds(contract_day)
    return TweetEvent(
        timestamp=start_dt + timedelta(hours=hours_after_noon),
        event_type="tweet",
        event_id=event_id,
        source=source,
    )


def _make_bot() -> GASKellyTradingBot:
    _, event_store = _make_store()
    return GASKellyTradingBot(
        clob_client=SimpleNamespace(),
        kelly_config=KellyConfig(),
        forecaster_config=ForecasterConfig(),
        bot_config=TradingBotConfig(sync_driven=True, dry_run=True),
        event_store=event_store,
    )


def test_event_store_merged_view_prefers_official_ids():
    contract_utils, event_store = _make_store()
    contract_day = date(2026, 1, 10)

    official = _make_event(contract_utils, contract_day, 1, "tweet-1", source="xtracker")
    provisional_dup = _make_event(contract_utils, contract_day, 2, "tweet-1", source="twikit")
    provisional_new = _make_event(contract_utils, contract_day, 3, "tweet-2", source="twikit")

    assert event_store.add_event(official) is True
    assert event_store.add_provisional_event(provisional_dup) is False
    assert event_store.add_provisional_event(provisional_new) is True

    merged = event_store.get_contract_day_events(contract_day)
    assert [event.event_id for event in merged] == ["tweet-1", "tweet-2"]
    assert [event.source for event in merged] == ["xtracker", "twikit"]
    assert event_store.get_contract_day_count(contract_day) == 2


def test_authoritative_replace_plus_overlay_clear_rebases_effective_count():
    contract_utils, event_store = _make_store()
    contract_day = date(2026, 1, 11)

    official = _make_event(contract_utils, contract_day, 1, "official-1", source="xtracker")
    provisional = _make_event(contract_utils, contract_day, 2, "prov-2", source="twikit")
    replacement = _make_event(contract_utils, contract_day, 1, "official-1", source="xtracker")

    assert event_store.add_event(official) is True
    assert event_store.add_provisional_event(provisional) is True
    assert event_store.get_contract_day_count(contract_day) == 2

    assert event_store.replace_official_day(contract_day, [replacement]) is False
    assert event_store.clear_provisional_day(contract_day) is True
    assert event_store.get_contract_day_count(contract_day) == 1
    assert [event.event_id for event in event_store.get_contract_day_events(contract_day)] == [
        "official-1"
    ]


def test_sync_tick_requests_coalesce_to_latest_version_and_highest_priority(monkeypatch):
    bot = _make_bot()
    calls = []

    async def fake_execute_tick(*, log_header, recompute, allow_authoritative_rebase=False):
        calls.append(
            {
                "log_header": log_header,
                "recompute": recompute,
                "allow_authoritative_rebase": allow_authoritative_rebase,
            }
        )
        return TickResult(
            num_candidates=0,
            num_executed=0,
            total_utility_gain=0.0,
            executions=[],
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(bot, "_execute_tick", fake_execute_tick)

    bot.request_sync_tick("realtime", 10)
    bot.request_sync_tick("realtime", 11)
    bot.request_sync_tick("xtracker", 10)

    request = bot._pop_pending_sync_tick()
    assert request is not None
    assert request.data_version == 11
    assert request.source == "xtracker"

    asyncio.run(bot._execute_sync_tick_request(request))

    assert calls == [
        {
            "log_header": True,
            "recompute": True,
            "allow_authoritative_rebase": False,
        }
    ]
    assert bot._last_processed_data_version == 11
    assert bot._last_processed_tick_source == "xtracker"


def test_authoritative_rebase_resets_regression_baseline():
    bot = _make_bot()
    bot.kelly_bot = DummyKellyBot()
    bot._cached_forecast_mean = 12.0
    bot._cached_forecast_std = 3.0
    bot._data_freshness_checker = lambda: (True, 0.0)
    bot._last_known_count = 9
    bot._get_timing = lambda: (24.0, 12.0)
    bot._get_market_cumulative_count = lambda: 7

    result = asyncio.run(bot._run_tick_impl(log_header=False, allow_authoritative_rebase=True))

    assert result is not None
    assert bot._last_known_count == 7
