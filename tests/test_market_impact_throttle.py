import time

from src.algo.musk_tweet_count.kelly.candidates import (
    TradeAction,
    TradeCandidate,
    _generate_buy_no_candidate,
    _generate_buy_yes_candidate,
    _generate_sell_no_candidate,
    _generate_sell_yes_candidate,
)
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import KellyExecutor
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio


def _make_orderbook(
    *,
    bin_index: int = 0,
    yes_bids: list[tuple[float, float]] | None = None,
    yes_asks: list[tuple[float, float]] | None = None,
) -> UnifiedOrderbook:
    return UnifiedOrderbook(
        bin_index=bin_index,
        yes_token_id=f"yes-{bin_index}",
        yes_bids=[OrderbookLevel(price=price, size=size) for price, size in (yes_bids or [])],
        yes_asks=[OrderbookLevel(price=price, size=size) for price, size in (yes_asks or [])],
    )


def _make_portfolio(
    *,
    capital: float = 100.0,
    probabilities: list[float] | None = None,
) -> Portfolio:
    probabilities = probabilities or [1.0]
    return Portfolio(
        initial_capital=capital,
        capital=capital,
        num_bins=len(probabilities),
        probabilities=probabilities,
    )


def _make_executor(*, probabilities: list[float] | None = None) -> KellyExecutor:
    portfolio = _make_portfolio(probabilities=probabilities or [1.0])
    return KellyExecutor(
        config=KellyConfig(),
        portfolio=portfolio,
        token_ids={0: "yes-0"},
        no_token_ids={0: "no-0"},
        event_name="impact-test",
    )


def _make_trade(
    *,
    action: TradeAction,
    size: float,
    price: float,
    limit_price: float,
    threshold_price: float,
) -> TradeCandidate:
    return TradeCandidate(
        bin_index=0,
        action=action,
        size=size,
        price=price,
        utility_gain=0.02,
        reservation_price=0.25,
        edge=0.05,
        limit_price=limit_price,
        threshold_price=threshold_price,
    )


def test_kelly_config_from_dict_parses_market_impact():
    config = KellyConfig.from_dict(
        {
            "market_impact": {
                "fresh_start_enabled": False,
                "fresh_start_minutes": 7.5,
                "fresh_start_edge_fraction": 0.1,
            }
        }
    )

    assert config.market_impact.fresh_start_enabled is False
    assert config.market_impact.fresh_start_minutes == 7.5
    assert config.market_impact.fresh_start_edge_fraction == 0.1


def test_candidate_threshold_prices_cover_buy_and_sell_paths():
    config = KellyConfig()

    buy_portfolio = _make_portfolio(probabilities=[0.30, 0.70])
    buy_yes_orderbook = _make_orderbook(
        yes_bids=[(0.19, 100)],
        yes_asks=[(0.20, 100)],
    )
    buy_yes = _generate_buy_yes_candidate(
        bin_index=0,
        orderbook=buy_yes_orderbook,
        portfolio=buy_portfolio,
        reservation_price=0.30,
        config=config,
        hours_to_settlement=24.0,
    )
    assert buy_yes is not None
    assert round(buy_yes.threshold_price, 4) == 0.28

    buy_no_orderbook = _make_orderbook(
        yes_bids=[(0.40, 100)],
        yes_asks=[(0.41, 100)],
    )
    buy_no = _generate_buy_no_candidate(
        bin_index=0,
        orderbook=buy_no_orderbook,
        portfolio=buy_portfolio,
        reservation_price=0.70,
        config=config,
        hours_to_settlement=24.0,
    )
    assert buy_no is not None
    assert round(buy_no.threshold_price, 4) == 0.68

    sell_yes_portfolio = _make_portfolio(probabilities=[0.32, 0.68])
    sell_yes_portfolio.execute_buy_yes(0, 20.0, 0.20, "yes-0")
    sell_yes_orderbook = _make_orderbook(
        yes_bids=[(0.35, 100)],
        yes_asks=[(0.36, 100)],
    )
    sell_yes = _generate_sell_yes_candidate(
        bin_index=0,
        orderbook=sell_yes_orderbook,
        portfolio=sell_yes_portfolio,
        model_probability=0.32,
        reservation_price=0.30,
        config=config,
        hours_to_settlement=24.0,
    )
    assert sell_yes is not None
    assert round(sell_yes.threshold_price, 4) == 0.32

    sell_no_portfolio = _make_portfolio(probabilities=[0.25, 0.75])
    sell_no_portfolio.execute_buy_no(0, 20.0, 0.60, "yes-0")
    sell_no_orderbook = _make_orderbook(
        yes_bids=[(0.24, 100)],
        yes_asks=[(0.25, 100)],
    )
    sell_no = _generate_sell_no_candidate(
        bin_index=0,
        orderbook=sell_no_orderbook,
        portfolio=sell_no_portfolio,
        model_probability=0.75,
        reservation_price=0.70,
        config=config,
        hours_to_settlement=24.0,
    )
    assert sell_no is not None
    assert round(sell_no.threshold_price, 4) == 0.72


def test_fresh_start_buy_throttle_blocks_large_low_price_walk():
    executor = _make_executor()
    executor.set_log_context(fresh_start_started_at=time.time())
    orderbook = _make_orderbook(
        yes_asks=[(0.01, 100), (0.02, 100)],
    )
    trade = _make_trade(
        action=TradeAction.BUY_YES,
        size=150.0,
        price=0.0133,
        limit_price=0.02,
        threshold_price=0.02,
    )

    throttled = executor._apply_fresh_start_market_impact([trade], {0: orderbook})

    assert len(throttled) == 1
    assert throttled[0].size == 100.0
    assert throttled[0].price == 0.01
    assert throttled[0].limit_price == 0.01


def test_fresh_start_buy_throttle_allows_modest_mid_price_walk():
    executor = _make_executor()
    executor.set_log_context(fresh_start_started_at=time.time())
    orderbook = _make_orderbook(
        yes_asks=[(0.50, 10), (0.51, 10), (0.52, 100)],
    )
    trade = _make_trade(
        action=TradeAction.BUY_YES,
        size=25.0,
        price=0.51,
        limit_price=0.52,
        threshold_price=0.54,
    )

    throttled = executor._apply_fresh_start_market_impact([trade], {0: orderbook})

    assert len(throttled) == 1
    assert throttled[0].size == 20.0
    assert round(throttled[0].price, 4) == 0.505
    assert throttled[0].limit_price == 0.51


def test_fresh_start_sell_throttle_uses_sell_side_threshold():
    executor = _make_executor()
    executor.set_log_context(fresh_start_started_at=time.time())
    orderbook = _make_orderbook(
        yes_bids=[(0.80, 10), (0.79, 10), (0.70, 100)],
    )
    trade = _make_trade(
        action=TradeAction.SELL_YES,
        size=25.0,
        price=0.77,
        limit_price=0.70,
        threshold_price=0.76,
    )

    throttled = executor._apply_fresh_start_market_impact([trade], {0: orderbook})

    assert len(throttled) == 1
    assert throttled[0].size == 20.0
    assert round(throttled[0].price, 4) == 0.795
    assert throttled[0].limit_price == 0.79


def test_fresh_start_throttle_active_within_twenty_minutes_only():
    orderbook = _make_orderbook(
        yes_asks=[(0.01, 100), (0.02, 100)],
    )
    trade = _make_trade(
        action=TradeAction.BUY_YES,
        size=150.0,
        price=0.0133,
        limit_price=0.02,
        threshold_price=0.02,
    )

    active_executor = _make_executor()
    active_executor.set_log_context(fresh_start_started_at=time.time() - (19 * 60))
    throttled = active_executor._apply_fresh_start_market_impact([trade], {0: orderbook})
    assert throttled[0].size == 100.0

    inactive_executor = _make_executor()
    inactive_executor.set_log_context(fresh_start_started_at=time.time() - (21 * 60))
    unthrottled = inactive_executor._apply_fresh_start_market_impact([trade], {0: orderbook})
    assert unthrottled[0].size == 150.0
