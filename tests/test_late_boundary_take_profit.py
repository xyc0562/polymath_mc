import pytest

from src.algo.musk_tweet_count.kelly.candidates import TradeAction, generate_candidates
from src.algo.musk_tweet_count.kelly.config import KellyConfig, LateBoundaryTakeProfitConfig
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio
from src.algo.musk_tweet_count.kelly.take_profit import (
    build_late_boundary_take_profit_context,
    compute_late_boundary_take_profit_threshold,
)


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


def _make_portfolio(probabilities: list[float]) -> Portfolio:
    return Portfolio(
        initial_capital=1_000.0,
        capital=1_000.0,
        num_bins=len(probabilities),
        probabilities=probabilities,
        bin_upper_bounds=[259, 279, 299][: len(probabilities)],
    )


def _default_config() -> LateBoundaryTakeProfitConfig:
    return LateBoundaryTakeProfitConfig(enabled=True)


def test_take_profit_threshold_schedule_matches_defaults():
    config = _default_config()

    assert compute_late_boundary_take_profit_threshold(config, 6.0) == pytest.approx(180.0)
    assert compute_late_boundary_take_profit_threshold(config, 5.0) == pytest.approx(150.0)
    assert compute_late_boundary_take_profit_threshold(config, 4.0) == pytest.approx(120.0)
    assert compute_late_boundary_take_profit_threshold(config, 3.0) == pytest.approx(90.0)
    assert compute_late_boundary_take_profit_threshold(config, 2.0) == pytest.approx(90.0)


def test_take_profit_uses_majority_vwap_not_best_bid():
    portfolio = _make_portfolio([0.1, 0.8, 0.1])
    portfolio.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    config = _default_config()
    context = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=130.0,
        portfolio=portfolio,
        orderbooks={
            0: _make_orderbook(
                yes_bids=[(0.85, 10.0), (0.78, 100.0)],
                yes_asks=[(0.86, 100.0)],
            )
        },
    )

    assert context.active is False
    assert context.skipped_reason == "trigger_price_not_met"
    assert context.reference_vwap < 0.80


def test_take_profit_requires_majority_depth():
    portfolio = _make_portfolio([0.1, 0.8, 0.1])
    portfolio.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    config = _default_config()
    context = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=130.0,
        portfolio=portfolio,
        orderbooks={
            0: _make_orderbook(
                yes_bids=[(0.82, 60.0)],
                yes_asks=[(0.83, 100.0)],
            )
        },
    )

    assert context.active is False
    assert context.skipped_reason == "insufficient_majority_depth"


def test_take_profit_context_activates_with_expected_band():
    portfolio = _make_portfolio([0.1, 0.8, 0.1])
    portfolio.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    config = _default_config()
    context = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=3.0,
        silence_minutes=150.0,
        portfolio=portfolio,
        orderbooks={
            0: _make_orderbook(
                yes_bids=[(0.86, 120.0)],
                yes_asks=[(0.87, 120.0)],
            )
        },
    )

    assert context.active is True
    assert context.current_bin_index == 0
    assert context.distance_to_next_bin == 1
    assert context.reference_size == pytest.approx(70.0)
    assert context.reference_vwap == pytest.approx(0.86)
    assert context.size_floor == pytest.approx(70.0)
    assert 70.0 <= context.size_cap <= 90.0


def test_take_profit_context_is_restart_stable_and_turns_off_when_silence_breaks():
    portfolio_a = _make_portfolio([0.1, 0.8, 0.1])
    portfolio_a.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    portfolio_b = _make_portfolio([0.1, 0.8, 0.1])
    portfolio_b.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    config = _default_config()
    orderbooks = {0: _make_orderbook(yes_bids=[(0.84, 120.0)], yes_asks=[(0.85, 120.0)])}

    context_a = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=130.0,
        portfolio=portfolio_a,
        orderbooks=orderbooks,
    )
    context_b = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=130.0,
        portfolio=portfolio_b,
        orderbooks=orderbooks,
    )
    context_broken = build_late_boundary_take_profit_context(
        config=config,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=30.0,
        portfolio=portfolio_b,
        orderbooks=orderbooks,
    )

    assert context_a == context_b
    assert context_a.active is True
    assert context_broken.active is False
    assert context_broken.skipped_reason == "insufficient_silence"


def test_generate_candidates_blocks_current_bin_buy_yes_and_adds_take_profit_sell():
    portfolio = _make_portfolio([0.1, 0.8, 0.1])
    portfolio.execute_buy_yes(0, 100.0, 0.20, "yes-0")
    config = KellyConfig(
        late_boundary_take_profit=LateBoundaryTakeProfitConfig(enabled=True),
    )
    orderbooks = {
        0: _make_orderbook(yes_bids=[(0.84, 120.0)], yes_asks=[(0.85, 120.0)]),
        1: _make_orderbook(yes_bids=[(0.10, 120.0)], yes_asks=[(0.12, 120.0)]),
        2: _make_orderbook(yes_bids=[(0.02, 120.0)], yes_asks=[(0.03, 120.0)]),
    }
    context = build_late_boundary_take_profit_context(
        config=config.late_boundary_take_profit,
        current_count=259,
        bin_ranges=[(240, 259), (260, 279), (280, 299)],
        hours_remaining=4.0,
        silence_minutes=130.0,
        portfolio=portfolio,
        orderbooks=orderbooks,
    )

    candidates = generate_candidates(
        portfolio=portfolio,
        orderbooks=orderbooks,
        config=config,
        hours_to_settlement=4.0,
        take_profit_context=context,
        verbose=False,
    )

    assert all(
        not (c.bin_index == 0 and c.action == TradeAction.BUY_YES)
        for c in candidates
    )
    tp = next(c for c in candidates if c.kind == "late_boundary_take_profit")
    assert tp.bin_index == 0
    assert tp.action == TradeAction.SELL_YES
    assert tp.size_floor >= 70.0
    assert tp.size_cap >= tp.size_floor
    assert tp.min_utility_override == 0.0
