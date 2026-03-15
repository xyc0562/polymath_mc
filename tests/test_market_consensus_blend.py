import subprocess
import sys
from pathlib import Path

import pytest

from src.algo.musk_tweet_count.kelly.candidates import TradeAction, generate_candidates
from src.algo.musk_tweet_count.kelly.config import KellyConfig, MarketConsensusConfig
from src.algo.musk_tweet_count.kelly.market_signals import (
    compute_market_consensus_blend,
    compute_market_quote_context,
)
from src.algo.musk_tweet_count.kelly.orderbook import OrderbookLevel, UnifiedOrderbook
from src.algo.musk_tweet_count.kelly.portfolio import Portfolio


def _make_orderbook(
    *,
    bin_index: int,
    yes_bids: list[tuple[float, float]] | None = None,
    yes_asks: list[tuple[float, float]] | None = None,
):
    return UnifiedOrderbook(
        bin_index=bin_index,
        yes_token_id=f"yes-{bin_index}",
        yes_bids=[OrderbookLevel(price=price, size=size) for price, size in (yes_bids or [])],
        yes_asks=[OrderbookLevel(price=price, size=size) for price, size in (yes_asks or [])],
    )


def _make_portfolio(probabilities: list[float], capital: float = 500.0) -> Portfolio:
    return Portfolio(
        initial_capital=capital,
        capital=capital,
        num_bins=len(probabilities),
        probabilities=probabilities,
    )


def test_market_quote_context_filters_untrusted_bins_and_dead_bins():
    probabilities = [0.25, 0.25, 0.20, 0.30, 0.0]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.20, 100)], yes_asks=[(0.22, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.30, 100)]),
        2: _make_orderbook(bin_index=2, yes_bids=[(0.40, 100)], yes_asks=[(0.39, 100)]),
        3: _make_orderbook(bin_index=3, yes_bids=[(0.45, 100)], yes_asks=[(0.60, 100)]),
        4: _make_orderbook(bin_index=4, yes_bids=[(0.10, 100)], yes_asks=[(0.12, 100)]),
    }
    config = MarketConsensusConfig(enabled=True, max_bin_spread=0.10, min_coverage_ratio=0.8)

    context = compute_market_quote_context(probabilities, [4], orderbooks, config)

    assert context is not None
    assert context["trusted_bins"] == [0]
    assert context["coverage_ratio"] == pytest.approx(0.25)
    assert context["blend_allowed"] is False


def test_market_quote_context_builds_trusted_subdistribution():
    probabilities = [0.40, 0.35, 0.25]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.19, 100)], yes_asks=[(0.21, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.49, 100)], yes_asks=[(0.51, 100)]),
        2: _make_orderbook(bin_index=2, yes_bids=[(0.10, 100)]),
    }
    config = MarketConsensusConfig(enabled=True, min_coverage_ratio=0.5, max_bin_spread=0.10)

    context = compute_market_quote_context(probabilities, [], orderbooks, config)

    assert context is not None
    assert context["trusted_bins"] == [0, 1]
    assert context["blend_allowed"] is True
    assert context["trusted_market_subdist"][0] == pytest.approx(0.2 / 0.7)
    assert context["trusted_market_subdist"][1] == pytest.approx(0.5 / 0.7)
    assert context["trusted_model_mass"] == pytest.approx(0.75)


def test_consensus_blend_passthrough_when_disabled():
    probabilities = [0.4, 0.6]
    blended, context = compute_market_consensus_blend(
        probabilities=probabilities,
        dead_bins=[],
        orderbooks={0: _make_orderbook(bin_index=0, yes_bids=[(0.4, 100)], yes_asks=[(0.42, 100)])},
        consensus_config=MarketConsensusConfig(enabled=False),
        hours_remaining=1.0,
    )

    assert blended == probabilities
    assert context is None


def test_consensus_blend_passthrough_when_coverage_is_too_low():
    probabilities = [0.6, 0.4]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.20, 100)], yes_asks=[(0.22, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.60, 100)]),
    }
    config = MarketConsensusConfig(enabled=True, time_enabled=True, min_coverage_ratio=0.8)

    blended, context = compute_market_consensus_blend(
        probabilities=probabilities,
        dead_bins=[],
        orderbooks=orderbooks,
        consensus_config=config,
        hours_remaining=2.0,
    )

    assert blended == probabilities
    assert context is None


def test_consensus_time_alpha_decreases_toward_settlement():
    probabilities = [0.5, 0.5]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.75, 100)], yes_asks=[(0.77, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.23, 100)], yes_asks=[(0.25, 100)]),
    }
    config = MarketConsensusConfig(enabled=True, time_enabled=True, time_tau=12.0)

    _, far = compute_market_consensus_blend(probabilities, [], orderbooks, config, hours_remaining=48.0)
    _, near = compute_market_consensus_blend(probabilities, [], orderbooks, config, hours_remaining=3.0)

    assert far is not None and near is not None
    assert far["alpha"] > near["alpha"]


def test_consensus_gap_alpha_decreases_with_larger_disagreement():
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.75, 100)], yes_asks=[(0.77, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.23, 100)], yes_asks=[(0.25, 100)]),
    }
    config = MarketConsensusConfig(enabled=True, gap_enabled=True, gap_scale=0.5, gap_gamma=1.5)

    _, small_gap = compute_market_consensus_blend([0.74, 0.26], [], orderbooks, config, hours_remaining=24.0)
    _, large_gap = compute_market_consensus_blend([0.10, 0.90], [], orderbooks, config, hours_remaining=24.0)

    assert small_gap is not None and large_gap is not None
    assert small_gap["alpha"] > large_gap["alpha"]


def test_consensus_gap_alpha_honors_gap_floor():
    probabilities = [0.05, 0.95]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.90, 100)], yes_asks=[(0.92, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.08, 100)], yes_asks=[(0.10, 100)]),
    }
    config = MarketConsensusConfig(
        enabled=True,
        gap_enabled=True,
        gap_scale=0.2,
        gap_gamma=1.5,
        gap_floor=0.80,
        min_model_weight=0.30,
    )

    _, context = compute_market_consensus_blend(probabilities, [], orderbooks, config, hours_remaining=24.0)

    assert context is not None
    assert context["alpha_p"] == pytest.approx(0.80)
    assert context["alpha"] == pytest.approx(0.80)


def test_consensus_alpha_honors_global_min_model_weight():
    probabilities = [0.05, 0.95]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.90, 100)], yes_asks=[(0.92, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.08, 100)], yes_asks=[(0.10, 100)]),
    }
    config = MarketConsensusConfig(
        enabled=True,
        time_enabled=True,
        gap_enabled=True,
        time_tau=12.0,
        gap_scale=0.2,
        gap_gamma=1.5,
        min_model_weight=0.30,
    )

    _, context = compute_market_consensus_blend(probabilities, [], orderbooks, config, hours_remaining=0.0)

    assert context is not None
    assert context["alpha"] == pytest.approx(0.30)


def test_consensus_blend_only_moves_trusted_bins():
    probabilities = [0.20, 0.30, 0.50]
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.19, 100)], yes_asks=[(0.21, 100)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.39, 100)], yes_asks=[(0.41, 100)]),
        2: _make_orderbook(bin_index=2, yes_bids=[(0.10, 100)]),
    }
    config = MarketConsensusConfig(
        enabled=True,
        time_enabled=True,
        min_model_weight=0.30,
        min_coverage_ratio=0.5,
    )

    blended, context = compute_market_consensus_blend(
        probabilities=probabilities,
        dead_bins=[],
        orderbooks=orderbooks,
        consensus_config=config,
        hours_remaining=0.0,
    )

    assert context is not None
    assert blended[2] == pytest.approx(probabilities[2])
    assert sum(blended[:2]) == pytest.approx(sum(probabilities[:2]))
    assert blended[0] == pytest.approx(0.1766666667)
    assert blended[1] == pytest.approx(0.3233333333)


def test_generate_candidates_blocks_untrusted_buys_when_consensus_enabled():
    portfolio = _make_portfolio([0.90, 0.10])
    config = KellyConfig(
        market_consensus=MarketConsensusConfig(enabled=True, require_trusted_quote_for_buys=True)
    )
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.10, 200)], yes_asks=[(0.21, 200)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.40, 200)], yes_asks=[(0.42, 200)]),
    }

    candidates = generate_candidates(
        portfolio=portfolio,
        orderbooks=orderbooks,
        config=config,
        hours_to_settlement=24.0,
        verbose=True,
    )

    assert all(
        not (candidate.bin_index == 0 and candidate.action in (TradeAction.BUY_YES, TradeAction.BUY_NO))
        for candidate in candidates
    )


def test_generate_candidates_allows_untrusted_buy_when_consensus_disabled():
    portfolio = _make_portfolio([0.90, 0.10])
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.10, 200)], yes_asks=[(0.21, 200)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.40, 200)], yes_asks=[(0.42, 200)]),
    }

    candidates = generate_candidates(
        portfolio=portfolio,
        orderbooks=orderbooks,
        config=KellyConfig(),
        hours_to_settlement=24.0,
        verbose=True,
    )

    assert any(candidate.bin_index == 0 and candidate.action == TradeAction.BUY_YES for candidate in candidates)


def test_generate_candidates_allows_sell_from_untrusted_bin():
    portfolio = _make_portfolio([0.20, 0.80])
    portfolio.execute_buy_yes(0, 20.0, 0.10, "yes-0")
    config = KellyConfig(
        market_consensus=MarketConsensusConfig(enabled=True, require_trusted_quote_for_buys=True)
    )
    orderbooks = {
        0: _make_orderbook(bin_index=0, yes_bids=[(0.40, 200)], yes_asks=[(0.51, 200)]),
        1: _make_orderbook(bin_index=1, yes_bids=[(0.70, 200)], yes_asks=[(0.72, 200)]),
    }

    candidates = generate_candidates(
        portfolio=portfolio,
        orderbooks=orderbooks,
        config=config,
        hours_to_settlement=24.0,
        verbose=True,
    )

    assert any(candidate.bin_index == 0 and candidate.action == TradeAction.SELL_YES for candidate in candidates)


def test_kelly_config_from_dict_parses_market_consensus():
    config = KellyConfig.from_dict(
        {
            "market_consensus": {
                "enabled": True,
                "time_enabled": True,
                "gap_enabled": True,
                "time_tau": 8.0,
                "gap_scale": 0.35,
                "gap_gamma": 2.0,
                "gap_floor": 0.8,
                "min_model_weight": 0.3,
            }
        }
    )

    assert config.market_consensus.enabled is True
    assert config.market_consensus.time_enabled is True
    assert config.market_consensus.gap_enabled is True
    assert config.market_consensus.time_tau == 8.0
    assert config.market_consensus.gap_scale == 0.35
    assert config.market_consensus.gap_gamma == 2.0
    assert config.market_consensus.gap_floor == 0.8
    assert config.market_consensus.min_model_weight == 0.3


def test_kelly_config_from_dict_rejects_removed_market_aware():
    with pytest.raises(ValueError, match="market_aware"):
        KellyConfig.from_dict({"market_aware": {"enabled": True}})


def test_backtest_cli_rejects_removed_market_aware_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.algo.musk_tweet_count.backtest.run_backtest",
            "--market-aware",
            "--list",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--market-aware" in result.stderr
