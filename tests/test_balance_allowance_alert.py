from src.algo.musk_tweet_count.kelly.candidates import TradeAction, TradeCandidate
from src.algo.musk_tweet_count.kelly.config import KellyConfig
from src.algo.musk_tweet_count.kelly.executor import KellyExecutor
from src.algo.musk_tweet_count.kelly.portfolio import BinPosition, Portfolio


def test_balance_allowance_error_callback_includes_local_and_clob_context():
    portfolio = Portfolio(initial_capital=1000.0, capital=275.0)
    portfolio.positions[2] = BinPosition(
        bin_index=2,
        yes_token_id="yes-token",
        no_token_id="no-token",
        yes_shares=0.0,
        no_shares=18.0,
        yes_avg_cost=0.0,
        no_avg_cost=0.41,
        collateral_used=7.38,
    )

    candidate = TradeCandidate(
        bin_index=2,
        action=TradeAction.SELL_NO,
        size=25.0,
        price=0.44,
        utility_gain=0.0123,
        reservation_price=0.39,
        edge=0.1282,
        limit_price=0.441,
        execution_bound_price=0.441,
    )

    captured = []
    executor = KellyExecutor(
        config=KellyConfig(),
        portfolio=portfolio,
        token_ids={2: "yes-token"},
        no_token_ids={2: "no-token"},
        event_name="Test Event",
        on_balance_allowance_error=captured.append,
    )
    executor.set_log_context(bin_ranges={2: "20-29"})
    executor._pending_orders["abc"] = (candidate, "no-token")

    executor._emit_balance_allowance_error(
        candidate=candidate,
        token_id="no-token",
        error_msg="not enough balance / allowance",
        diagnostics={
            "clob_available_shares": 13.0,
            "nonzero_allowances": 1,
            "raw_balance": "13000000",
            "allowances": {"0xexchange": "1"},
        },
    )

    assert len(captured) == 1
    context = captured[0]
    assert context.event_name == "Test Event"
    assert context.action == "SELL_NO"
    assert context.side == "SELL"
    assert context.token_type == "NO"
    assert context.bin_range == "20-29"
    assert context.requested_size == 25.0
    assert context.requested_limit_price == 0.441
    assert context.portfolio_available_capital == 275.0
    assert context.portfolio_total_collateral == 7.38
    assert context.pending_orders_count == 1
    assert context.local_no_shares == 18.0
    assert context.local_no_avg_cost == 0.41
    assert context.clob_available_shares == 13.0
    assert context.allowances == {"0xexchange": "1"}
