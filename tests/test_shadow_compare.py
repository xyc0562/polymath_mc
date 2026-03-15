from pathlib import Path

from src.algo.musk_tweet_count.backtest.replay_seed import extract_seed_state_from_log
from src.algo.musk_tweet_count.backtest.shadow_compare import (
    extract_live_snapshot_from_log,
    extract_replay_sections,
)


def test_extract_live_snapshot_from_log_parses_structured_block(tmp_path: Path) -> None:
    log_path = tmp_path / "live.log"
    log_path.write_text(
        "\n".join(
            [
                "26-03-13 15:16:03[I]:   Event: Mar 06 - Mar 13  |  Count: 378  |  Time Left: 0.7h  |  Forecast @11:16:03",
                "26-03-13 15:16:03[I]: [Mar 06 - Mar 13] Event capital: budget=$1000.00, invested=$985.24, available=$14.76 (was $14.76)",
                "26-03-13 15:16:03[I]: Candidate rejection reasons:",
                "26-03-13 15:16:03[I]:   Bin 18: BUY_YES: have 357 NO shares (sell NO first); SELL_NO: sell friction (bid=20.6% < fair+friction=44.3%)",
                "26-03-13 15:16:03[I]: All BUY candidates (priority order):",
                "26-03-13 15:16:03[I]:   Bin 18 BUY_NO: util=0.002047 edge=+101.64% fair=0.423 thresh=0.403 vwap=0.210 size=9.5",
                "26-03-13 15:16:03[I]: [Mar 06 - Mar 13][SIM iter=0] Reject BUY_NO bin=18 (360-379): sized_utility_below_min",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = extract_live_snapshot_from_log(
        log_path=log_path,
        event_name="Mar 06 - Mar 13",
        snapshot_timestamp=1773414963,
    )

    assert snapshot.current_count == 378
    assert snapshot.time_left_hours == 0.7
    assert snapshot.forecast_wallclock == "11:16:03"
    assert snapshot.available_capital == 14.76
    assert snapshot.invested_capital == 985.24
    assert snapshot.candidate_reason_lines == [
        "Bin 18: BUY_YES: have 357 NO shares (sell NO first); SELL_NO: sell friction (bid=20.6% < fair+friction=44.3%)"
    ]
    assert snapshot.all_buy_candidate_lines == [
        "Bin 18 BUY_NO: util=0.002047 edge=+101.64% fair=0.423 thresh=0.403 vwap=0.210 size=9.5"
    ]
    assert snapshot.reject_lines == [
        "[Mar 06 - Mar 13][SIM iter=0] Reject BUY_NO bin=18 (360-379): sized_utility_below_min"
    ]


def test_extract_replay_sections_finds_reasons_candidates_rejects_and_trades() -> None:
    log_text = "\n".join(
        [
            "2026-03-15 10:00:00 [INFO] Candidate rejection reasons:",
            "2026-03-15 10:00:00 [INFO]   Bin 19: SELL_YES: sell friction (bid=20.3% < fair+friction=44.3%)",
            "2026-03-15 10:00:00 [INFO] All BUY candidates (priority order):",
            "2026-03-15 10:00:00 [INFO]   Bin 18 BUY_NO: util=0.002047 edge=+101.64% fair=0.423 thresh=0.403 vwap=0.210 size=9.5",
            "2026-03-15 10:00:00 [INFO] [unknown][SIM iter=0] Reject BUY_NO bin=18 (360-379): sized_utility_below_min",
            "2026-03-15 10:00:00 [INFO]    41 SELL YES  18 360-379      2407.0  0.650 $1565.75  65.0% 67.5%  18.1%   1.9x          -",
        ]
    )

    reasons, candidates, rejects, trades = extract_replay_sections(log_text)

    assert reasons == [
        "Bin 19: SELL_YES: sell friction (bid=20.3% < fair+friction=44.3%)"
    ]
    assert candidates == [
        "Bin 18 BUY_NO: util=0.002047 edge=+101.64% fair=0.423 thresh=0.403 vwap=0.210 size=9.5"
    ]
    assert rejects == [
        "[unknown][SIM iter=0] Reject BUY_NO bin=18 (360-379): sized_utility_below_min"
    ]
    assert trades == [
        "41 SELL YES  18 360-379      2407.0  0.650 $1565.75  65.0% 67.5%  18.1%   1.9x          -"
    ]


def test_extract_seed_state_from_log_normalizes_zero_padded_event_names(tmp_path: Path) -> None:
    log_path = tmp_path / "seed.log"
    log_path.write_text(
        "\n".join(
            [
                "26-03-13 15:16:03[I]: [Mar 06 - Mar 13] Event capital: budget=$1000.00, invested=$985.24, available=$14.76 (was $14.76)",
                "26-03-13 15:16:03[I]: [Mar 06 - Mar 13][KELLY] iter=1 Synced from API: base_capital=$14.76, base_invested=$985.24, effective_capital=$14.76, overlay_entries=0",
                "26-03-13 15:16:03[I]: Multi-bin Kelly state:",
                "26-03-13 15:16:03[I]:   Capital: $14.76, Invested: $985.24",
                "26-03-13 15:16:03[I]:   Bin 18: YES=0.0 NO=357.3 cost=$224.10 | W[18]=$649.28 | c*_YES=0.577 c*_NO=0.423 | model_p=0.379",
                "26-03-13 15:16:03[I]:   Top model probability bins:",
            ]
        ),
        encoding="utf-8",
    )

    seed = extract_seed_state_from_log(
        log_path=log_path,
        event_name="Mar 6 - Mar 13",
        snapshot_timestamp=1773414963,
    )

    assert seed.available_capital == 14.76
    assert seed.event_budget == 1000.0
    assert 18 in seed.positions
    assert seed.positions[18].no_shares == 357.3
