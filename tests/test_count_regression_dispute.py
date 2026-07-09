from datetime import datetime, timedelta, timezone

from src.algo.musk_tweet_count.forecaster.trading_bot import (
    GASKellyTradingBot,
    TradingBotConfig,
)


def _make_bot(**config_overrides) -> GASKellyTradingBot:
    config = TradingBotConfig(
        count_regression_min_confirmations=3,
        count_regression_min_age_seconds=1200.0,
        count_regression_posts_tolerance=2,
        **config_overrides,
    )
    bot = object.__new__(GASKellyTradingBot)
    bot.config = config
    bot._authoritative_count = None
    bot._count_dispute_value = None
    bot._count_dispute_first_ts = None
    bot._count_dispute_confirmations = 0
    return bot


def _age_dispute(bot: GASKellyTradingBot, seconds: float) -> None:
    bot._count_dispute_first_ts = datetime.now(timezone.utc) - timedelta(seconds=seconds)


def test_higher_count_accepted_immediately():
    bot = _make_bot()

    assert bot.set_authoritative_count(400) is True
    assert bot.set_authoritative_count(412) is True
    assert bot._authoritative_count == 412


def test_single_regression_keeps_higher_value():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    changed = bot.set_authoritative_count(396, posts_count=396)

    assert changed is False
    assert bot._authoritative_count == 412
    assert bot._count_dispute_value == 396


def test_regression_rebases_after_confirmations_age_and_posts_agreement():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    assert bot.set_authoritative_count(396, posts_count=396) is False
    assert bot.set_authoritative_count(396, posts_count=396) is False
    _age_dispute(bot, 1300)

    changed = bot.set_authoritative_count(396, posts_count=397)

    assert changed is True
    assert bot._authoritative_count == 396
    assert bot._count_dispute_value is None  # dispute cleared


def test_regression_not_rebased_without_posts_agreement():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    bot.set_authoritative_count(396, posts_count=411)
    bot.set_authoritative_count(396, posts_count=411)
    _age_dispute(bot, 1300)

    changed = bot.set_authoritative_count(396, posts_count=411)

    assert changed is False
    assert bot._authoritative_count == 412


def test_regression_not_rebased_before_min_age():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    assert bot.set_authoritative_count(396, posts_count=396) is False
    assert bot.set_authoritative_count(396, posts_count=396) is False
    # confirmations met (3rd call) but dispute is still young
    assert bot.set_authoritative_count(396, posts_count=396) is False
    assert bot._authoritative_count == 412


def test_recovered_count_clears_dispute():
    bot = _make_bot()
    bot.set_authoritative_count(412)
    bot.set_authoritative_count(396, posts_count=396)

    bot.set_authoritative_count(412, posts_count=412)

    assert bot._count_dispute_value is None
    assert bot._count_dispute_confirmations == 0
    assert bot._authoritative_count == 412


def test_dead_bins_use_disputed_lower_value_while_open():
    """A bin must never be marked dead on the basis of a contested count."""
    bot = _make_bot()
    bot.set_authoritative_count(412)
    bot.set_authoritative_count(396, posts_count=396)

    # Pinned value stays 412 for forecasting, but the dead-bin floor is 396
    assert bot._authoritative_count == 412
    assert bot.get_effective_count_for_dead_bins(380) == 396
    # Posts-derived count above the disputed value still wins
    assert bot.get_effective_count_for_dead_bins(405) == 405


def test_dead_bins_use_pinned_value_when_no_dispute():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    assert bot.get_effective_count_for_dead_bins(380) == 412
    assert bot.get_effective_count_for_dead_bins(500) == 500


def test_rebase_target_follows_latest_disputed_value():
    bot = _make_bot()
    bot.set_authoritative_count(412)

    bot.set_authoritative_count(396, posts_count=398)
    bot.set_authoritative_count(398, posts_count=398)
    _age_dispute(bot, 1300)
    changed = bot.set_authoritative_count(398, posts_count=398)

    assert changed is True
    assert bot._authoritative_count == 398
