import asyncio

from src.algo.musk_tweet_count.notifications import SlackConfig, SlackNotifier


def test_slack_config_from_env(monkeypatch):
    monkeypatch.setenv("SLACK_ENABLED", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
    monkeypatch.setenv("SLACK_NOTIFY_DRY_RUN", "yes")
    monkeypatch.setenv("SLACK_HEALTH_INTERVAL_SECONDS", "900")
    monkeypatch.setenv("SLACK_HEALTH_ON_CHANGE_ONLY", "false")
    monkeypatch.setenv("SLACK_FILL_SUMMARY_INTERVAL_SECONDS", "1800")
    monkeypatch.setenv("SLACK_FILL_SUMMARY_MAX_EXAMPLES", "3")
    monkeypatch.setenv("SLACK_BALANCE_ALLOWANCE_COOLDOWN_SECONDS", "7200")
    monkeypatch.setenv("SLACK_MIN_LEVEL", "warning")

    config = SlackConfig.from_env()

    assert config.enabled is True
    assert config.bot_token == "xoxb-test"
    assert config.channel_id == "C123"
    assert config.notify_dry_run is True
    assert config.health_interval_seconds == 900
    assert config.health_on_change_only is False
    assert config.fill_summary_interval_seconds == 1800
    assert config.fill_summary_max_examples == 3
    assert config.balance_allowance_cooldown_seconds == 7200
    assert config.min_level == "warning"


def test_slack_notifier_disabled_for_dry_run_by_default():
    notifier = SlackNotifier(
        SlackConfig(
            enabled=True,
            bot_token="xoxb-test",
            channel_id="C123",
            notify_dry_run=False,
        ),
        dry_run=True,
    )

    assert notifier.enabled is False


def test_slack_notifier_dedupes_messages(monkeypatch):
    async def scenario():
        notifier = SlackNotifier(
            SlackConfig(
                enabled=True,
                bot_token="xoxb-test",
                channel_id="C123",
                notify_dry_run=True,
            ),
            dry_run=True,
        )
        sent_messages = []

        async def fake_post(text):
            sent_messages.append(text)
            return True

        await notifier.start()
        monkeypatch.setattr(notifier, "_post_message_with_retry", fake_post)

        notifier.notify_warning(
            "Realtime gap detected",
            ["Forcing authoritative XTracker refresh."],
            dedupe_key="realtime-gap",
            cooldown_seconds=60.0,
        )
        notifier.notify_warning(
            "Realtime gap detected",
            ["Forcing authoritative XTracker refresh."],
            dedupe_key="realtime-gap",
            cooldown_seconds=60.0,
        )

        await notifier.flush()
        await notifier.stop()

        assert sent_messages == [
            "[DRY RUN] [WARNING] Realtime gap detected\nForcing authoritative XTracker refresh."
        ]

    asyncio.run(scenario())
