"""
Slack notification helpers for long-running trading processes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import aiohttp

logger = logging.getLogger(__name__)


_LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default=%s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default=%s", name, raw, default)
        return default


@dataclass(frozen=True)
class SlackMessage:
    text: str
    dedupe_key: Optional[str] = None
    cooldown_seconds: float = 0.0


@dataclass(frozen=True)
class SlackConfig:
    enabled: bool = False
    bot_token: str = ""
    channel_id: str = ""
    notify_dry_run: bool = False
    health_interval_seconds: int = 21600
    health_on_change_only: bool = True
    fill_summary_interval_seconds: int = 3600
    fill_summary_quiet_seconds: int = 900
    fill_summary_max_examples: int = 5
    balance_allowance_cooldown_seconds: int = 3600
    mention_user_id: str = ""
    min_level: str = "info"
    request_timeout_seconds: float = 10.0
    queue_size: int = 200
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "SlackConfig":
        return cls(
            enabled=_env_bool("SLACK_ENABLED", False),
            bot_token=os.getenv("SLACK_BOT_TOKEN", "").strip(),
            channel_id=os.getenv("SLACK_CHANNEL_ID", "").strip(),
            notify_dry_run=_env_bool("SLACK_NOTIFY_DRY_RUN", False),
            health_interval_seconds=max(0, _env_int("SLACK_HEALTH_INTERVAL_SECONDS", 21600)),
            health_on_change_only=_env_bool("SLACK_HEALTH_ON_CHANGE_ONLY", True),
            fill_summary_interval_seconds=max(0, _env_int("SLACK_FILL_SUMMARY_INTERVAL_SECONDS", 3600)),
            fill_summary_quiet_seconds=max(0, _env_int("SLACK_FILL_SUMMARY_QUIET_SECONDS", 900)),
            fill_summary_max_examples=max(1, _env_int("SLACK_FILL_SUMMARY_MAX_EXAMPLES", 5)),
            balance_allowance_cooldown_seconds=max(0, _env_int("SLACK_BALANCE_ALLOWANCE_COOLDOWN_SECONDS", 3600)),
            mention_user_id=os.getenv("SLACK_MENTION_USER_ID", "").strip(),
            min_level=os.getenv("SLACK_MIN_LEVEL", "info").strip().lower() or "info",
            request_timeout_seconds=max(1.0, _env_float("SLACK_REQUEST_TIMEOUT_SECONDS", 10.0)),
            queue_size=max(1, _env_int("SLACK_QUEUE_SIZE", 200)),
            max_retries=max(1, _env_int("SLACK_MAX_RETRIES", 3)),
        )


class SlackNotifier:
    """Async Slack sender with best-effort queueing semantics."""

    POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

    def __init__(self, config: SlackConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.health_interval_seconds = config.health_interval_seconds
        self.health_on_change_only = config.health_on_change_only
        self.fill_summary_interval_seconds = config.fill_summary_interval_seconds
        self.fill_summary_quiet_seconds = config.fill_summary_quiet_seconds
        self.fill_summary_max_examples = config.fill_summary_max_examples
        self.balance_allowance_cooldown_seconds = config.balance_allowance_cooldown_seconds
        self._level_threshold = _LEVELS.get(config.min_level, _LEVELS["info"])
        self._queue: asyncio.Queue[Optional[SlackMessage]] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._last_sent_at: dict[str, float] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._started = False
        self._accepting = False
        self._enabled = self._compute_enabled()

    @classmethod
    def from_env(cls, *, dry_run: bool = False) -> "SlackNotifier":
        return cls(SlackConfig.from_env(), dry_run=dry_run)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _compute_enabled(self) -> bool:
        if not self.config.enabled:
            return False
        if self.dry_run and not self.config.notify_dry_run:
            return False
        if not self.config.bot_token or not self.config.channel_id:
            return False
        return True

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._accepting = self._enabled

        if not self.config.enabled:
            logger.info("Slack notifier disabled (SLACK_ENABLED is false)")
            return
        if self.dry_run and not self.config.notify_dry_run:
            logger.info("Slack notifier disabled for dry-run mode")
            return
        if not self.config.bot_token or not self.config.channel_id:
            logger.warning(
                "Slack notifier enabled but missing SLACK_BOT_TOKEN or SLACK_CHANNEL_ID"
            )
            return

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="slack_notifier",
        )
        logger.info("Slack notifier enabled for channel %s", self.config.channel_id)

    async def flush(self) -> None:
        if not self._started or not self._worker_task:
            return
        await self._queue.join()

    async def stop(self) -> None:
        if not self._started:
            return

        self._accepting = False
        if self._worker_task:
            await self.flush()
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None

        if self._session:
            await self._session.close()
            self._session = None

    def notify(
        self,
        level: str,
        title: str,
        lines: Optional[Iterable[object]] = None,
        *,
        dedupe_key: Optional[str] = None,
        cooldown_seconds: float = 0.0,
        mention: bool = False,
    ) -> None:
        if not self._accepting:
            return
        if _LEVELS.get(level, _LEVELS["info"]) < self._level_threshold:
            return

        text = self._format_message(level, title, lines)
        if mention and self.config.mention_user_id:
            text = f"<@{self.config.mention_user_id}>\n{text}"
        try:
            self._queue.put_nowait(
                SlackMessage(
                    text=text,
                    dedupe_key=dedupe_key,
                    cooldown_seconds=max(0.0, cooldown_seconds),
                )
            )
        except asyncio.QueueFull:
            logger.warning("Slack notification queue full, dropping message: %s", title)

    def notify_info(
        self,
        title: str,
        lines: Optional[Iterable[object]] = None,
        *,
        mention: bool = False,
        **kwargs,
    ) -> None:
        self.notify("info", title, lines, mention=mention, **kwargs)

    def notify_warning(
        self,
        title: str,
        lines: Optional[Iterable[object]] = None,
        *,
        mention: bool = False,
        **kwargs,
    ) -> None:
        self.notify("warning", title, lines, mention=mention, **kwargs)

    def notify_error(
        self,
        title: str,
        lines: Optional[Iterable[object]] = None,
        *,
        mention: bool = False,
        **kwargs,
    ) -> None:
        self.notify("error", title, lines, mention=mention, **kwargs)

    def _format_message(
        self,
        level: str,
        title: str,
        lines: Optional[Iterable[object]],
    ) -> str:
        parts = [f"[{level.upper()}] {title}"]
        for line in lines or ():
            if line is None:
                continue
            rendered = str(line).strip()
            if rendered:
                parts.append(rendered)
        text = "\n".join(parts)
        if self.dry_run:
            return f"[DRY RUN] {text}"
        return text

    def _suppressed(self, message: SlackMessage) -> bool:
        if not message.dedupe_key or message.cooldown_seconds <= 0:
            return False
        last_sent = self._last_sent_at.get(message.dedupe_key)
        if last_sent is None:
            return False
        return (time.monotonic() - last_sent) < message.cooldown_seconds

    async def _worker_loop(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                if message is None:
                    return
                if self._suppressed(message):
                    continue
                sent = await self._post_message_with_retry(message.text)
                if sent and message.dedupe_key and message.cooldown_seconds > 0:
                    self._last_sent_at[message.dedupe_key] = time.monotonic()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Slack notifier worker error: %s", exc)
            finally:
                self._queue.task_done()

    async def _post_message_with_retry(self, text: str) -> bool:
        if not self._session:
            return False

        payload = {
            "channel": self.config.channel_id,
            "text": text,
            "mrkdwn": True,
        }
        headers = {
            "Authorization": f"Bearer {self.config.bot_token}",
        }

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self._session.post(
                    self.POST_MESSAGE_URL,
                    json=payload,
                    headers=headers,
                ) as response:
                    data = await response.json(content_type=None)
                    if response.status == 429:
                        retry_after = float(response.headers.get("Retry-After", "1"))
                        await asyncio.sleep(min(retry_after, 60.0))
                        continue

                    if response.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=str(data),
                            headers=response.headers,
                        )

                    if response.status != 200 or not data.get("ok", False):
                        logger.warning(
                            "Slack API rejected message: status=%s error=%s",
                            response.status,
                            data.get("error"),
                        )
                        return False

                    return True
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= self.config.max_retries:
                    logger.warning("Failed to send Slack message after retries: %s", exc)
                    return False
                await asyncio.sleep(min(2 ** (attempt - 1), 10.0))

        return False
