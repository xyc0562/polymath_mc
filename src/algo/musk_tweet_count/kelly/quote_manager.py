"""
Passive maker quoting: resting GTD bids on bins Kelly wants to buy.

Maker mode is NOT a new strategy — it is the existing Kelly buy path
with a passive entry price. A quote is only placed on a bin where the
optimizer already wants to accumulate (same edge-buffer threshold, same
screening-utility test as the taker path), at a price inside the spread
instead of crossing it. Any fill is a position Kelly wanted anyway,
acquired cheaper.

Safety architecture:
- Desired-state reconciliation: every cycle rebuilds the wanted quote
  set from current state and diffs it against exchange-reported open
  orders. There is no incremental order state machine.
- Exchange-side GTD expiration is the crash-safety floor: every order
  dies at placement + ttl with no action from us.
- Hard gates (all must pass or everything is cancelled): quiet activity
  state from a fresh tracker, more than no_quote_final_hours to
  settlement, breaker closed, no integrity freeze, fresh position sync,
  fresh orderbook, minimum spread, strictly non-crossing prices.
- Kill-on-signal: any new post (countable or reply activity) triggers
  cancel-all plus a re-quote cooldown.

Gate calibration comes from the 2026-07 markout study: quiet/earlier
bin-price drift is ~0.1-0.4c over 5-30min vs 2-4c half-spread capture,
while final-12h and active/storm states run 2-10x hotter.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .activity_state import ActivityStateTracker
from .candidates import (
    MIN_ORDER_VALUE_USD,
    TradeAction,
    TradeCandidate,
    _compute_portfolio_utility_gain,
    compute_buy_yes_threshold,
)
from .config import KellyConfig, MakerConfig
from .executor import (
    ORDER_MANAGER_BREAKER,
    SYNC_STALENESS_HALT_SECONDS,
    _snap_price_to_tick,
)
from .maker_event_log import MakerEventLog
from .orderbook import UnifiedOrderbook
from .portfolio import Portfolio

logger = logging.getLogger(__name__)

# Screening chunk for the maker utility test, mirroring candidate
# generation's SCREENING_CHUNK_USD: small enough that utility gain
# approximates marginal utility at the quote price.
MAKER_SCREENING_USD = 2.0

# Repost when less than this fraction of the TTL remains, so a healthy
# quote is refreshed before exchange-side expiration opens a gap.
REPOST_TTL_FRACTION = 1.0 / 3.0


@dataclass
class DesiredQuote:
    """A quote the reconcile cycle wants resting on the book."""

    bin_index: int
    action: TradeAction  # BUY_YES or BUY_NO
    token_id: str
    price: float
    size: int
    fair_value: float  # model probability of the quoted token
    screening_utility: float
    best_bid: float  # quoted side's touch at planning time (for spread capture)
    best_ask: float


@dataclass
class RestingOrder:
    """A maker order we believe is open at the exchange (or virtual, in shadow)."""

    order_id: str
    bin_index: int
    action: TradeAction
    token_id: str
    price: float
    size: float
    placed_at: float
    expires_at: float
    fair_value: float  # model probability of the quoted token at placement


class QuoteManager:
    """Per-event manager for resting maker quotes.

    One reconcile() call per trading tick, after the taker path has run:
    refresh open-order truth, evaluate gates, compute desired quotes,
    diff, act. All state is rebuilt from exchange truth each cycle; the
    local registry only bridges the gap between cycles (collateral
    accounting and kill-switch cancels).
    """

    def __init__(
        self,
        maker_config: MakerConfig,
        kelly_config: KellyConfig,
        order_executor,
        kelly_executor,
        portfolio: Portfolio,
        token_ids: Dict[int, str],
        no_token_ids: Dict[int, Optional[str]],
        event_name: str = "unknown",
        activity_tracker: Optional[ActivityStateTracker] = None,
        user_stream=None,
    ):
        self.config = maker_config
        self.kelly_config = kelly_config
        self.order_executor = order_executor
        self.kelly_executor = kelly_executor
        self.portfolio = portfolio
        self.token_ids = dict(token_ids)
        self.no_token_ids = dict(no_token_ids)
        self.event_name = event_name
        self.activity_tracker = activity_tracker
        self.user_stream = user_stream

        # order_id -> RestingOrder. In shadow mode order_ids are synthetic.
        self._orders: Dict[str, RestingOrder] = {}
        self._shadow_seq = 0
        self._cooldown_until: float = 0.0
        self._last_kill_reason: Optional[str] = None
        self._last_gate_failure: Optional[str] = None
        self._shadow_would_fills: int = 0

        # Per-reconcile histogram of why bins/sides did NOT produce a quote,
        # populated by compute_desired_quotes. Emitted (deduped) when the gate
        # is open but the desired set is empty — the diagnostic that was
        # missing when the shadow ran 11 days and placed zero quotes.
        self._last_quote_diag: Dict[str, int] = {}
        self._last_quote_diag_emitted: Optional[tuple] = None

        # Durable structured record + forward-markout tracking. The event log
        # is a no-op when no directory is configured (default in tests).
        self.event_log = MakerEventLog(
            maker_config.event_log_dir, event_name, maker_config.mode
        )
        self._markout_horizons = sorted(
            float(h) for h in maker_config.markout_horizons_seconds if h > 0
        )
        # Each entry tracks the quoted token's mid at each horizon after a fill.
        self._pending_markouts: List[dict] = []

        # Every token we may quote or have quoted, for filtering the
        # account-wide open-orders listing down to this event.
        self._own_tokens = {t for t in self.token_ids.values() if t}
        self._own_tokens |= {t for t in self.no_token_ids.values() if t}

    def attach(self) -> None:
        """Hook into the Kelly executor: collateral lock + self-trade guard."""
        self.kelly_executor.maker_collateral_provider = self.open_collateral
        self.kelly_executor.maker_pre_trade_hook = self.cancel_bin

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    def open_collateral(self) -> float:
        """USDC locked in open resting bids (0 in shadow: nothing rests)."""
        if self.config.shadow:
            return 0.0
        return sum(o.price * o.size for o in self._orders.values())

    def get_status(self) -> dict:
        return {
            "mode": self.config.mode,
            "open_orders": len(self._orders),
            "open_collateral": round(self.open_collateral(), 2),
            "cooldown_remaining": max(0.0, self._cooldown_until - time.time()),
            "last_kill_reason": self._last_kill_reason,
            "last_gate_failure": self._last_gate_failure,
            "shadow_would_fills": self._shadow_would_fills,
            "pending_markouts": len(self._pending_markouts),
            "event_log": str(self.event_log.path) if self.event_log.path else None,
        }

    # ------------------------------------------------------------------
    # Kill switches
    # ------------------------------------------------------------------

    def kill(self, reason: str) -> None:
        """Synchronous kill switch: start cancel-all and arm the cooldown.

        Safe to call from any async context (manager's realtime ingest,
        prob-jump detection). Never raises.
        """
        self._cooldown_until = time.time() + self.config.requote_cooldown_seconds
        self._last_kill_reason = reason
        if not self._orders:
            return
        logger.info(
            f"[{self.event_name}][MAKER] Kill switch ({reason}): cancelling "
            f"{len(self._orders)} resting quote(s)"
        )
        if self.config.shadow:
            self._orders.clear()
            return
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.cancel_all(reason))
        except RuntimeError:
            # No running loop (sync test context): cancel inline.
            self._cancel_order_ids(list(self._orders.keys()))

    async def cancel_all(self, reason: str) -> None:
        """Cancel every resting quote."""
        if not self._orders:
            return
        if self.config.shadow:
            logger.info(
                f"[{self.event_name}][MAKER-SHADOW] cancel_all ({reason}): "
                f"{len(self._orders)} virtual quote(s) dropped"
            )
            self._orders.clear()
            return
        self._cancel_order_ids(list(self._orders.keys()))

    async def cancel_bin(self, bin_index: int) -> None:
        """Self-trade guard: drop resting quotes on a bin before the taker
        path submits an order there."""
        ids = [oid for oid, o in self._orders.items() if o.bin_index == bin_index]
        if not ids:
            return
        if self.config.shadow:
            for oid in ids:
                self._orders.pop(oid, None)
            logger.info(
                f"[{self.event_name}][MAKER-SHADOW] taker conflict on bin {bin_index}: "
                f"dropped {len(ids)} virtual quote(s)"
            )
            return
        logger.info(
            f"[{self.event_name}][MAKER] taker conflict on bin {bin_index}: "
            f"cancelling {len(ids)} resting quote(s)"
        )
        self._cancel_order_ids(ids)

    def _cancel_order_ids(self, order_ids: List[str]) -> None:
        """Cancel and deregister; on failure keep tracking (collateral stays
        counted) until the exchange listing confirms the order is gone —
        GTD expiry bounds the worst case."""
        if not order_ids:
            return
        if self.order_executor.cancel_orders(order_ids):
            for oid in order_ids:
                self._orders.pop(oid, None)
        else:
            logger.warning(
                f"[{self.event_name}][MAKER] cancel failed for {len(order_ids)} "
                f"order(s); keeping tracked until expiry/reconcile"
            )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _check_gates(self, hours_to_settlement: float, now: float) -> Tuple[bool, str]:
        cfg = self.config

        if hours_to_settlement <= cfg.no_quote_final_hours:
            return False, f"final_{cfg.no_quote_final_hours:.0f}h_window"

        if now < self._cooldown_until:
            return False, f"cooldown_{self._last_kill_reason or 'unknown'}"

        if ORDER_MANAGER_BREAKER.is_open():
            return False, "order_manager_breaker_open"

        if self.kelly_executor._integrity_state.frozen:
            return False, "integrity_frozen"

        if not self.config.shadow and not self.order_executor.dry_run:
            last_sync = self.kelly_executor._last_successful_sync_time
            if last_sync is None or now - last_sync > SYNC_STALENESS_HALT_SECONDS:
                return False, "position_sync_stale"

        tracker = self.activity_tracker
        if tracker is None:
            return False, "no_activity_tracker"
        poll_age = tracker.seconds_since_poll(now)
        if poll_age is None or poll_age > cfg.activity_staleness_seconds:
            return False, "activity_tracker_stale"
        state = tracker.state(now)
        if state != "quiet":
            return False, f"activity_{state}"

        return True, ""

    # ------------------------------------------------------------------
    # Desired quotes
    # ------------------------------------------------------------------

    def _bin_book_view(
        self, orderbook: UnifiedOrderbook, action: TradeAction
    ) -> Optional[Tuple[float, float]]:
        """(best_bid, best_ask) of the quoted token's own book side."""
        if action == TradeAction.BUY_YES:
            bid, ask = orderbook.best_yes_bid, orderbook.best_yes_ask
        else:
            bid, ask = orderbook.best_no_bid, orderbook.best_no_ask
        if bid is None or ask is None:
            return None
        return bid, ask

    def _quoted_mid(
        self, orderbook: Optional[UnifiedOrderbook], action: TradeAction
    ) -> Optional[float]:
        """Mid of the quoted token's own book side, or None if not two-sided."""
        if orderbook is None:
            return None
        view = self._bin_book_view(orderbook, action)
        if view is None:
            return None
        bid, ask = view
        return (bid + ask) / 2.0

    # ------------------------------------------------------------------
    # Forward markout tracking
    # ------------------------------------------------------------------

    def _register_markout(
        self, order: RestingOrder, fill_mid: Optional[float], now: float
    ) -> None:
        """Start tracking the quoted token's mid at each horizon after a fill."""
        if not self._markout_horizons:
            return
        self._pending_markouts.append(
            {
                "token_id": order.token_id,
                "bin_index": order.bin_index,
                "action": order.action,
                "side": order.action.value,
                "fill_ts": now,
                "fill_price": order.price,
                "fill_mid": fill_mid,
                "fair_value": order.fair_value,
                "horizons": list(self._markout_horizons),
                "samples": {},  # str(int(horizon)) -> mid (or None if book gone)
            }
        )

    def _sample_markouts(
        self, orderbooks: Dict[int, UnifiedOrderbook], now: float
    ) -> None:
        """Sample any due horizons for pending fills; finalize when complete."""
        if not self._pending_markouts:
            return
        overdue = max(self._markout_horizons) + self.config.max_book_age_seconds
        still: List[dict] = []
        for pm in self._pending_markouts:
            orderbook = orderbooks.get(pm["bin_index"])
            for horizon in pm["horizons"]:
                key = str(int(horizon))
                if key in pm["samples"]:
                    continue
                if now - pm["fill_ts"] >= horizon:
                    pm["samples"][key] = self._quoted_mid(orderbook, pm["action"])
            complete = all(str(int(h)) in pm["samples"] for h in pm["horizons"])
            if complete or now - pm["fill_ts"] > overdue:
                self._finalize_markout(pm, now)
            else:
                still.append(pm)
        self._pending_markouts = still

    def _finalize_markout(self, pm: dict, now: float) -> None:
        """Emit the completed markout record for one fill."""
        horizon_mids = {}
        markout = {}
        for horizon in pm["horizons"]:
            key = str(int(horizon))
            mid = pm["samples"].get(key)
            horizon_mids[key] = mid
            # Quoted token is bought at fill_price; markout is favorable when
            # its mid drifts up afterward (we hold the position we bought).
            markout[key] = round(mid - pm["fill_price"], 6) if mid is not None else None
        self.event_log.emit(
            "markout",
            now=now,
            bin=pm["bin_index"],
            token_id=pm["token_id"],
            side=pm["side"],
            fill_price=pm["fill_price"],
            fill_mid=pm["fill_mid"],
            fair_value=pm["fair_value"],
            horizon_mids=horizon_mids,
            markout=markout,
        )

    def _emit_quote_placed(self, quote: DesiredQuote, now: float) -> None:
        self.event_log.emit(
            "quote_placed",
            now=now,
            bin=quote.bin_index,
            token_id=quote.token_id,
            side=quote.action.value,
            price=quote.price,
            size=quote.size,
            fair_value=quote.fair_value,
            screening_utility=round(quote.screening_utility, 6),
            best_bid=quote.best_bid,
            best_ask=quote.best_ask,
            spread=round(quote.best_ask - quote.best_bid, 6),
        )

    def _candidate_for_side(
        self,
        bin_index: int,
        action: TradeAction,
        fair: float,
        orderbook: UnifiedOrderbook,
        effective: Portfolio,
        tick_str: str,
        budget_remaining: float,
        capital_remaining: float,
    ) -> Tuple[Optional[DesiredQuote], str]:
        """Build the passive-entry quote for one side of one bin.

        Returns (quote, "") on success or (None, reason) so the caller can
        aggregate why no quote was produced.
        """
        cfg = self.config
        tick = float(tick_str)

        token_id = (
            self.token_ids.get(bin_index)
            if action == TradeAction.BUY_YES
            else self.no_token_ids.get(bin_index)
        )
        if not token_id:
            return None, "no_token"

        view = self._bin_book_view(orderbook, action)
        if view is None:
            return None, "no_book_view"
        best_bid, best_ask = view

        # Same entry-price ceiling as the taker path: friction-adjusted
        # fair value. Quoting above it would be a trade Kelly rejects.
        threshold = compute_buy_yes_threshold(fair, self.kelly_config.edge_buffer)
        if threshold <= 0:
            return None, "threshold_nonpositive"

        # Improve the touch by one tick, but never beyond the threshold
        # and never crossing (strictly below the ask). The epsilon keeps
        # float noise (e.g. 1 - ask on the NO side) from snapping a full
        # tick lower than intended.
        price = min(best_bid + tick, threshold, best_ask - tick)
        price = _snap_price_to_tick(price + 1e-9, tick_str, mode="floor")
        if price < best_bid - 1e-9:
            # The touch already outbids our max acceptable price; resting
            # behind it would never fill at useful frequency.
            return None, "touch_above_max_price"
        if not (cfg.quote_zone_min <= price <= cfg.quote_zone_max):
            return None, "outside_quote_zone"
        if price >= best_ask:
            return None, "price_crosses_ask"

        # Screening utility at the quote price: identical test to taker
        # candidate generation, just at our passive price.
        screen_shares = max(1.0, MAKER_SCREENING_USD / price)
        if action == TradeAction.BUY_YES:
            after = effective.simulate_buy_yes(bin_index, screen_shares, price)
        else:
            after = effective.simulate_buy_no(bin_index, screen_shares, price)
        utility = _compute_portfolio_utility_gain(effective, after, self.kelly_config)
        if utility < self.kelly_config.min_buy_utility:
            return None, "below_min_utility"

        # Sizing: capped by per-quote max, per-bin collateral remaining,
        # and the running budget/capital room the caller tracks across the
        # desired set (planned as the post-reconcile target state, so a
        # currently-resting quote's own collateral does not count against
        # its replacement).
        position = effective.get_position(bin_index)
        bin_used = position.collateral_used if position else 0.0
        c_bin_max = self.kelly_config.collateral.c_bin_max
        bin_room = max(0.0, c_bin_max - bin_used) if c_bin_max > 0 else float("inf")

        dollars = min(
            cfg.max_quote_usd,
            bin_room,
            max(0.0, budget_remaining),
            max(0.0, capital_remaining),
        )
        if dollars < max(cfg.min_quote_usd, MIN_ORDER_VALUE_USD):
            return None, "below_min_size_dollars"
        size = int(math.floor(dollars / price))
        if size < 1:
            return None, "size_lt_1"

        return DesiredQuote(
            bin_index=bin_index,
            action=action,
            token_id=token_id,
            price=price,
            size=size,
            fair_value=fair,
            screening_utility=utility,
            best_bid=best_bid,
            best_ask=best_ask,
        ), ""

    def _event_allocation(self) -> float:
        limit = self.portfolio.external_capital_limit
        if limit is not None:
            return limit
        return self.kelly_config.collateral.c_event_max

    def compute_desired_quotes(
        self,
        orderbooks: Dict[int, UnifiedOrderbook],
        now: Optional[float] = None,
    ) -> List[DesiredQuote]:
        """Desired quote set from current model + book state. At most one
        quote per bin (the side Kelly wants; both passing is impossible
        with positive friction)."""
        now = now if now is not None else time.time()
        cfg = self.config
        desired: List[DesiredQuote] = []
        diag: Dict[str, int] = {}

        def note(reason: str) -> None:
            diag[reason] = diag.get(reason, 0) + 1

        effective = self.kelly_executor._build_effective_portfolio()
        # The replayed base carries stale probabilities; planning uses the
        # bot portfolio's current (post-EMA, post-consensus) values.
        effective.probabilities = list(self.portfolio.probabilities)
        effective.dead_bins = list(self.portfolio.dead_bins)

        if not effective.probabilities:
            self._last_quote_diag = {"no_probabilities": 1}
            return []

        # Plan the post-reconcile target state: currently-resting quotes
        # will be freed or replaced by the diff, so their locked collateral
        # is added back before splitting the budget across the desired set.
        budget_remaining = self.config.budget_fraction * self._event_allocation()
        capital_remaining = max(0.0, effective.available_capital) + self.open_collateral()

        dead = set(effective.dead_bins)
        for bin_index, yes_token in self.token_ids.items():
            if bin_index in dead or bin_index >= len(effective.probabilities):
                note("bin_dead_or_oob")
                continue
            orderbook = orderbooks.get(bin_index)
            if orderbook is None:
                note("no_orderbook")
                continue
            if (
                orderbook.last_updated is None
                or now - orderbook.last_updated > cfg.max_book_age_seconds
            ):
                note("book_stale")
                continue
            spread = orderbook.yes_spread
            if spread is None or spread < cfg.min_spread:
                note("spread_below_min")
                continue

            fair_yes = effective.probabilities[bin_index]
            tick_str = self.order_executor.get_tick_size(yes_token) or "0.01"

            best: Optional[DesiredQuote] = None
            for action, fair in (
                (TradeAction.BUY_YES, fair_yes),
                (TradeAction.BUY_NO, 1.0 - fair_yes),
            ):
                q, reason = self._candidate_for_side(
                    bin_index,
                    action,
                    fair,
                    orderbook,
                    effective,
                    tick_str,
                    budget_remaining,
                    capital_remaining,
                )
                if q is None:
                    note(f"side_{reason}")
                elif best is None or q.screening_utility > best.screening_utility:
                    best = q
            if best is not None:
                note("quoted")
                desired.append(best)
                cost = best.price * best.size
                budget_remaining -= cost
                capital_remaining -= cost

        self._last_quote_diag = diag
        return desired

    def _emit_quote_diag_if_changed(self, now: float) -> None:
        """When the gate is open but no quote was produced, emit the reason
        histogram (deduped by content) so the empty desired-set is diagnosable.
        No-op when a quote was produced (`quoted` present)."""
        diag = self._last_quote_diag
        if not diag or diag.get("quoted"):
            self._last_quote_diag_emitted = None
            return
        signature = tuple(sorted(diag.items()))
        if signature == self._last_quote_diag_emitted:
            return
        self._last_quote_diag_emitted = signature
        self.event_log.emit("no_quotes", now=now, reasons=dict(diag))
        logger.info(
            f"[{self.event_name}][MAKER] gate open, no quotes: {dict(diag)}"
        )

    # ------------------------------------------------------------------
    # Reconcile
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        now: Optional[float] = None,
    ) -> None:
        """One maker cycle: refresh truth, gate, compute desired, diff, act."""
        if not self.config.enabled:
            return
        now = now if now is not None else time.time()

        if self.config.shadow:
            self._reconcile_shadow(orderbooks, hours_to_settlement, now)
            return

        self._sample_markouts(orderbooks, now)

        # 1. Refresh registry from exchange truth. On listing failure fail
        #    closed: no new quotes this cycle, keep tracking what we have.
        listed = self.order_executor.get_open_orders()
        if listed is None:
            logger.warning(
                f"[{self.event_name}][MAKER] open-orders listing failed; "
                f"skipping quote cycle (fail closed)"
            )
            return
        open_ours: Dict[str, dict] = {}
        unknown_ids: List[str] = []
        for entry in listed:
            asset = entry.get("asset_id") or entry.get("assetId") or ""
            if asset not in self._own_tokens:
                continue
            oid = entry.get("id") or entry.get("orderID") or ""
            if not oid:
                continue
            if oid in self._orders:
                open_ours[oid] = entry
            else:
                # Not ours in this process's lifetime: leftover from a
                # previous run or a manual order on a bot token. Cancel
                # loudly — takers are FAK and never rest.
                unknown_ids.append(oid)
        if unknown_ids:
            logger.warning(
                f"[{self.event_name}][MAKER] cancelling {len(unknown_ids)} unknown "
                f"resting order(s) on event tokens (restart leftovers?)"
            )
            self.order_executor.cancel_orders(unknown_ids)

        # Registry entries the exchange no longer lists are gone
        # (filled, expired, or cancelled) — drop them.
        for oid in list(self._orders.keys()):
            if oid not in open_ours:
                self._orders.pop(oid, None)

        # 2. Gates.
        ok, reason = self._check_gates(hours_to_settlement, now)
        if not ok:
            if reason != self._last_gate_failure:
                logger.info(f"[{self.event_name}][MAKER] gated off: {reason}")
                self.event_log.emit("gate_change", now=now, gated=True, reason=reason)
            self._last_gate_failure = reason
            await self.cancel_all(f"gate:{reason}")
            return
        if self._last_gate_failure is not None:
            self.event_log.emit("gate_change", now=now, gated=False, reason=None)
        self._last_gate_failure = None

        # 3. Desired set.
        desired = self.compute_desired_quotes(orderbooks, now)
        self._emit_quote_diag_if_changed(now)
        desired_by_key = {(q.bin_index, q.action): q for q in desired}

        # 4. Diff: cancel stale/mispriced/expiring, then post missing.
        to_cancel: List[str] = []
        kept_keys = set()
        for oid, order in self._orders.items():
            key = (order.bin_index, order.action)
            want = desired_by_key.get(key)
            tick = float(self.order_executor.get_tick_size(order.token_id) or 0.01)
            if want is None:
                to_cancel.append(oid)
            elif abs(want.price - order.price) >= tick - 1e-9:
                to_cancel.append(oid)
            elif order.expires_at - now < self.config.ttl_seconds * REPOST_TTL_FRACTION:
                to_cancel.append(oid)
            else:
                kept_keys.add(key)
        if to_cancel:
            self._cancel_order_ids(to_cancel)
            # A failed cancel leaves the old order resting; do not post a
            # replacement on the same key this cycle (no double quotes).
            for oid in to_cancel:
                still_open = self._orders.get(oid)
                if still_open is not None:
                    kept_keys.add((still_open.bin_index, still_open.action))

        for key, quote in desired_by_key.items():
            if key in kept_keys:
                continue
            self._place_quote(quote, now)

    def _place_quote(self, quote: DesiredQuote, now: float) -> None:
        response = self.order_executor.place_gtd_order(
            token_id=quote.token_id,
            side="BUY",
            price=quote.price,
            size=float(quote.size),
            ttl_seconds=self.config.ttl_seconds,
        )
        order_id = (response or {}).get("orderID") or None
        if not order_id:
            logger.warning(
                f"[{self.event_name}][MAKER] quote rejected: bin={quote.bin_index} "
                f"{quote.action.value} {quote.size} @ {quote.price:.4f} "
                f"err={(response or {}).get('errorMsg', 'no response')}"
            )
            return

        self._orders[order_id] = RestingOrder(
            order_id=order_id,
            bin_index=quote.bin_index,
            action=quote.action,
            token_id=quote.token_id,
            price=quote.price,
            size=float(quote.size),
            placed_at=now,
            expires_at=now + self.config.ttl_seconds,
            fair_value=quote.fair_value,
        )
        self._emit_quote_placed(quote, now)

        # Register with the existing fill machinery so a fill flows into
        # the overlay ledger exactly like a taker fill.
        candidate = TradeCandidate(
            bin_index=quote.bin_index,
            action=quote.action,
            size=float(quote.size),
            price=quote.price,
            utility_gain=quote.screening_utility,
            reservation_price=quote.fair_value,
            edge=(quote.fair_value - quote.price) / quote.fair_value
            if quote.fair_value > 0
            else 0.0,
            limit_price=quote.price,
            kind="maker",
        )
        self.kelly_executor._pending_orders[order_id] = (candidate, quote.token_id)
        self.kelly_executor._remember_order_context(order_id, candidate, quote.token_id)

        if self.user_stream is not None:
            from .user_stream import PendingOrder

            pending = PendingOrder(
                order_id=order_id,
                token_id=quote.token_id,
                side="BUY",
                price=quote.price,
                size=float(quote.size),
                bin_index=quote.bin_index,
                condition_id=quote.token_id,
                stale_after=self.config.ttl_seconds + 60.0,
                resting=True,
            )
            asyncio.create_task(self.user_stream.add_pending_order(pending))

        logger.info(
            f"[{self.event_name}][MAKER] quote placed: bin={quote.bin_index} "
            f"{quote.action.value} {quote.size} @ {quote.price:.4f} "
            f"(fair={quote.fair_value:.4f}, util={quote.screening_utility:.6f}, "
            f"ttl={self.config.ttl_seconds:.0f}s, id={order_id[:16]}...)"
        )

    # ------------------------------------------------------------------
    # Shadow mode
    # ------------------------------------------------------------------

    def _reconcile_shadow(
        self,
        orderbooks: Dict[int, UnifiedOrderbook],
        hours_to_settlement: float,
        now: float,
    ) -> None:
        """Same cycle as live, but virtual: no REST calls, no orders.

        Would-be fills are detected when the book trades down through a
        virtual bid (best bid drops below our price, or the ask crosses
        it) — the conditional-markout sample the live pilot decision
        needs.
        """
        self._sample_markouts(orderbooks, now)

        # Expire virtual orders past TTL.
        for oid in list(self._orders.keys()):
            if self._orders[oid].expires_at <= now:
                order = self._orders.pop(oid)
                self.event_log.emit(
                    "expire",
                    now=now,
                    bin=order.bin_index,
                    token_id=order.token_id,
                    side=order.action.value,
                    price=order.price,
                    size=order.size,
                    rested_seconds=round(now - order.placed_at, 1),
                )
                logger.info(
                    f"[{self.event_name}][MAKER-SHADOW] expired unfilled: "
                    f"bin={order.bin_index} {order.action.value} "
                    f"{order.size:.0f} @ {order.price:.4f}"
                )

        # Would-be fill detection against the fresh books.
        for oid, order in list(self._orders.items()):
            orderbook = orderbooks.get(order.bin_index)
            if orderbook is None:
                continue
            view = self._bin_book_view(orderbook, order.action)
            if view is None:
                continue
            best_bid, best_ask = view
            if best_bid < order.price or best_ask <= order.price:
                self._orders.pop(oid, None)
                self._shadow_would_fills += 1
                fill_mid = (best_bid + best_ask) / 2.0
                self.event_log.emit(
                    "would_fill",
                    now=now,
                    bin=order.bin_index,
                    token_id=order.token_id,
                    side=order.action.value,
                    price=order.price,
                    size=order.size,
                    fair_value=order.fair_value,
                    fill_mid=round(fill_mid, 6),
                    best_bid=best_bid,
                    best_ask=best_ask,
                    rested_seconds=round(now - order.placed_at, 1),
                )
                self._register_markout(order, fill_mid, now)
                logger.info(
                    f"[{self.event_name}][MAKER-SHADOW] WOULD-FILL: "
                    f"bin={order.bin_index} {order.action.value} "
                    f"{order.size:.0f} @ {order.price:.4f} "
                    f"(book now bid={best_bid:.4f} ask={best_ask:.4f}, "
                    f"rested={now - order.placed_at:.0f}s, ts={now:.0f})"
                )

        ok, reason = self._check_gates(hours_to_settlement, now)
        if not ok:
            if self._orders:
                logger.info(
                    f"[{self.event_name}][MAKER-SHADOW] gated off ({reason}): "
                    f"dropping {len(self._orders)} virtual quote(s)"
                )
                self._orders.clear()
            if reason != self._last_gate_failure:
                logger.info(f"[{self.event_name}][MAKER-SHADOW] gated off: {reason}")
                self.event_log.emit("gate_change", now=now, gated=True, reason=reason)
            self._last_gate_failure = reason
            return
        if self._last_gate_failure is not None:
            self.event_log.emit("gate_change", now=now, gated=False, reason=None)
        self._last_gate_failure = None

        desired = self.compute_desired_quotes(orderbooks, now)
        self._emit_quote_diag_if_changed(now)
        desired_by_key = {(q.bin_index, q.action): q for q in desired}

        kept_keys = set()
        for oid, order in list(self._orders.items()):
            key = (order.bin_index, order.action)
            want = desired_by_key.get(key)
            tick = float(self.order_executor.get_tick_size(order.token_id) or 0.01)
            if want is None or abs(want.price - order.price) >= tick - 1e-9:
                self._orders.pop(oid, None)
                logger.info(
                    f"[{self.event_name}][MAKER-SHADOW] repriced/withdrawn: "
                    f"bin={order.bin_index} {order.action.value} @ {order.price:.4f}"
                )
            else:
                kept_keys.add(key)

        for key, quote in desired_by_key.items():
            if key in kept_keys:
                continue
            self._shadow_seq += 1
            oid = f"shadow-{self._shadow_seq}"
            self._orders[oid] = RestingOrder(
                order_id=oid,
                bin_index=quote.bin_index,
                action=quote.action,
                token_id=quote.token_id,
                price=quote.price,
                size=float(quote.size),
                placed_at=now,
                expires_at=now + self.config.ttl_seconds,
                fair_value=quote.fair_value,
            )
            self._emit_quote_placed(quote, now)
            logger.info(
                f"[{self.event_name}][MAKER-SHADOW] quote: bin={quote.bin_index} "
                f"{quote.action.value} {quote.size} @ {quote.price:.4f} "
                f"(fair={quote.fair_value:.4f}, util={quote.screening_utility:.6f}, "
                f"ts={now:.0f})"
            )
