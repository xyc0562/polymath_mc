"""
Capital pool manager for multi-event trading.

Manages a shared capital pool that multiple concurrent events can draw from.
Each event requests capital on start and returns it at settlement.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventAllocation:
    """Tracks capital allocated to a single event."""

    event_id: str
    initial_allocation: float  # Capital allocated at start
    current_value: float  # Current portfolio value (cash + positions)
    allocated_at: datetime
    settled_at: Optional[datetime] = None
    final_value: Optional[float] = None  # Value at settlement


@dataclass
class CapitalPoolConfig:
    """Configuration for capital pool."""

    # Total capital in the pool (0 = auto-detect from API on startup)
    total_capital: float = 0.0

    # Minimum capital to allocate (don't start event if less available)
    min_allocation: float = 100.0


class CapitalPool:
    """
    Manages shared capital across multiple concurrent events.

    Features:
    - Thread-safe capital allocation/return
    - Tracks allocations per event
    - Enforces per-event limits

    Usage:
        pool = CapitalPool(CapitalPoolConfig(total_capital=5000))

        # Event starts
        allocated = await pool.request_capital("event_123", max_amount=1000)

        # During event, update value for monitoring
        await pool.update_value("event_123", current_value=1050)

        # Event settles
        await pool.return_capital("event_123", final_value=1100)
    """

    def __init__(self, config: CapitalPoolConfig, max_per_event: float = 500.0):
        """
        Initialize capital pool.

        Args:
            config: Pool configuration
            max_per_event: Maximum capital per event (from KellyConfig.collateral.c_event_max)
        """
        self.config = config
        self.max_per_event = max_per_event
        self._available: float = config.total_capital
        self._allocations: Dict[str, EventAllocation] = {}
        self._lock = asyncio.Lock()
        self._history: list[EventAllocation] = []  # Settled events
        self._initialized_from_api: bool = False
        self._last_negative_warning_time: float = 0.0  # Rate-limit negative warnings

        if config.total_capital > 0:
            logger.info(
                f"[CAPITAL][INIT] baseline_total=${config.total_capital:.2f} "
                f"max_per_event=${max_per_event:.2f}"
            )
        else:
            logger.info(
                "[CAPITAL][INIT] baseline_total=auto_detect "
                "(will fetch from API on startup)"
            )

    @property
    def available_capital(self) -> float:
        """Capital available for new allocations."""
        return self._available

    @property
    def allocated_capital(self) -> float:
        """Total capital currently allocated to events."""
        return sum(a.current_value for a in self._allocations.values())

    @property
    def total_value(self) -> float:
        """Total pool value (available + allocated)."""
        return self._available + self.allocated_capital

    @property
    def num_active_events(self) -> int:
        """Number of events currently trading."""
        return len(self._allocations)

    async def request_capital(
        self,
        event_id: str,
        max_amount: Optional[float] = None,
    ) -> float:
        """
        Request capital for a new event.

        Args:
            event_id: Unique identifier for the event
            max_amount: Maximum amount to allocate (defaults to max_per_event)

        Returns:
            Amount actually allocated (may be less than requested, or 0 if none available)
        """
        async with self._lock:
            # Check if capital pool is initialized
            if self.config.total_capital <= 0 and not self._initialized_from_api:
                logger.warning(
                    "[CAPITAL][WARN] pool not initialized "
                    "(call set_total_from_api() first or configure total_capital)"
                )
                return 0.0

            # Check if event already has allocation (e.g., restored from positions).
            # A restored allocation equals the positions' cost basis, which can
            # be far below max_per_event — top it up from available capital so
            # a restarted event trades with the same budget a fresh one gets.
            if event_id in self._allocations:
                existing = self._allocations[event_id]
                top_up = min(
                    max(0.0, self.max_per_event - existing.current_value),
                    self._available,
                )
                if top_up > 0:
                    existing.current_value += top_up
                    # Keep initial_allocation in step so realized P&L
                    # (final_value - initial_allocation) stays meaningful.
                    existing.initial_allocation += top_up
                    self._available -= top_up
                logger.info(
                    f"[CAPITAL][ALLOCATE] event={event_id} "
                    f"restored_allocation_reused=${existing.current_value:.2f} "
                    f"(top_up=${top_up:.2f})"
                )
                return existing.current_value

            # Determine max allocation
            if max_amount is None:
                max_amount = self.max_per_event
            else:
                max_amount = min(max_amount, self.max_per_event)

            # Determine actual allocation
            allocated = min(max_amount, self._available)

            # Check minimum threshold
            if allocated < self.config.min_allocation:
                logger.warning(
                    f"[CAPITAL][ALLOCATE] event={event_id} "
                    f"requested=${max_amount:.2f} granted=${allocated:.2f} "
                    f"min_required=${self.config.min_allocation:.2f} result=insufficient"
                )
                return 0.0

            # Create allocation
            available_before = self._available
            allocation = EventAllocation(
                event_id=event_id,
                initial_allocation=allocated,
                current_value=allocated,
                allocated_at=datetime.now(timezone.utc),
            )

            self._allocations[event_id] = allocation
            self._available -= allocated

            logger.info(
                f"[CAPITAL][ALLOCATE] event={event_id} "
                f"requested=${max_amount:.2f} granted=${allocated:.2f} "
                f"pool_unallocated_idle=${available_before:.2f}->${self._available:.2f} "
                f"active_events={len(self._allocations)}"
            )

            return allocated

    async def return_capital(
        self,
        event_id: str,
        final_value: float,
    ) -> None:
        """
        Return capital from a settled event.

        Args:
            event_id: Event identifier
            final_value: Final portfolio value at settlement
        """
        async with self._lock:
            if event_id not in self._allocations:
                logger.warning(f"[CAPITAL][WARN] no allocation found for event={event_id}")
                return

            allocation = self._allocations[event_id]
            allocation.settled_at = datetime.now(timezone.utc)
            allocation.final_value = final_value

            # Calculate P&L
            pnl = final_value - allocation.initial_allocation
            pnl_pct = (pnl / allocation.initial_allocation * 100) if allocation.initial_allocation > 0 else 0

            # Return capital to pool
            available_before = self._available
            self._available += final_value

            # Move to history
            self._history.append(allocation)
            del self._allocations[event_id]

            logger.info(
                f"[CAPITAL][RELEASE] event={event_id} "
                f"alloc_budget=${allocation.initial_allocation:.2f} "
                f"returned=${final_value:.2f} realized_pnl=${pnl:+.2f} ({pnl_pct:+.1f}%) "
                f"pool_unallocated_idle=${available_before:.2f}->${self._available:.2f} "
                f"active_events={len(self._allocations)}"
            )

    async def update_value(
        self,
        event_id: str,
        current_value: float,
    ) -> None:
        """
        Update current value for an active event (for monitoring).

        IMPORTANT: Adjusts _available to keep total_value constant.
        This ensures the pool tracks actual capital, not phantom gains.

        Args:
            event_id: Event identifier
            current_value: Current portfolio value (cash + positions mark-to-market)
        """
        async with self._lock:
            if event_id not in self._allocations:
                return

            old_value = self._allocations[event_id].current_value
            self._allocations[event_id].current_value = current_value

            # Adjust _available to keep total constant
            # If value increased, that capital came from available (reduce it)
            # If value decreased, capital returns to available (increase it)
            delta = current_value - old_value
            self._available -= delta

            if self._available < 0:
                now = time.time()
                if now - self._last_negative_warning_time > 60:
                    self._last_negative_warning_time = now
                    logger.warning(
                        f"[CAPITAL][WARN] tracking_pool_available_negative=${self._available:.2f} "
                        f"alloc_current=${self.allocated_capital:.2f} "
                        f"baseline_total=${self.config.total_capital:.2f}"
                    )

    async def get_allocation(self, event_id: str) -> Optional[EventAllocation]:
        """Get allocation info for an event."""
        async with self._lock:
            return self._allocations.get(event_id)

    def get_summary(self) -> dict:
        """Get pool summary for monitoring."""
        return {
            "total_capital": self.config.total_capital,
            "available_capital": self._available,
            "allocated_capital": self.allocated_capital,
            "total_value": self.total_value,
            "num_active_events": len(self._allocations),
            "num_settled_events": len(self._history),
            "active_events": {
                event_id: {
                    "initial": alloc.initial_allocation,
                    "current": alloc.current_value,
                    "pnl": alloc.current_value - alloc.initial_allocation,
                    "allocated_at": alloc.allocated_at.isoformat(),
                }
                for event_id, alloc in self._allocations.items()
            },
            "config": {
                "max_per_event": self.max_per_event,
                "min_allocation": self.config.min_allocation,
            },
        }

    async def restore_allocation(
        self,
        event_id: str,
        current_value: float,
    ) -> None:
        """
        Restore an allocation from computed state (used on restart).

        Unlike request_capital(), this doesn't check limits - it just
        records the allocation as it exists on-chain.

        Args:
            event_id: Event identifier
            current_value: Current portfolio value for this event
        """
        async with self._lock:
            if event_id in self._allocations:
                logger.warning(
                    f"[CAPITAL][RESTORE] event={event_id} already present, updating current value"
                )
                self._allocations[event_id].current_value = current_value
                return

            allocation = EventAllocation(
                event_id=event_id,
                initial_allocation=current_value,  # Unknown, use current as initial
                current_value=current_value,
                allocated_at=datetime.now(timezone.utc),
            )

            self._allocations[event_id] = allocation
            available_before = self._available
            self._available -= current_value

            logger.info(
                f"[CAPITAL][RESTORE] event={event_id} "
                f"restored_basis=${current_value:.2f} "
                f"pool_unallocated_idle=${available_before:.2f}->${self._available:.2f}"
            )

    async def set_total_from_api(self, total_value: float) -> None:
        """
        Set total capital from API query (used on startup/restart).

        Adjusts available capital to match actual on-chain balance.

        Args:
            total_value: Total capital (USDC + position values)
        """
        async with self._lock:
            allocated = sum(a.current_value for a in self._allocations.values())
            self._available = total_value - allocated
            self.config.total_capital = total_value
            self._initialized_from_api = True

            logger.info(
                f"[CAPITAL][SYNC_API] baseline_total=${total_value:.2f} "
                f"alloc_current=${allocated:.2f} pool_unallocated_idle=${self._available:.2f}"
            )

    @property
    def needs_initialization(self) -> bool:
        """Check if capital pool needs to be initialized from API."""
        return self.config.total_capital <= 0 and not self._initialized_from_api

    def get_performance_summary(self) -> dict:
        """Get historical performance summary."""
        if not self._history:
            return {
                "num_events": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "win_rate": 0.0,
            }

        pnls = [
            (a.final_value - a.initial_allocation)
            for a in self._history
            if a.final_value is not None
        ]

        wins = sum(1 for p in pnls if p > 0)

        return {
            "num_events": len(self._history),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "win_rate": wins / len(pnls) if pnls else 0,
            "best_event": max(pnls) if pnls else 0,
            "worst_event": min(pnls) if pnls else 0,
        }
