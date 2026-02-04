"""
Capital pool manager for multi-event trading.

Manages a shared capital pool that multiple concurrent events can draw from.
Each event requests capital on start and returns it at settlement.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
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

        if config.total_capital > 0:
            logger.info(
                f"Capital pool initialized: total=${config.total_capital:.2f}, "
                f"max_per_event=${max_per_event:.2f}"
            )
        else:
            logger.info(
                "Capital pool initialized with auto-detect mode "
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
                    "Capital pool not initialized (total_capital=0). "
                    "Call set_total_from_api() first or set total_capital in config."
                )
                return 0.0

            # Check if event already has allocation (e.g., restored from positions)
            if event_id in self._allocations:
                existing = self._allocations[event_id]
                # Return existing allocation value so the event can start
                logger.info(
                    f"Event {event_id} has restored allocation: ${existing.current_value:.2f}"
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
                    f"Insufficient capital for event {event_id}: "
                    f"${allocated:.2f} < min ${self.config.min_allocation:.2f}"
                )
                return 0.0

            # Create allocation
            allocation = EventAllocation(
                event_id=event_id,
                initial_allocation=allocated,
                current_value=allocated,
                allocated_at=datetime.utcnow(),
            )

            self._allocations[event_id] = allocation
            self._available -= allocated

            logger.info(
                f"Allocated ${allocated:.2f} to event {event_id} "
                f"(pool: ${self._available:.2f} available, "
                f"{len(self._allocations)} active events)"
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
                logger.warning(f"No allocation found for event {event_id}")
                return

            allocation = self._allocations[event_id]
            allocation.settled_at = datetime.utcnow()
            allocation.final_value = final_value

            # Calculate P&L
            pnl = final_value - allocation.initial_allocation
            pnl_pct = (pnl / allocation.initial_allocation * 100) if allocation.initial_allocation > 0 else 0

            # Return capital to pool
            self._available += final_value

            # Move to history
            self._history.append(allocation)
            del self._allocations[event_id]

            logger.info(
                f"Event {event_id} deallocated: ${final_value:.2f} returned to pool "
                f"(P&L: ${pnl:+.2f}, {pnl_pct:+.1f}%) "
                f"(pool: ${self._available:.2f} available, "
                f"{len(self._allocations)} active events)"
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
                logger.warning(
                    f"Capital pool available went negative: ${self._available:.2f} "
                    f"(event {event_id} value changed by ${delta:+.2f})"
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
                logger.warning(f"Event {event_id} already has allocation, updating value")
                self._allocations[event_id].current_value = current_value
                return

            allocation = EventAllocation(
                event_id=event_id,
                initial_allocation=current_value,  # Unknown, use current as initial
                current_value=current_value,
                allocated_at=datetime.utcnow(),
            )

            self._allocations[event_id] = allocation
            self._available -= current_value

            logger.info(
                f"Restored allocation for event {event_id}: ${current_value:.2f} "
                f"(pool: ${self._available:.2f} available)"
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
                f"Capital pool synced from API: ${total_value:.2f} total, "
                f"${allocated:.2f} allocated, ${self._available:.2f} available"
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
