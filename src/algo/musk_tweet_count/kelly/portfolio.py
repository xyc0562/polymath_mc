"""
Portfolio state management for Kelly trading.

Tracks:
- Current capital (USDC balance)
- YES/NO positions for each bin
- Collateral usage
- Position history
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import time

from .kelly_math import (
    compute_all_terminal_wealths,
    compute_normalizer_S,
    compute_all_reservation_prices,
    compute_expected_log_utility,
)

logger = logging.getLogger(__name__)


@dataclass
class BinPosition:
    """Position in a single bin."""

    bin_index: int
    yes_token_id: str
    no_token_id: Optional[str] = None  # May not always have separate NO token

    # Current positions (shares)
    yes_shares: float = 0.0
    no_shares: float = 0.0

    # Average cost basis (for P&L tracking)
    yes_avg_cost: float = 0.0
    no_avg_cost: float = 0.0

    # Shares whose acquisition cost is not yet priceable from API/local fills.
    yes_unpriced_shares: float = 0.0
    no_unpriced_shares: float = 0.0

    # Conservative collateral reserve for unresolved shares.
    yes_unpriced_reserve: float = 0.0
    no_unpriced_reserve: float = 0.0

    # Wall-clock timestamp when unpriced shares first appeared (0 = none).
    yes_unpriced_since: float = 0.0
    no_unpriced_since: float = 0.0

    # Position side is missing from positions API but not yet verified closed.
    yes_api_missing_unverified: bool = False
    no_api_missing_unverified: bool = False
    yes_api_missing_since: float = 0.0
    no_api_missing_since: float = 0.0
    yes_api_missing_count: int = 0
    no_api_missing_count: int = 0

    # Collateral tied up in positions
    collateral_used: float = 0.0

    # Realized P&L from closed trades (sells)
    realized_pnl: float = 0.0

    @property
    def has_yes_position(self) -> bool:
        """Check if we hold YES shares."""
        return self.yes_shares > 0.01  # Small threshold for floating point

    @property
    def has_no_position(self) -> bool:
        """Check if we hold NO shares."""
        return self.no_shares > 0.01

    @property
    def total_investment(self) -> float:
        """Total invested in this bin."""
        return (
            (self.priced_yes_shares * self.yes_avg_cost)
            + (self.priced_no_shares * self.no_avg_cost)
            + self.yes_unpriced_reserve
            + self.no_unpriced_reserve
        )

    @property
    def priced_yes_shares(self) -> float:
        """YES shares with resolved cost basis."""
        return max(0.0, self.yes_shares - self.yes_unpriced_shares)

    @property
    def priced_no_shares(self) -> float:
        """NO shares with resolved cost basis."""
        return max(0.0, self.no_shares - self.no_unpriced_shares)

    @property
    def has_yes_unpriced_increment(self) -> bool:
        """Whether this bin has unresolved YES shares."""
        return self.yes_unpriced_shares > 0.01

    @property
    def has_no_unpriced_increment(self) -> bool:
        """Whether this bin has unresolved NO shares."""
        return self.no_unpriced_shares > 0.01

    @property
    def has_yes_api_missing_unverified(self) -> bool:
        """Whether YES is temporarily retained despite missing from positions API."""
        return self.yes_api_missing_unverified

    @property
    def has_no_api_missing_unverified(self) -> bool:
        """Whether NO is temporarily retained despite missing from positions API."""
        return self.no_api_missing_unverified

    def recompute_collateral_used(self) -> None:
        """Recompute collateral from priced cost basis plus unresolved reserve."""
        self.collateral_used = (
            self.priced_yes_shares * self.yes_avg_cost
            + self.priced_no_shares * self.no_avg_cost
            + self.yes_unpriced_reserve
            + self.no_unpriced_reserve
        )

    def _consume_unpriced_yes(self, shares: float) -> tuple[float, float]:
        """Consume unresolved YES shares first and return (shares, reserve)."""
        if shares <= 0 or self.yes_unpriced_shares <= 0:
            return 0.0, 0.0

        take = min(shares, self.yes_unpriced_shares)
        reserve_before = self.yes_unpriced_reserve
        unpriced_before = self.yes_unpriced_shares
        reserve_released = reserve_before * (take / unpriced_before) if unpriced_before > 0 else 0.0
        self.yes_unpriced_shares = max(0.0, self.yes_unpriced_shares - take)
        self.yes_unpriced_reserve = max(0.0, self.yes_unpriced_reserve - reserve_released)
        return take, reserve_released

    def _consume_unpriced_no(self, shares: float) -> tuple[float, float]:
        """Consume unresolved NO shares first and return (shares, reserve)."""
        if shares <= 0 or self.no_unpriced_shares <= 0:
            return 0.0, 0.0

        take = min(shares, self.no_unpriced_shares)
        reserve_before = self.no_unpriced_reserve
        unpriced_before = self.no_unpriced_shares
        reserve_released = reserve_before * (take / unpriced_before) if unpriced_before > 0 else 0.0
        self.no_unpriced_shares = max(0.0, self.no_unpriced_shares - take)
        self.no_unpriced_reserve = max(0.0, self.no_unpriced_reserve - reserve_released)
        return take, reserve_released

    def add_yes(self, shares: float, price: float) -> None:
        """Add YES shares at a price."""
        if shares <= 0:
            return

        # Update average cost basis
        priced_shares = self.priced_yes_shares
        total_priced_shares = priced_shares + shares
        if total_priced_shares > 0:
            self.yes_avg_cost = (
                (priced_shares * self.yes_avg_cost) + (shares * price)
            ) / total_priced_shares

        self.yes_shares += shares
        self.recompute_collateral_used()

    def remove_yes(self, shares: float, price: float) -> float:
        """
        Remove YES shares (sell).

        Returns realized P&L.
        """
        shares = min(shares, self.yes_shares)
        if shares <= 0:
            return 0.0

        # Calculate P&L
        unresolved_removed, unresolved_released = self._consume_unpriced_yes(shares)
        priced_shares = max(0.0, shares - unresolved_removed)
        cost_basis = unresolved_released + (priced_shares * self.yes_avg_cost)
        proceeds = shares * price
        pnl = proceeds - cost_basis

        self.yes_shares = max(0.0, self.yes_shares - shares)
        if self.priced_yes_shares <= 0.01:
            self.yes_avg_cost = 0.0
        if self.yes_shares <= 0.01:
            self.yes_shares = 0.0
            self.yes_unpriced_shares = 0.0
            self.yes_unpriced_reserve = 0.0
            self.yes_avg_cost = 0.0
            self.yes_api_missing_unverified = False
            self.yes_api_missing_since = 0.0
            self.yes_api_missing_count = 0
        self.recompute_collateral_used()
        self.realized_pnl += pnl

        return pnl

    def add_no(self, shares: float, price: float) -> None:
        """Add NO shares at a price."""
        if shares <= 0:
            return

        priced_shares = self.priced_no_shares
        total_priced_shares = priced_shares + shares
        if total_priced_shares > 0:
            self.no_avg_cost = (
                (priced_shares * self.no_avg_cost) + (shares * price)
            ) / total_priced_shares

        self.no_shares += shares
        self.recompute_collateral_used()

    def remove_no(self, shares: float, price: float) -> float:
        """
        Remove NO shares (sell).

        Returns realized P&L.
        """
        shares = min(shares, self.no_shares)
        if shares <= 0:
            return 0.0

        unresolved_removed, unresolved_released = self._consume_unpriced_no(shares)
        priced_shares = max(0.0, shares - unresolved_removed)
        cost_basis = unresolved_released + (priced_shares * self.no_avg_cost)
        proceeds = shares * price
        pnl = proceeds - cost_basis

        self.no_shares = max(0.0, self.no_shares - shares)
        if self.priced_no_shares <= 0.01:
            self.no_avg_cost = 0.0
        if self.no_shares <= 0.01:
            self.no_shares = 0.0
            self.no_unpriced_shares = 0.0
            self.no_unpriced_reserve = 0.0
            self.no_avg_cost = 0.0
            self.no_api_missing_unverified = False
            self.no_api_missing_since = 0.0
            self.no_api_missing_count = 0
        self.recompute_collateral_used()
        self.realized_pnl += pnl

        return pnl


@dataclass
class Portfolio:
    """
    Complete portfolio state for Kelly trading.

    Manages capital, positions across all bins, and Kelly calculations.
    """

    # Initial and current capital
    initial_capital: float
    capital: float  # Current USDC balance

    # Bin positions
    positions: Dict[int, BinPosition] = field(default_factory=dict)

    # Probability estimates for each bin
    probabilities: List[float] = field(default_factory=list)

    # Bin metadata
    num_bins: int = 0
    bin_upper_bounds: List[int] = field(default_factory=list)
    dead_bins: List[int] = field(default_factory=list)

    # Trading state
    total_realized_pnl: float = 0.0
    last_update_time: float = field(default_factory=time.time)

    # Phantom capital for Kelly utility calculation only.
    # Inflates Kelly's perceived wealth so it sizes more aggressively.
    # Does NOT affect available_capital (real USDC only).
    phantom_capital: float = 0.0

    # External capital constraint (from shared pool manager)
    # When set, available_capital is min(internal_available, external_limit)
    external_capital_limit: Optional[float] = None

    @property
    def total_collateral_used(self) -> float:
        """Total collateral across all positions."""
        return sum(pos.collateral_used for pos in self.positions.values())

    @property
    def available_capital(self) -> float:
        """
        Capital available for new trades.

        NOTE: capital = liquid USDC (already decreased when buying).
        Do NOT subtract collateral_used - that would double-count.
        collateral_used is only for per-bin limit tracking (c_bin_max).

        If external_capital_limit is set (from shared pool manager),
        returns the minimum of capital and external limit.
        """
        # capital IS the available USDC - don't subtract collateral
        internal_available = self.capital

        if self.external_capital_limit is not None:
            return min(internal_available, self.external_capital_limit)

        return internal_available

    def set_external_capital_limit(self, limit: Optional[float]) -> None:
        """
        Set external capital constraint from shared pool.

        Args:
            limit: Maximum additional capital this portfolio can use,
                   or None to remove constraint.
        """
        self.external_capital_limit = limit

    @property
    def yes_positions(self) -> Dict[int, float]:
        """Map of bin_index -> YES shares."""
        return {idx: pos.yes_shares for idx, pos in self.positions.items()}

    @property
    def no_positions(self) -> Dict[int, float]:
        """Map of bin_index -> NO shares."""
        return {idx: pos.no_shares for idx, pos in self.positions.items()}

    def get_position(self, bin_index: int) -> Optional[BinPosition]:
        """Get position for a specific bin."""
        return self.positions.get(bin_index)

    def ensure_position(self, bin_index: int, yes_token_id: str) -> BinPosition:
        """Get or create position for a bin."""
        if bin_index not in self.positions:
            self.positions[bin_index] = BinPosition(
                bin_index=bin_index,
                yes_token_id=yes_token_id,
            )
        return self.positions[bin_index]

    def update_probabilities(
        self,
        probabilities: List[float],
        renormalize: bool = True,
    ) -> None:
        """
        Update probability estimates.

        Args:
            probabilities: New probability distribution
            renormalize: If True, renormalize after zeroing dead bins
        """
        self.probabilities = probabilities.copy()

        if renormalize and self.dead_bins:
            # Zero out dead bins
            for i in self.dead_bins:
                if i < len(self.probabilities):
                    self.probabilities[i] = 0.0

            # Renormalize
            total = sum(self.probabilities)
            if total > 0:
                self.probabilities = [p / total for p in self.probabilities]

    def get_terminal_wealths(self, w_floor: float = 1.0) -> List[float]:
        """
        Compute terminal wealth for each possible winning bin.

        Uses current capital (+ phantom capital for Kelly utility) and positions.
        Phantom capital inflates Kelly's perceived wealth so it sizes more
        aggressively, but real capital still gates execution.
        """
        return compute_all_terminal_wealths(
            capital=self.capital + self.phantom_capital,
            yes_positions=self.yes_positions,
            no_positions=self.no_positions,
            num_bins=self.num_bins,
        )

    def get_reservation_prices(
        self,
        w_floor: float = 1.0,
        kelly_fraction: float = 1.0,
    ) -> tuple[List[float], List[float]]:
        """
        Compute Kelly reservation prices for all bins.

        Returns:
            Tuple of (yes_prices, no_prices)
        """
        terminal_wealths = self.get_terminal_wealths(w_floor)
        return compute_all_reservation_prices(
            self.probabilities,
            terminal_wealths,
            w_floor,
            kelly_fraction,
        )

    def get_expected_utility(self, w_floor: float = 1.0, kelly_fraction: float = 1.0) -> float:
        """Compute current expected utility."""
        terminal_wealths = self.get_terminal_wealths(w_floor)
        return compute_expected_log_utility(
            self.probabilities,
            terminal_wealths,
            w_floor,
            kelly_fraction,
        )

    def simulate_buy_yes(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> "Portfolio":
        """
        Simulate buying YES shares.

        Returns a new Portfolio with the simulated trade.
        """
        # Create copy
        new_portfolio = self._copy()

        # Get or create position
        pos = new_portfolio.positions.get(bin_index)
        if not pos:
            pos = BinPosition(bin_index=bin_index, yes_token_id="")
            new_portfolio.positions[bin_index] = pos

        # Add shares
        cost = shares * price
        pos.add_yes(shares, price)
        new_portfolio.capital -= cost

        return new_portfolio

    def simulate_sell_yes(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> "Portfolio":
        """Simulate selling YES shares."""
        new_portfolio = self._copy()

        pos = new_portfolio.positions.get(bin_index)
        if not pos or pos.yes_shares < shares:
            return new_portfolio  # Can't sell more than we have

        proceeds = shares * price
        pnl = pos.remove_yes(shares, price)
        new_portfolio.capital += proceeds
        new_portfolio.total_realized_pnl += pnl

        return new_portfolio

    def simulate_buy_no(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> "Portfolio":
        """Simulate buying NO shares."""
        new_portfolio = self._copy()

        pos = new_portfolio.positions.get(bin_index)
        if not pos:
            pos = BinPosition(bin_index=bin_index, yes_token_id="")
            new_portfolio.positions[bin_index] = pos

        cost = shares * price
        pos.add_no(shares, price)
        new_portfolio.capital -= cost

        return new_portfolio

    def simulate_sell_no(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> "Portfolio":
        """Simulate selling NO shares."""
        new_portfolio = self._copy()

        pos = new_portfolio.positions.get(bin_index)
        if not pos or pos.no_shares < shares:
            return new_portfolio

        proceeds = shares * price
        pnl = pos.remove_no(shares, price)
        new_portfolio.capital += proceeds
        new_portfolio.total_realized_pnl += pnl

        return new_portfolio

    def _copy(self) -> "Portfolio":
        """Create a deep copy of the portfolio."""
        new_positions = {}
        for idx, pos in self.positions.items():
            new_positions[idx] = BinPosition(
                bin_index=pos.bin_index,
                yes_token_id=pos.yes_token_id,
                no_token_id=pos.no_token_id,
                yes_shares=pos.yes_shares,
                no_shares=pos.no_shares,
                yes_avg_cost=pos.yes_avg_cost,
                no_avg_cost=pos.no_avg_cost,
                yes_unpriced_shares=pos.yes_unpriced_shares,
                no_unpriced_shares=pos.no_unpriced_shares,
                yes_unpriced_reserve=pos.yes_unpriced_reserve,
                no_unpriced_reserve=pos.no_unpriced_reserve,
                yes_unpriced_since=pos.yes_unpriced_since,
                no_unpriced_since=pos.no_unpriced_since,
                yes_api_missing_unverified=pos.yes_api_missing_unverified,
                no_api_missing_unverified=pos.no_api_missing_unverified,
                yes_api_missing_since=pos.yes_api_missing_since,
                no_api_missing_since=pos.no_api_missing_since,
                yes_api_missing_count=pos.yes_api_missing_count,
                no_api_missing_count=pos.no_api_missing_count,
                collateral_used=pos.collateral_used,
                realized_pnl=pos.realized_pnl,
            )

        return Portfolio(
            initial_capital=self.initial_capital,
            capital=self.capital,
            positions=new_positions,
            probabilities=self.probabilities.copy(),
            num_bins=self.num_bins,
            bin_upper_bounds=self.bin_upper_bounds.copy(),
            dead_bins=self.dead_bins.copy(),
            total_realized_pnl=self.total_realized_pnl,
            last_update_time=self.last_update_time,
            phantom_capital=self.phantom_capital,
            external_capital_limit=self.external_capital_limit,
        )

    def execute_buy_yes(
        self,
        bin_index: int,
        shares: float,
        price: float,
        yes_token_id: str,
    ) -> None:
        """Actually execute a YES buy (update real portfolio)."""
        pos = self.ensure_position(bin_index, yes_token_id)
        cost = shares * price
        pos.add_yes(shares, price)
        self.capital -= cost
        self.last_update_time = time.time()

    def execute_sell_yes(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> float:
        """Actually execute a YES sell. Returns P&L."""
        pos = self.positions.get(bin_index)
        if not pos:
            return 0.0

        proceeds = shares * price
        pnl = pos.remove_yes(shares, price)
        self.capital += proceeds
        self.total_realized_pnl += pnl
        self.last_update_time = time.time()
        return pnl

    def execute_buy_no(
        self,
        bin_index: int,
        shares: float,
        price: float,
        yes_token_id: str,
    ) -> None:
        """Actually execute a NO buy."""
        pos = self.ensure_position(bin_index, yes_token_id)
        cost = shares * price
        pos.add_no(shares, price)
        self.capital -= cost
        self.last_update_time = time.time()

    def execute_sell_no(
        self,
        bin_index: int,
        shares: float,
        price: float,
    ) -> float:
        """Actually execute a NO sell. Returns P&L."""
        pos = self.positions.get(bin_index)
        if not pos:
            return 0.0

        proceeds = shares * price
        pnl = pos.remove_no(shares, price)
        self.capital += proceeds
        self.total_realized_pnl += pnl
        self.last_update_time = time.time()
        return pnl

    def to_summary(self) -> dict:
        """Get summary of portfolio state."""
        return {
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "available_capital": self.available_capital,
            "total_collateral": self.total_collateral_used,
            "realized_pnl": self.total_realized_pnl,
            "num_positions": len([p for p in self.positions.values()
                                  if p.has_yes_position or p.has_no_position]),
            "expected_utility": self.get_expected_utility(),
        }
