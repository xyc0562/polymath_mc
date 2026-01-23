# Kelly Trading Algorithm - Implementation Plan

**Document**: Implementation plan for Polymarket Tweet-Count Bins Kelly Trading Strategy
**Date**: 2026-01-23
**Version**: 1.2 (Final - Ready for Implementation)
**Status**: Approved

---

## 1. Executive Summary

This document describes the implementation plan for a Kelly criterion-based trading algorithm for Polymarket's Elon Musk tweet count prediction markets. The algorithm:

- Maximizes expected log-utility (Kelly criterion) across multiple mutually exclusive bin outcomes
- Trades both YES and NO tokens for each bin via Polymarket CLOB API
- Uses WebSocket streaming for real-time orderbook updates
- Implements edge buffers for model uncertainty protection
- Uses adaptive chunk sizing based on liquidity
- Holds positions to settlement after T_stop (3 hours before settlement)

---

## 2. Confirmed Requirements

| Requirement | Decision |
|-------------|----------|
| Initial Capital (C) | Full USDC balance in Polymarket account |
| YES/NO Orderbooks | Same orderbook - BUY YES @ X = SELL NO @ (1-X) |
| Multiple Events | Separate strategies with independent capital pools |
| T_stop | 3 hours before settlement |
| Trading Fees | No explicit fees; spread modeled as configurable buffer |
| Dead Bin NO Tokens | Skip - no arbitrage (market efficient) |
| Order Execution | Via `py_clob_client` (existing) |
| Orderbook Data | WebSocket streaming (new) |
| Model Uncertainty | Edge buffer with dynamic scaling |
| Chunk Sizing | Adaptive based on liquidity and time |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PolymarketTradingBot                            │
│                    (Main orchestrator - existing)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                      ▼
┌─────────────────────┐                            ┌─────────────────────┐
│  Event A Strategy   │                            │  Event B Strategy   │
│  (Independent $)    │                            │  (Independent $)    │
└─────────────────────┘                            └─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          KellyOptimizer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  KellyMath   │  │  Orderbook   │  │  Candidate   │  │   Greedy    │ │
│  │  - W_j       │  │  Analyzer    │  │  Generator   │  │  Executor   │ │
│  │  - S         │  │  - VWAP      │  │  - Edge buf  │  │  - Execute  │ │
│  │  - c*_YES/NO │  │  - Depth     │  │  - Feasible  │  │  - Update   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
         │                              ▲
         │                              │
         ▼                              │
┌─────────────────────┐      ┌─────────────────────────────────────────┐
│   py_clob_client    │      │        WebSocket Client                 │
│   (Order Execution) │      │  wss://ws-subscriptions-clob.polymarket │
│   - create_order    │      │  - Orderbook streaming                  │
│   - post_order      │      │  - Price updates                        │
│   - cancel          │      │  - Dynamic subscriptions                │
└─────────────────────┘      └─────────────────────────────────────────┘
```

---

## 4. Edge Buffer for Model Uncertainty

### 4.1 Problem

Kelly assumes probabilities are accurate. In reality, our model estimates have uncertainty. Without a buffer, we might trade on edges that don't actually exist.

### 4.2 Solution: Dynamic Edge Requirement

Combine a base edge requirement with price-level scaling:

```python
def compute_required_edge(
    reservation_price: float,
    base_edge_pct: float = 0.05,      # 5% base
    extreme_multiplier: float = 2.0,   # Up to 2x at extremes
) -> float:
    """
    Require larger edge at extreme prices where model errors
    have larger relative impact.

    At p=0.50: require base_edge (5%)
    At p=0.10: require ~8% (1.6x)
    At p=0.05: require ~9% (1.8x)
    """
    # Distance from center (0 at p=0.5, 1 at p=0 or p=1)
    extremity = abs(reservation_price - 0.5) * 2

    # Linear scaling: 1x at center, extreme_multiplier at edges
    multiplier = 1.0 + extremity * (extreme_multiplier - 1.0)

    return base_edge_pct * multiplier


def should_trade(
    market_price: float,
    reservation_price: float,
    action: str,  # "BUY" or "SELL"
    config: EdgeBufferConfig,
) -> bool:
    """
    Determine if trade meets edge requirement.
    """
    required_edge = compute_required_edge(
        reservation_price,
        config.base_edge_pct,
        config.extreme_multiplier,
    )

    if action == "BUY":
        # Buy if market_price < reservation * (1 - edge)
        threshold = reservation_price * (1 - required_edge)
        return market_price < threshold
    else:  # SELL
        # Sell if market_price > reservation * (1 + edge)
        threshold = reservation_price * (1 + required_edge)
        return market_price > threshold
```

### 4.3 Example

```
Reservation price: 0.40
Base edge: 5%
Dynamic edge at 0.40: 5% * 1.2 = 6%
Threshold: 0.40 * (1 - 0.06) = 0.376

Market ask: 0.38 → NO TRADE (0.38 > 0.376)
Market ask: 0.37 → TRADE (0.37 < 0.376)
```

---

## 5. Adaptive Chunk Sizing

### 5.1 Problem

Fixed chunk size (delta) doesn't adapt to:
- Thin orderbooks (high slippage)
- Approaching T_stop (need finer control)

### 5.2 Solution

```python
def compute_adaptive_delta(
    base_delta: float,
    available_depth: float,
    hours_to_settlement: float,
    t_stop_hours: float,
    max_depth_fraction: float = 0.10,
    min_delta: float = 1.0,
) -> float:
    """
    Adapt chunk size based on market conditions.
    """
    # 1. Liquidity constraint: never take >10% of visible depth
    liquidity_delta = available_depth * max_depth_fraction

    # 2. Time constraint: ramp down as we approach T_stop
    hours_until_stop = hours_to_settlement - t_stop_hours
    if hours_until_stop <= 0:
        time_delta = min_delta  # At or past T_stop
    elif hours_until_stop >= 6:
        time_delta = base_delta  # Full size
    else:
        # Linear ramp: 100% at 6h, 50% at 0h before T_stop
        time_factor = 0.5 + 0.5 * (hours_until_stop / 6.0)
        time_delta = base_delta * time_factor

    return max(min_delta, min(base_delta, liquidity_delta, time_delta))
```

---

## 6. WebSocket Orderbook Streaming

### 6.1 Polymarket WebSocket API

**Endpoint**: `wss://ws-subscriptions-clob.polymarket.com/ws/market`

**Subscription Message**:
```json
{
  "type": "MARKET",
  "assets_ids": ["token_id_1", "token_id_2", ...],
  "auth": {}
}
```

**Update Message Types**:
- `book` - Full orderbook snapshot
- `price_change` - Price update
- `last_trade_price` - Last trade

### 6.2 Implementation

```python
class OrderbookWebSocket:
    """
    WebSocket client for real-time orderbook streaming.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, token_ids: List[str]):
        self.token_ids = token_ids
        self.orderbooks: Dict[str, OrderBook] = {}
        self._ws = None
        self._running = False

    async def connect(self):
        """Establish WebSocket connection."""
        self._ws = await websockets.connect(self.WS_URL)
        await self._subscribe(self.token_ids)
        self._running = True

    async def _subscribe(self, token_ids: List[str]):
        """Subscribe to orderbook updates."""
        msg = {
            "type": "MARKET",
            "assets_ids": token_ids,
            "auth": {}
        }
        await self._ws.send(json.dumps(msg))

    async def listen(self):
        """Listen for orderbook updates."""
        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                self._handle_message(data)
            except websockets.ConnectionClosed:
                await self._reconnect()

    def _handle_message(self, data: dict):
        """Process orderbook update."""
        if data.get("event_type") == "book":
            token_id = data["asset_id"]
            self.orderbooks[token_id] = self._parse_orderbook(data)

    def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """Get current orderbook for token."""
        return self.orderbooks.get(token_id)

    async def add_subscriptions(self, token_ids: List[str]):
        """Dynamically add new subscriptions."""
        msg = {
            "assets_ids": token_ids,
            "operation": "subscribe"
        }
        await self._ws.send(json.dumps(msg))
        self.token_ids.extend(token_ids)
```

---

## 7. Order Execution via py_clob_client

### 7.1 Existing Integration

The bot already uses `py_clob_client` for order execution:

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

class OrderExecutor:
    """
    Handles order execution via Polymarket CLOB.
    """

    def __init__(self, clob_client: ClobClient):
        self.client = clob_client

    def place_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
    ) -> Optional[dict]:
        """
        Place a limit order.

        For Kelly strategy:
        - BUY YES: side="BUY", token_id=yes_token_id
        - SELL YES: side="SELL", token_id=yes_token_id
        - BUY NO: side="BUY", token_id=no_token_id (or SELL YES)
        - SELL NO: side="SELL", token_id=no_token_id (or BUY YES)
        """
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            order_type=OrderType.GTC,  # Good-til-cancelled
        )

        try:
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)
            return response
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
```

### 7.2 NO Token Execution

Since YES/NO share orderbook, we have two options for NO trades:

**Option A**: Use NO token directly (if py_clob_client supports it)
```python
# BUY NO at price 0.60
place_limit_order(no_token_id, "BUY", 0.60, size)
```

**Option B**: Trade via YES (mathematically equivalent)
```python
# BUY NO at price 0.60 = SELL YES at price 0.40
place_limit_order(yes_token_id, "SELL", 0.40, size)
```

We'll verify which approach works during implementation.

---

## 8. Complete Configuration

```yaml
# config/musk_tweet_count.yaml

kelly:
  # Enable Kelly optimizer
  enabled: true

  # Fractional Kelly (0.25 = quarter Kelly)
  kappa: 0.25

  # Minimum utility gain threshold
  tau: 0.001

  # Minimum terminal wealth floor ($)
  w_floor: 1.0

  # Max iterations per tick
  max_iters_per_tick: 100

  # Trading cutoff (hours before settlement)
  t_stop_hours: 3.0

  # Renormalize probabilities after dead-bin removal
  renormalize_probabilities: true

  # Edge buffer for model uncertainty
  edge_buffer:
    base_edge_pct: 0.05      # 5% base edge required
    extreme_multiplier: 2.0  # Up to 2x at extreme prices

  # Adaptive chunk sizing
  adaptive_delta:
    base_delta: 10.0         # Base chunk size (shares)
    max_depth_fraction: 0.10 # Never take >10% of depth
    time_ramp_hours: 6.0     # Ramp down over 6h before T_stop
    min_delta: 1.0           # Minimum 1 share

  # WebSocket settings
  websocket:
    enabled: true
    reconnect_delay_seconds: 5
    heartbeat_interval_seconds: 30

collateral:
  # Maximum collateral per event (USD)
  c_event_max: 500.0

  # Maximum collateral per bin (USD)
  c_bin_max: 100.0
```

---

## 9. File Structure

```
src/algo/musk_tweet_count/
├── __init__.py
├── musk_tweet_count.py          # Existing main bot (modified)
├── abi/
│   └── polymarket.js            # Existing ABI
└── kelly/                       # NEW: Kelly optimizer package
    ├── __init__.py
    ├── kelly_math.py            # W_j, S, reservation prices
    ├── orderbook.py             # Unified orderbook, VWAP
    ├── websocket_client.py      # WebSocket streaming
    ├── portfolio.py             # Portfolio state management
    ├── candidates.py            # Trade candidate generation
    ├── executor.py              # Greedy execution loop
    └── config.py                # Kelly configuration
```

---

## 10. Implementation Phases

### Phase 1: Core Kelly Math
- `kelly_math.py`: W_j, S, reservation prices
- Unit tests for correctness

### Phase 2: Orderbook Handling
- `orderbook.py`: Unified YES/NO, VWAP calculation
- `websocket_client.py`: Real-time streaming

### Phase 3: Portfolio & Candidates
- `portfolio.py`: State tracking, collateral
- `candidates.py`: Generation with edge buffer

### Phase 4: Executor & Integration
- `executor.py`: Greedy loop with adaptive delta
- Integration with main bot
- Configuration loading

### Phase 5: Testing
- Dry-run against live markets
- Edge case testing

---

## 11. Sources

- [Polymarket WebSocket Overview](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [Polymarket Python Client](https://github.com/Polymarket/py-clob-client)
- [Polymarket Real-Time Data Client](https://github.com/Polymarket/real-time-data-client)
- [polymarket-apis on PyPI](https://pypi.org/project/polymarket-apis/)

---

*Document finalized 2026-01-23. Ready for implementation.*
