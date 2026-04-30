"""Offline backtest engine for the lead-lag algo.

Shares the production trading core: `run_strategy_tick`, `PositionManager`,
`signal_comparator`, `position_manager.compute_targets/compute_deltas` are
exercised byte-identically. The backtest only swaps the data source
(`SqliteMarketDataProvider`) and the order gateway (`SimulatedOrderGateway`).
"""
