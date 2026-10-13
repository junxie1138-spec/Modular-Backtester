from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DispersionDriftParams:
    drift_window: int = 3
    risk_window: int = 20
    dispersion_ceiling: float = 0.65
    drift_floor: float = 0.50
    target_daily_vol: float = 0.008
    profit_target: float = 0.012
    holding_bars: int = 4
    min_size: float = 0.05
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DispersionDriftParams]):
    strategy_id = "gen_local_1791228578"

    @classmethod
    def params_type(cls) -> type[DispersionDriftParams]:
        return DispersionDriftParams

    def warmup_bars(self, params: DispersionDriftParams) -> int:
        return int(max(params.drift_window, params.risk_window) + 2)

    def indicators(self, data: pd.DataFrame, params: DispersionDriftParams) -> pd.DataFrame:
        if min(params.drift_window, params.risk_window) < 2:
            raise ValueError("Return windows must be at least two bars")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Size bounds must satisfy 0 < min_size <= max_size <= 1")
        if min(params.target_daily_vol, params.profit_target, params.dispersion_ceiling, params.drift_floor) <= 0:
            raise ValueError("Volatility, profit, and entry thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["risk"] = out["return"].pow(2).rolling(params.risk_window).mean().pow(0.5)
        out["drift"] = out["return"].rolling(params.drift_window).mean()
        out["dispersion"] = out["return"].rolling(params.drift_window).std(ddof=0)
        denominator = out["risk"].clip(lower=0.0001)
        out["relative_dispersion"] = out["dispersion"] / denominator
        out["relative_drift"] = out["drift"] / denominator
        valid = out[["risk", "relative_dispersion", "relative_drift", "return"]].notna().all(axis=1)
        compressed = out["relative_dispersion"].lt(params.dispersion_ceiling)
        long_setup = valid & compressed & out["relative_drift"].gt(params.drift_floor) & out["return"].gt(0)
        short_setup = valid & compressed & out["relative_drift"].lt(-params.drift_floor) & out["return"].lt(0)
        out["candidate"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        out["exposure"] = (params.target_daily_vol / denominator).clip(params.min_size, params.max_size).fillna(params.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DispersionDriftParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        pending_entry = False
        for i in range(len(data)):
            if pending_entry:
                entry_price = opens[i]
                pending_entry = False
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
            if position:
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or age >= params.holding_bars:
                    position = 0
                    age = 0
                    entry_price = np.nan
            elif confirmed[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(confirmed[i])
                pending_entry = True
                age = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = indicators["exposure"].shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = 100.0 * np.exp(np.arange(60) * 0.002)
    sample = pd.DataFrame({"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=60))
    strategy = GeneratedStrategy()
    params = DispersionDriftParams()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params).data
    assert result["signal"].iloc[:22].eq(0).all()
    assert result["signal"].iloc[22] == 1 and result["signal"].iloc[26] == 0
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    quick_profit = DispersionDriftParams(profit_target=0.001)
    quick = strategy.generate_signals(sample, strategy.indicators(sample, quick_profit), None, quick_profit).data
    assert quick["signal"].iloc[22] == 1 and quick["signal"].iloc[23] == 0
