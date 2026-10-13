from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    volume_floor: float = 1.1
    efficiency_gain: float = 1.15
    annual_vol_target: float = 0.12
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791312427"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.lookback < 2 or not 3 <= params.max_hold <= 5:
            raise ValueError("lookback must be >= 2 and max_hold must be 3 through 5")
        if not all(np.isfinite(x) and x > 0 for x in (
            params.volume_floor, params.efficiency_gain,
            params.annual_vol_target, params.profit_target,
        )):
            raise ValueError("thresholds and volatility target must be finite and positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        volume = data["volume"].where(np.isfinite(data["volume"]) & data["volume"].gt(0))
        out["return"] = close.pct_change(fill_method=None)
        out["volume_baseline"] = volume.rolling(params.lookback, min_periods=params.lookback).median().shift(1)
        out["relative_volume"] = volume / out["volume_baseline"].where(out["volume_baseline"].gt(0))
        out["demand_efficiency"] = out["return"] / out["relative_volume"].where(out["relative_volume"].gt(0))
        spread = (data["high"] - data["low"]).where(data["high"].gt(data["low"]))
        out["close_location"] = (close - data["low"]) / spread
        out["annual_vol"] = out["return"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0) * np.sqrt(252.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["qualified"] = (
            out["return"].gt(0)
            & close.gt(data["open"])
            & out["relative_volume"].ge(params.volume_floor)
            & out["close_location"].ge(0.65)
        ).fillna(False)
        out["confirmed"] = (
            out["qualified"]
            & out["qualified"].shift(1, fill_value=False)
            & out["relative_volume"].lt(out["relative_volume"].shift(1))
            & out["demand_efficiency"].ge(params.efficiency_gain * out["demand_efficiency"].shift(1))
            & out["annual_vol"].gt(0)
        ).fillna(False)
        out["allocation"] = (
            params.annual_vol_target / out["annual_vol"].where(out["annual_vol"].gt(0))
        ).clip(lower=0.05, upper=1.0).fillna(0.05)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        allocations = indicators["allocation"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        trade_size = 0.05

        for i in range(n):
            if entry_bar >= 0:
                # Read the entry open only when that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_hit = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_hit:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
                    sizes[i] = trade_size
            elif i >= self.warmup_bars(params) - 1 and confirmed[i]:
                trade_size = float(allocations[i])
                if np.isfinite(trade_size) and trade_size > 0:
                    entry_bar = i + 1
                    signals[i] = 1
                    sizes[i] = trade_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(lookback=2)
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": [100.0, 100.0, 100.0, 100.0, 100.0, 102.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"confirmed": False, "allocation": 0.4}, index=index)
    ind.loc[index[3], "confirmed"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert result.loc[index[4], "size"] == 0.4
    params.profit_target = 1.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    flat = data.assign(close=100.0)
    derived = strategy.indicators(flat, params)
    assert not derived["confirmed"].any()
    assert np.isfinite(derived["allocation"]).all()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()
    prefix = strategy.generate_signals(data.iloc[:7], ind.iloc[:7], ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:7], prefix)


if __name__ == "__main__":
    _self_check()
