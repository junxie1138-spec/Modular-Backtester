from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    dispersion_ratio: float = 0.35
    minimum_dispersion: float = 0.003
    range_floor: float = 0.80
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791310013"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["normalized_range"] = (out["true_range"] / previous_close).replace([np.inf, -np.inf], np.nan)
        out["range_mean"] = out["normalized_range"].rolling(3, min_periods=3).mean()
        out["range_dispersion"] = out["normalized_range"].rolling(3, min_periods=3).std(ddof=0)
        out["baseline_mean"] = out["range_mean"].shift(3)
        out["baseline_dispersion"] = out["range_dispersion"].shift(3)
        out["baseline_return"] = (data["close"].shift(3) / data["close"].shift(6).where(lambda x: x > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["stabilized"] = (
            (out["baseline_dispersion"] >= params.minimum_dispersion)
            & (out["range_dispersion"] <= params.dispersion_ratio * out["baseline_dispersion"])
            & (out["range_mean"] >= params.range_floor * out["baseline_mean"])
            & (out["baseline_return"] < 0)
        ).fillna(False)
        out["entry"] = (
            out["stabilized"]
            & ~out["stabilized"].shift(1, fill_value=False)
            & (data["close"] > previous_close)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if not (0 < params.dispersion_ratio < 1 and params.minimum_dispersion > 0 and params.range_floor > 0 and params.profit_target > 0 and 3 <= params.max_hold <= 5):
            raise ValueError("Invalid stabilization thresholds or holding horizon")
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The pending entry fills at this bar's open.
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].iloc[:7].any()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
