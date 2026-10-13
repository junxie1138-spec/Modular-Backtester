from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    acceleration_threshold: float = 0.12
    stop_atr: float = 1.75


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791335743"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        returns = close.pct_change(fill_method=None)
        path_length = returns.abs().rolling(20, min_periods=20).sum()
        net_return = close.pct_change(20, fill_method=None)
        roc = close.pct_change(3, fill_method=None)
        acceleration = roc.diff() / (atr / close.replace(0.0, np.nan)).replace(0.0, np.nan)
        result = pd.DataFrame({
            "atr": atr,
            "roc": roc,
            "acceleration": acceleration,
            "efficiency": net_return.abs() / path_length.replace(0.0, np.nan),
            "net_return": net_return,
            "mean_close": close.rolling(10, min_periods=10).mean(),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        acceleration = indicators["acceleration"]
        turning = (
            (acceleration > params.acceleration_threshold)
            & (acceleration.shift(1) <= params.acceleration_threshold)
        )
        plastic = (
            (indicators["efficiency"] > 0.35)
            & (indicators["net_return"] > 0)
            & (indicators["roc"] > 0)
        )
        elastic = (
            (indicators["efficiency"] <= 0.35)
            & (indicators["net_return"] >= -0.02)
            & (indicators["roc"] < 0)
            & (data["close"] < indicators["mean_close"])
        )
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0)
        entries = (ready & turning & (plastic | elastic)).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw_signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = 0.0
        fixed_stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive")
                fixed_stop = opens[i] - params.stop_atr * entry_atr
                active = True
                pending = False
                age = 0
            if active:
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    raise ValueError("Held position close must be finite and positive")
                age += 1
                if closes[i] < fixed_stop or age >= 4:
                    active = False
                else:
                    raw_signal[i] = 1
            elif entries[i]:
                entry_atr = atr[i]
                pending = True
                raw_signal[i] = 1
        df = pd.DataFrame({"signal": raw_signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    n = 160
    t = np.arange(n, dtype=float)
    close = 100.0 + 0.04 * t + 2.0 * np.sin(t / 3.0)
    prices = pd.DataFrame({
        "open": close + 0.1,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "volume": np.full(n, 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    strategy = GeneratedStrategy()
    params = StrategyParams()
    output = strategy.generate_signals(prices, strategy.indicators(prices, params), None, params).data
    assert output["signal"].isin([0, 1]).all() and output["signal"].iloc[0] == 0
    assert output["signal"].eq(1).any() and output["size"].gt(0).all()
    altered = prices.copy()
    altered.iloc[100:, :4] *= 1.5
    changed = strategy.generate_signals(altered, strategy.indicators(altered, params), None, params).data
    pd.testing.assert_frame_equal(output.iloc[:100], changed.iloc[:100])
