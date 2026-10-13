from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    ma_bars: int = 20
    flow_bars: int = 3
    entry_z: float = 1.2
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791220938"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.ma_bars, params.flow_bars) + 1

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        if params.ma_bars < 2 or params.flow_bars < 1:
            raise ValueError("Lookbacks must be positive; ma_bars must exceed one.")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("entry_z must be positive and hold_bars must be 3, 4, or 5.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        prior = close.shift(1)
        out["ma"] = prior.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["price_std"] = prior.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].where(out["price_std"] > 0)
        out["overnight"] = np.log(opening / prior)
        out["intraday"] = np.log(close / opening)
        out["overnight_sum"] = out["overnight"].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out["intraday_sum"] = out["intraday"].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AbsorptionParams) -> SignalFrame:
        valid = np.isfinite(indicators[["z", "overnight_sum", "intraday_sum"]]).all(axis=1)
        eligible = (valid & (indicators["z"] <= -params.entry_z)
                    & (indicators["intraday_sum"] > 0)
                    & (indicators["overnight_sum"] < -2.0 * indicators["intraday_sum"]))
        conviction = (-indicators["z"] / (2.0 * params.entry_z)).clip(0.25, 0.95).fillna(0.25)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        exit_bar = -1
        entry_size = 0.25
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif eligible.iloc[i]:
                exit_bar = i + params.hold_bars
                entry_size = float(conviction.iloc[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AbsorptionParams()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame({name: np.full(40, 100.0)
                         for name in ("open", "high", "low", "close", "volume")}, index=index)
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert flat["signal"].eq(0).all()
    forced = pd.DataFrame({"z": -1.5, "overnight_sum": -0.03,
                           "intraday_sum": 0.01}, index=index)
    forced.iloc[2, 0] = -4.0
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].iloc[:11].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert np.allclose(result["size"].iloc[1:5], 0.625)
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
