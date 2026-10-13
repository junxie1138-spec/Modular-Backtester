from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 5
    vol_bars: int = 20
    acceleration_floor: float = 0.25
    target_vol: float = 0.12
    hold_bars: int = 8

    def __post_init__(self):
        if self.roc_bars < 2 or self.vol_bars < 5 or self.hold_bars < 1:
            raise ValueError("Lookbacks and holding period are too small")
        if not np.isfinite(self.acceleration_floor) or self.acceleration_floor < 0:
            raise ValueError("acceleration_floor must be finite and nonnegative")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1790861607"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(params.roc_bars + 3, params.vol_bars + 2, 7))

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["slow_vol"] = out["return"].rolling(params.vol_bars, min_periods=params.vol_bars).std()
        out["fast_vol"] = out["return"].rolling(5, min_periods=5).std()
        out["risk_vol"] = out[["slow_vol", "fast_vol"]].max(axis=1, skipna=False)
        out["roc_change"] = out["roc"].diff()
        out["acceleration"] = (out["roc_change"] / out["risk_vol"].where(out["risk_vol"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["annual_vol"] = out["risk_vol"] * np.sqrt(252.0)
        out["exposure"] = (params.target_vol / out["annual_vol"].clip(lower=0.03)).clip(lower=0.01, upper=1.0).fillna(1.0)
        out["setup"] = (
            (out["roc"] < 0)
            & (out["acceleration"] > params.acceleration_floor)
            & out["risk_vol"].notna()
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if i == exit_bar:
                continue
            if i > exit_bar:
                if not confirmed[i] or not np.isfinite(exposure[i]) or exposure[i] <= 0:
                    continue
                exit_bar = i + params.hold_bars
                entry_size = float(exposure[i])
            signals[i] = 1
            sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = AccelerationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    close = pd.Series(100.0 + np.sin(np.arange(40)), index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    ind = strategy.indicators(data, p)
    assert not ind["confirmed"].iloc[:p.vol_bars + 1].any()
    setup = pd.Series(False, index=index)
    setup.iloc[2:5] = True
    ind["confirmed"] = setup & setup.shift(1, fill_value=False)
    ind["exposure"] = 0.4
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(p))
    result = strategy.generate_signals(data, ind, ctx, p).data
    expected = np.zeros(len(data), dtype=int)
    expected[4:4 + p.hold_bars] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result.loc[result["signal"] == 1, "size"] == 0.4).all()
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
