from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 10
    threshold: float = 0.06


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790922101"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.window < 2 or not 0.0 < params.threshold < 0.5:
            raise ValueError("window must be >= 2 and threshold must be between 0 and 0.5")
        n = params.window
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        out["gap"] = (opening / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"].where(lambda s: s > 0) / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        gap = out["gap"]
        intraday = out["intraday"]
        area = pd.Series(0.0, index=data.index)
        # Half the antisymmetric second-order signature of the cumulative path.
        for lag in range(1, n):
            cross = gap.shift(lag) * intraday - intraday.shift(lag) * gap
            area = area + cross.rolling(n - lag, min_periods=n - lag).sum()
        out["area"] = (0.5 * area).fillna(0.0)
        out["gap_variation"] = gap.abs().rolling(n, min_periods=n).sum().fillna(0.0)
        out["intraday_variation"] = intraday.abs().rolling(n, min_periods=n).sum().fillna(0.0)
        out["normalizer"] = out["gap_variation"] * out["intraday_variation"]
        valid = gap.notna() & intraday.notna()
        out["ready"] = valid.astype(int).rolling(n, min_periods=n).sum().eq(n) & out["normalizer"].gt(0.0)
        out["rotation"] = (out["area"] / out["normalizer"].where(out["normalizer"] > 0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["gap"] = out["gap"].fillna(0.0)
        out["intraday"] = out["intraday"].fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        rotation = indicators["rotation"]
        ready = indicators["ready"]
        events = pd.Series(np.nan, index=data.index)
        # Mirrored thresholds: enter below -threshold, exit above +threshold.
        events.loc[ready & rotation.lt(-params.threshold)] = 1.0
        events.loc[ready & rotation.gt(params.threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SignatureParams(window=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"rotation": [-0.1, 0.0, -0.02, 0.1, 0.0, -0.1], "ready": True}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    gap = np.sin(np.arange(20)) * 0.003
    intraday = np.cos(np.arange(20)) * 0.004
    close = 100.0 * np.cumprod((1.0 + gap) * (1.0 + intraday))
    opening = np.r_[100.0, close[:-1]] * (1.0 + gap)
    prices = pd.DataFrame({"open": opening, "high": np.maximum(opening, close), "low": np.minimum(opening, close), "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=20))
    calculated = strategy.indicators(prices, params)
    g = calculated["gap"].iloc[-3:].to_numpy()
    r = calculated["intraday"].iloc[-3:].to_numpy()
    expected = 0.5 * np.triu(np.outer(g, r) - np.outer(r, g), 1).sum()
    assert np.isclose(calculated["area"].iloc[-1], expected)
    assert np.isfinite(calculated["rotation"]).all()
    assert not calculated["ready"].iloc[:3].any()
