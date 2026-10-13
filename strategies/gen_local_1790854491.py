from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    entry_z: float = 1.5
    hold_bars: int = 8
    target_vol: float = 0.12

    def __post_init__(self):
        if self.ma_window < 2 or self.hold_bars < 1:
            raise ValueError("ma_window >= 2 and hold_bars >= 1 required")
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790854491"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.ma_window + 1, 24))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        window = close.rolling(params.ma_window, min_periods=params.ma_window)
        out["distance_ma"] = window.mean()
        out["distance_std"] = window.std(ddof=0).replace(0.0, np.nan)
        out["z"] = (close - out["distance_ma"]) / out["distance_std"]
        out["previous_z"] = out["z"].shift(1)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["daily_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        out["prior_vol"] = out["daily_vol"].shift(1).replace(0.0, np.nan)
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["spike"] = (out["return"].abs() >= 2.5 * out["prior_vol"]).astype(float)
        out["spike"] = out["spike"].where(out["prior_vol"].notna())
        # Require three completed shock-free sessions before admission.
        out["blocked"] = out["spike"].rolling(3, min_periods=3).max().fillna(1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        z = indicators["z"]
        previous = indicators["previous_z"]
        ready = indicators[["z", "previous_z", "regime_ma", "annual_vol"]].notna().all(axis=1)
        ready &= indicators["blocked"].eq(0.0) & indicators["annual_vol"].gt(0.0)
        long_entry = (
            ready & data["close"].gt(indicators["regime_ma"])
            & previous.le(-params.entry_z) & z.gt(-params.entry_z) & z.lt(0.0)
        )
        short_entry = (
            ready & data["close"].lt(indicators["regime_ma"])
            & previous.ge(params.entry_z) & z.lt(params.entry_z) & z.gt(0.0)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        annual_vol = indicators["annual_vol"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        position, entry_bar, entry_size = 0, -1, 0.95
        for i in range(len(data)):
            if position:
                if i - entry_bar >= params.hold_bars:
                    position = 0
            elif candidates[i]:
                position = int(candidates[i])
                entry_bar = i
                entry_size = float(np.clip(params.target_vol / max(annual_vol[i], 0.04), 0.05, 0.95))
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=11, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    z = pd.Series([-2.0, -0.5, 2.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0], index=index)
    features = pd.DataFrame({
        "z": z,
        "previous_z": z.shift(1),
        "regime_ma": [90.0, 90.0] + [110.0] * 9,
        "annual_vol": [0.2, 0.2] + [0.4] * 9,
        "blocked": 0.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams(hold_bars=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert np.allclose(result["size"].iloc[2:5], 0.6)
    assert np.allclose(result["size"].iloc[7:10], 0.3)
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(StrategyParams(ma_window=250)) >= 251


if __name__ == "__main__":
    _self_check()
