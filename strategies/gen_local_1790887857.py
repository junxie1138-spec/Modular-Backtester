from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 8
    participation_window: int = 20
    atr_window: int = 14
    saturation: float = 0.68
    innovation_threshold: float = 0.20
    trail_atr: float = 1.8
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.slope_window, self.participation_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.5 < self.saturation < 1.0:
            raise ValueError("Saturation must lie between 0.5 and 1.")
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0:
            raise ValueError("Innovation threshold must be finite and positive.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("ATR multiplier must be finite and positive.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("Holding period must be three to five bars.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790887857"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.slope_window + 4, params.participation_window + 2, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["delta"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean().replace(0, np.nan)
        out["slope"] = out["delta"].rolling(params.slope_window).mean()
        out["predicted_slope"] = out["slope"].shift(1)
        out["innovation"] = out["delta"] - out["predicted_slope"]
        out["innovation_atr"] = out["innovation"] / out["atr"]
        out["innovation_change"] = out["innovation_atr"].diff()
        volume = data["volume"].replace([np.inf, -np.inf], np.nan).clip(lower=0)
        out["up_volume"] = volume.where(out["delta"] > 0, 0.0).where(out["delta"].notna())
        out["volume_sum"] = volume.rolling(params.participation_window).sum()
        out["up_volume_sum"] = out["up_volume"].rolling(params.participation_window).sum()
        out["participation"] = out["up_volume_sum"] / out["volume_sum"].replace(0, np.nan)
        out["regime"] = np.select([
            out["participation"] >= params.saturation,
            out["participation"].between(0.45, params.saturation, inclusive="left"),
        ], [2, 1], default=0)
        expanding = (out["regime"] == 1) & (out["innovation_atr"] > params.innovation_threshold)
        saturated = (
            (out["regime"] == 2)
            & (out["innovation_atr"] < -params.innovation_threshold)
            & (out["innovation_change"] > 0)
        )
        valid = np.isfinite(out[["atr", "slope", "innovation_atr", "innovation_change", "participation"]]).all(axis=1)
        out["candidate"] = (valid & (out["slope"] > 0) & (expanding | saturated)).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(False).astype(bool)
        confirmed = (
            candidate
            & candidate.shift(1, fill_value=False)
            & indicators["regime"].eq(indicators["regime"].shift(1))
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if closes[i] <= stop or i - entry_bar >= params.hold_bars:
                        active = False
            elif valid and confirmed[i]:
                active = True
                entry_bar = i
                # Entry fills next bar; initialize the high from its close.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"close": [100., 100., 101., 104., 103., 102., 102., 102.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 1., 3., 3., 1., 1.],
        "candidate": [True, True, False, False, False, False, False, False],
        "regime": 1,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    indicators.loc[index[1], "candidate"] = False
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    indicators.loc[index[1], "candidate"] = True
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
