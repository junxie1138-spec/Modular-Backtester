from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_window: int = 6
    percentile_window: int = 60
    entry_quantile: float = 0.85
    breakout_window: int = 8
    atr_period: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4
    size: float = 1.0

    def __post_init__(self):
        windows = (self.slope_window, self.percentile_window,
                   self.breakout_window, self.atr_period)
        if any(not isinstance(w, int) or w < 2 for w in windows):
            raise ValueError("Lookbacks must be integers >= 2")
        if not isinstance(self.hold_bars, int) or not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer from 3 to 5")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        if not np.isfinite(self.size) or not 0.0 < self.size <= 1.0:
            raise ValueError("size must be between zero and one")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791292552"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.slope_window + params.percentile_window + 3,
                       params.atr_period + 1, params.breakout_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["relative_atr"] = out["atr"] / data["close"].where(data["close"] > 0)
        for wall in ("high", "low"):
            out[wall + "_log"] = np.log(data[wall].where(data[wall] > 0))
            out[wall + "_level"] = out[wall + "_log"].rolling(params.slope_window).mean()
            out[wall + "_slope"] = out[wall + "_level"].diff()
            # Innovation relative to a one-step constant-slope prediction.
            out[wall + "_innovation"] = out[wall + "_slope"].diff()
        out["score"] = out[["high_innovation", "low_innovation"]].min(
            axis=1, skipna=False
        ) / out["relative_atr"].where(out["relative_atr"] > 0)
        out["threshold"] = out["score"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        out["breakout_level"] = data["high"].shift(1).rolling(
            params.breakout_window
        ).max()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        ready = indicators[["score", "threshold", "breakout_level", "atr",
                            "high_slope", "low_slope"]].notna().all(axis=1)
        entry = (
            ready & (indicators["score"] > indicators["threshold"])
            & (indicators["score"] > 0)
            & (indicators["high_slope"] > 0)
            & (indicators["low_slope"] > 0)
            & (data["close"] > indicators["breakout_level"])
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Observe the fill-session open only after that bar arrives.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= params.hold_bars
                invalid = not np.isfinite(closes[i]) or not np.isfinite(fixed_stop)
                if invalid or closes[i] < fixed_stop or expired:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif entry[i]:
                entry_bar = i + 1
                entry_atr = atr[i]
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100., 110., 115., 112., 108., 106., 106., 106.],
        "close": [100., 111., 120., 108., 106., 106., 106., 106.],
        "high": 125., "low": 95., "volume": 1000.,
    }, index=index)
    ind = pd.DataFrame({
        "score": [1.] + [0.] * 7, "threshold": 0.5,
        "breakout_level": 99., "atr": [2.] + [50.] * 7,
        "high_slope": 1., "low_slope": 1.,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    stopped = strategy.generate_signals(
        data, ind, ctx, GeneratedParams(hold_bars=5)
    ).data
    # Entry open 110 gives a fixed floor of 107: the rise to 120 does not trail.
    assert stopped["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data["close"] = 110.
    timed = strategy.generate_signals(data, ind, ctx, GeneratedParams()).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert timed["size"].gt(0).all()
    actual = strategy.indicators(data, GeneratedParams())
    assert strategy.generate_signals(data, actual, ctx, GeneratedParams()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
