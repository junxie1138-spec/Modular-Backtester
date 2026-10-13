from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 9
    correlation_window: int = 20
    atr_window: int = 14
    innovation_min: float = 0.08
    correlation_min: float = 0.10
    stop_atr: float = 1.75
    hold_bars: int = 4
    entry_slippage_bps: float = 5.0

    def __post_init__(self):
        if min(self.trend_window, self.correlation_window, self.atr_window) < 3:
            raise ValueError("Lookbacks must be at least three bars.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not (np.isfinite(self.innovation_min) and self.innovation_min > 0):
            raise ValueError("innovation_min must be finite and positive.")
        if not (np.isfinite(self.correlation_min) and 0 <= self.correlation_min < 1):
            raise ValueError("correlation_min must be in [0, 1).")
        if not (np.isfinite(self.stop_atr) and self.stop_atr > 0):
            raise ValueError("stop_atr must be finite and positive.")
        if not (np.isfinite(self.entry_slippage_bps) and self.entry_slippage_bps >= 0):
            raise ValueError("entry_slippage_bps must be finite and nonnegative.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791242408"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_window + 1,
                       params.correlation_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["lag_return"] = out["return"].shift(1)
        out["time"] = np.arange(len(data), dtype=float)
        n = params.trend_window
        out["slope"] = out["log_close"].rolling(n, min_periods=n).cov(
            out["time"], ddof=0
        ) / ((n * n - 1) / 12.0)
        out["slope_change"] = out["slope"].diff()
        out["return_vol"] = out["return"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).std(ddof=0)
        out["innovation"] = out["slope_change"] / out["return_vol"].where(
            out["return_vol"] > 0
        )
        out["autocorrelation"] = out["return"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(out["lag_return"])
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = (
            (indicators["innovation"] > params.innovation_min)
            & (indicators["slope"] > 0)
            & (indicators["autocorrelation"] > params.correlation_min)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Freeze the stop using the opening fill and decision-bar ATR.
                    entry_price = opens[i] * (1 + params.entry_slippage_bps / 10000)
                    stop = entry_price - params.stop_atr * atr[i - 1]
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 94.0,
        "close": [100, 110, 99, 96, 100, 100, 100, 100, 100, 100],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "innovation": 1.0, "slope": 1.0, "autocorrelation": 0.5, "atr": 2.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    # Includes the one-bar delay, fixed (not trailing) stop, and four-day expiry.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    indicators["autocorrelation"] = -0.5
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()
    indicators["autocorrelation"] = 0.5
    indicators["innovation"] = np.nan
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
