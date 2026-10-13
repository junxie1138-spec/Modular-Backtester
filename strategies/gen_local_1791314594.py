from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendParams:
    trend_window: int = 16
    noise_window: int = 12
    atr_window: int = 14
    innovation_min: float = 0.25
    innovation_max: float = 2.5
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[TrendParams]):
    strategy_id = "gen_local_1791314594"

    @classmethod
    def params_type(cls) -> type[TrendParams]:
        return TrendParams

    def warmup_bars(self, params: TrendParams) -> int:
        return int(max(params.trend_window + params.noise_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
        if min(params.trend_window, params.noise_window) < 3 or params.atr_window < 1:
            raise ValueError("Invalid indicator window")
        if not 0 <= params.innovation_min < params.innovation_max:
            raise ValueError("Invalid innovation thresholds")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.max_hold_bars < 1:
            raise ValueError("Invalid exit parameters")
        out = pd.DataFrame(index=data.index)
        if data.empty:
            return out.assign(atr=np.nan, confirmed=False)
        n = params.trend_window
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["mean"] = out["log_close"].rolling(n, min_periods=n).mean()
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        slope = np.convolve(out["log_close"].to_numpy(), weights[::-1], mode="full")[:len(data)]
        out["slope"] = pd.Series(slope / np.square(weights).sum(), index=data.index).where(out["mean"].notna())
        out["level"] = out["mean"] + out["slope"] * (n - 1) / 2.0
        out["forecast"] = (out["level"] + out["slope"]).shift(1)
        out["forecast_error"] = out["log_close"] - out["forecast"]
        out["noise"] = out["forecast_error"].shift(1).rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0)
        out["innovation"] = out["forecast_error"] / out["noise"].where(out["noise"] > 1e-12)
        out["log_variance"] = out["log_close"].rolling(n, min_periods=n).var(ddof=0)
        out["residual_rms"] = np.sqrt((out["log_variance"] - out["slope"].pow(2) * (n * n - 1) / 12.0).clip(lower=0))
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["slope_innovation"] = out["slope"].diff()
        out["qualifies"] = (
            out["slope"].gt(0)
            & out["slope_innovation"].gt(0)
            & out["innovation"].between(params.innovation_min, params.innovation_max)
            & out["residual_rms"].lt(out["residual_rms"].shift(3))
            & out["atr"].gt(0)
        ).fillna(False)
        out["confirmed"] = out["qualifies"] & out["qualifies"].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TrendParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                continue
            # The previous close's target fills at this bar's open.
            if not active and i > 0 and raw[i - 1] == 1:
                entry = opening
                peak = entry
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                # Check the previously established stop before raising it.
                if low <= stop or age >= params.max_hold_bars:
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # New stops are evaluated at the close; no assumed high/low ordering.
                    if close <= stop:
                        active = False
                        continue
                raw[i] = 1
            elif confirmed[i]:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100, 100, 101, 104, 103, 100],
        "high": [100, 101, 104, 106, 104, 101],
        "low": [99, 99.5, 100.5, 103, 99, 99],
        "close": [100, 101, 104, 105, 100, 100],
        "volume": [1000] * 6,
    }, index=index)
    ind = pd.DataFrame({"atr": 2.0, "confirmed": [False, True, True, True, True, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, TrendParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
    rising = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": np.exp(4.6 + np.arange(80) * 0.002), "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=80))
    derived = GeneratedStrategy.indicators(None, rising, TrendParams())
    assert np.allclose(derived["slope"].dropna(), 0.002)
    assert derived["confirmed"].equals(derived["qualifies"] & derived["qualifies"].shift(1, fill_value=False))


if __name__ == "__main__":
    _self_check()
