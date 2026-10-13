from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    percentile_window: int = 126
    entry_percentile: float = 0.75
    atr_window: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791236745"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params):
        return int(max(16 + 5 + 5 + params.percentile_window, params.atr_window + 1))

    def indicators(self, data, params):
        if (params.percentile_window < 2 or params.atr_window < 1
                or params.hold_bars < 1 or not np.isfinite(params.trail_atr)
                or params.trail_atr <= 0 or not 0 < params.entry_percentile < 1):
            raise ValueError("Invalid wavelet parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(5, min_periods=5).mean()
        total = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["centroid"] = (
            out["energy_2"] + 2.0 * out["energy_4"] + 3.0 * out["energy_8"]
        ) / total.where(total > 0)
        out["migration"] = out["centroid"].diff(5)
        out["threshold"] = out["migration"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        aligned = (out["haar_2"] * out["haar_8"] > 0) & (out["haar_4"] * out["haar_8"] > 0)
        ready = out[["migration", "threshold", "atr", "haar_8"]].notna().all(axis=1)
        setup = ready & aligned & (out["migration"] > out["threshold"]) & (out["migration"] > 0)
        out["candidate"] = np.where(setup, np.sign(out["haar_8"].fillna(0)), 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        position = age = 0
        high = low = lower_stop = upper_stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = age = 0
            elif position:
                # Initialize on the fill bar, after the entry decision.
                if age == 0:
                    high = low = close[i]
                    lower_stop = high - params.trail_atr * atr[i]
                    upper_stop = low + params.trail_atr * atr[i]
                else:
                    high = max(high, close[i])
                    low = min(low, close[i])
                    lower_stop = max(lower_stop, high - params.trail_atr * atr[i])
                    upper_stop = min(upper_stop, low + params.trail_atr * atr[i])
                age += 1
                # The required high-water drawdown exit applies to both sides.
                # Shorts additionally have a mirrored adverse-move trailing stop.
                if (close[i] < lower_stop
                        or (position == -1 and close[i] > upper_stop)
                        or age >= params.hold_bars):
                    position = age = 0
            elif i >= self.warmup_bars(params) - 1:
                position = int(candidate[i])
                age = 0
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams(percentile_window=2, trail_atr=2.0)
    warmup = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=warmup + 7)
    prices = np.r_[np.full(warmup, 100.0), [100, 101, 105, 104, 101, 100, 100]]
    data = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "candidate": 0}, index=index)
    indicators.iloc[warmup, indicators.columns.get_loc("candidate")] = 1
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=warmup)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[warmup:warmup + 7].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators["candidate"] *= -1
    short = strategy.generate_signals(data, indicators, ctx, params).data
    assert short["signal"].iloc[warmup:warmup + 5].tolist() == [0, -1, -1, 0, 0]
    assert result.index.equals(data.index) and (result["size"] > 0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
