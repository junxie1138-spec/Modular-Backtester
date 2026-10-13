from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapWaveParams:
    correlation_window: int = 40
    atr_window: int = 14
    min_forecast: float = 0.08
    stop_atr: float = 2.0
    holding_bars: int = 8

    def __post_init__(self):
        if self.correlation_window < 3 or self.atr_window < 1:
            raise ValueError("Lookbacks must be positive; correlation requires at least three bars.")
        if not 5 <= self.holding_bars <= 10:
            raise ValueError("Holding horizon must be five to ten sessions.")
        if not np.isfinite(self.min_forecast) or self.min_forecast <= 0:
            raise ValueError("min_forecast must be finite and positive.")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapWaveParams]):
    strategy_id = "gen_local_1790908418"

    @classmethod
    def params_type(cls):
        return GapWaveParams

    def warmup_bars(self, params):
        return int(params.atr_window + params.correlation_window + 2)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] - out["prior_close"]) / out["prior_atr"]
        out["body"] = (data["close"] - data["open"]) / out["prior_atr"]
        for primitive in ("gap", "body"):
            series = out[primitive].replace([np.inf, -np.inf], np.nan)
            out[primitive] = series
            out[primitive + "_rho"] = series.rolling(
                params.correlation_window, min_periods=params.correlation_window
            ).corr(series.shift(1)).shift(1).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
            out[primitive + "_forecast"] = out[primitive + "_rho"] * series
        long_entry = (out["gap_forecast"] > params.min_forecast) & (out["body_forecast"] > params.min_forecast)
        short_entry = (out["gap_forecast"] < -params.min_forecast) & (out["body_forecast"] < -params.min_forecast)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        stop_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The previous close's decision fills at this opening price.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                    else:
                        stop_price = opens[i] - position * params.stop_atr * atr[i - 1]
                if position:
                    stopped = np.isfinite(closes[i]) and position * (closes[i] - stop_price) < 0
                    expired = i - entry_bar + 1 >= params.holding_bars
                    if stopped or expired:
                        position = 0
                # An exit decision stays flat for this bar; no same-close re-entry.
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                stop_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapWaveParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[:8], "open"] = [100, 101, 100, 100, 100, 100, 100, 100]
    data.loc[index[:8], "close"] = [100, 105, 100, 98, 100, 97, 103, 100]
    derived = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0]
    data["open"] = data["close"] = 100.0
    derived["entry"] = [1] + [0] * 11
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    flat = strategy.indicators(data, params)
    assert flat.index.equals(data.index) and not flat["entry"].any()
