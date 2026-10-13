from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_window: int = 5
    acceleration_lag: int = 3
    percentile_window: int = 126
    entry_percentile: float = 0.80
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    stop_atr: float = 2.75
    hold_bars: int = 18

    def __post_init__(self):
        windows = (self.roc_window, self.acceleration_lag, self.percentile_window,
                   self.volatility_window, self.atr_window, self.hold_bars)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 2 for x in windows):
            raise ValueError("Lookbacks and holding period must be integers >= 2")
        if not np.isfinite(self.entry_percentile) or not 0 < self.entry_percentile < 1:
            raise ValueError("entry_percentile must be between zero and one")
        if any(not np.isfinite(x) or x <= 0 for x in (self.target_volatility, self.stop_atr)):
            raise ValueError("Volatility target and stop multiplier must be positive")


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791254759"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params):
        return int(max(params.roc_window + params.acceleration_lag
                       + params.percentile_window + 1,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_window, fill_method=None)
        out["prior_roc"] = out["roc"].shift(params.acceleration_lag)
        out["acceleration"] = out["roc"] - out["prior_roc"]
        out["threshold"] = out["acceleration"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile).shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=1) * np.sqrt(252.0)
        out["target_size"] = (params.target_volatility /
                              out["volatility"].where(out["volatility"] > 0)).clip(0.05, 0.95)
        out["entry"] = (
            (out["acceleration"] > out["threshold"])
            & (out["acceleration"].shift(1) <= out["threshold"].shift(1))
            & (out["roc"] > 0)
            & (out["prior_roc"] < 0)
            & (out["atr"] > 0)
            & (out["volatility"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(count):
            if active:
                if i == entry_bar:
                    # Entry fills on this bar; freeze the decision-bar ATR.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                if (not np.isfinite(closes[i]) or not np.isfinite(fixed_stop)
                        or closes[i] < fixed_stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals}, index=data.index)
        df["size"] = indicators["target_size"].replace(
            [np.inf, -np.inf], np.nan
        ).clip(0.05, 0.95).shift(1).fillna(0.05).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100, 105, 110, 104, 99],
                         "close": [100, 120, 104, 99, 99]}, index=index)
    indicators = pd.DataFrame({"atr": [2.0] * 5,
                               "entry": [True, False, False, False, False],
                               "target_size": [0.5] * 5}, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, AccelerationParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    data["close"] = 110.0
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, AccelerationParams(hold_bars=3)
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
