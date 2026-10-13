from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    energy_window: int = 5
    reference_window: int = 20
    percentile_window: int = 126
    change_percentile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.energy_window, self.reference_window, self.percentile_window, self.atr_window, self.hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive")
        if not 0.0 < self.change_percentile < 1.0 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError("Invalid percentile or stop multiplier")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791267129"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(8 + params.energy_window + params.reference_window + params.percentile_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        for half in (1, 2, 4):
            block = out["return"].rolling(half, min_periods=half).sum()
            out[f"haar_{half}"] = (block - block.shift(half)) / np.sqrt(2.0 * half)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        energy_columns = ["energy_1", "energy_2", "energy_4"]
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3).replace(0.0, np.nan)
        for half in (1, 2, 4):
            out[f"share_{half}"] = out[f"energy_{half}"] / out["total_energy"]
            out[f"reference_{half}"] = out[f"share_{half}"].rolling(params.reference_window, min_periods=params.reference_window).mean().shift(1)
            out[f"migration_{half}"] = out[f"share_{half}"] - out[f"reference_{half}"]
        out["change"] = out[["migration_1", "migration_2", "migration_4"]].abs().sum(axis=1, min_count=3) / 2.0
        out["threshold"] = out["change"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.change_percentile).shift(1)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["four_day_progress"] = data["close"] - data["close"].shift(4)
        out["entry"] = (
            (out["change"] > out["threshold"])
            & (out["change"].shift(1) <= out["threshold"].shift(1))
            & (out["migration_2"] > 0.0)
            & (out["migration_1"] < 0.0)
            & (out["haar_2"] > 0.0)
            & (out["four_day_progress"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        entry_bar = 0
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # The preceding close requested entry; this bar's open is the fill proxy.
                active = np.isfinite(opens[i]) and opens[i] > 0.0
                stop = opens[i] - params.stop_atr * entry_atr
                entry_bar = i
                pending = False
            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]):
                signals[i] = 1
                pending = True
                entry_atr = atr[i]
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams(hold_bars=8)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 94.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "entry": False}, index=index)
    indicators.iloc[0, indicators.columns.get_loc("entry")] = True
    data.iloc[2, data.columns.get_loc("close")] = 105.0
    data.iloc[3, data.columns.get_loc("close")] = 97.0
    data.iloc[4, data.columns.get_loc("close")] = 95.0
    indicators.iloc[1:, indicators.columns.get_loc("atr")] = 20.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert (signals["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
