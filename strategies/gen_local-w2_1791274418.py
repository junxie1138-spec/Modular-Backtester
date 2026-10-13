from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalEnergyParams:
    season_window: int = 252
    percentile_window: int = 126
    energy_percentile: float = 0.8
    atr_period: int = 14
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalEnergyParams]):
    strategy_id = "gen_local-w2_1791274418"

    @classmethod
    def params_type(cls):
        return SeasonalEnergyParams

    def warmup_bars(self, params):
        return int(max(params.season_window + params.hold_bars,
                       params.percentile_window + 20, params.atr_period + 1))

    def indicators(self, data, params):
        if not (params.season_window >= 40 and params.percentile_window >= 20
                and 0 < params.energy_percentile < 1 and params.atr_period >= 1
                and params.stop_atr > 0 and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        log_close = np.log(close)
        out["recent_return"] = log_close.diff(params.hold_bars)
        energies = []
        for half in (1, 2, 4, 8):
            average = log_close.rolling(half, min_periods=half).mean()
            coefficient = np.sqrt(half / 2.0) * (average - average.shift(half))
            out[f"haar_{half}"] = coefficient
            out[f"energy_{half}"] = coefficient.pow(2).rolling(4, min_periods=4).mean()
            energies.append(out[f"energy_{half}"])
        out["total_energy"] = pd.concat(energies, axis=1).sum(axis=1, min_count=4)
        out["fine_share"] = (out["energy_1"] + out["energy_2"]) / out["total_energy"].where(out["total_energy"] > 0)
        out["energy_cutoff"] = out["fine_share"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.energy_percentile)
        weekdays = pd.Series(data.index.dayofweek, index=data.index)
        # A return observed today belongs to the decision weekday hold_bars ago.
        origin_weekday = weekdays.shift(params.hold_bars)
        seasonal = pd.Series(np.nan, index=data.index)
        for weekday in range(5):
            observed = origin_weekday.eq(weekday) & out["recent_return"].notna()
            count = observed.astype(float).rolling(
                params.season_window, min_periods=params.season_window
            ).sum()
            total = out["recent_return"].where(observed, 0.0).rolling(
                params.season_window, min_periods=params.season_window
            ).sum()
            out[f"weekday_mean_{weekday}"] = total / count.where(count >= 8)
            seasonal = seasonal.where(weekdays.ne(weekday), out[f"weekday_mean_{weekday}"])
        out["seasonal_edge"] = seasonal
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        edge = indicators["seasonal_edge"]
        qualified = (
            indicators["fine_share"].gt(indicators["energy_cutoff"])
            & (indicators["recent_return"] * edge).lt(0)
            & indicators["atr"].gt(0)
        )
        candidates = np.sign(edge).where(qualified, 0).fillna(0).to_numpy(dtype=int)
        candidates[:self.warmup_bars(params)] = 0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        stop, entry_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    # Entry fills at this open; ATR was frozen at the prior close.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                if stopped or age >= params.hold_bars or not np.isfinite(closes[i]):
                    position, age = 0, 0
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position, age = int(candidates[i]), 0
                entry_atr = atrs[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SeasonalEnergyParams(season_window=40, percentile_window=20)
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params)
    index = pd.bdate_range("2020-01-01", periods=n + 7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"seasonal_edge": 1.0, "recent_return": -0.01,
                        "fine_share": 0.9, "energy_cutoff": 0.8, "atr": 1.0},
                       index=index)
    result = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert result.iloc[n:n + 6].tolist() == [0, 1, 1, 1, 1, 0]
    data.iloc[n + 2, data.columns.get_loc("close")] = 97.0
    result = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert result.iloc[n:n + 4].tolist() == [0, 1, 1, 0]
    ind["seasonal_edge"] = -1.0
    ind["recent_return"] = 0.01
    data["close"] = 100.0
    data.iloc[n + 2, data.columns.get_loc("close")] = 103.0
    result = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert result.iloc[n:n + 4].tolist() == [0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
