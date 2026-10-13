from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 126
    volume_window: int = 20
    atr_window: int = 14
    entry_percentile: float = 0.8
    stop_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791263946"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.volume_window + params.rank_window - 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (min(params.rank_window, params.volume_window, params.atr_window) < 2
                or params.hold_bars < 1
                or not 0.5 < params.entry_percentile < 1.0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).replace(0, np.nan)
        out["volume_mean"] = data["volume"].rolling(params.volume_window).mean()
        out["pressure"] = ((data["close"] - data["open"]) / out["range"]
                           * data["volume"] / out["volume_mean"].replace(0, np.nan))
        out["pressure"] = out["pressure"].replace([np.inf, -np.inf], np.nan)
        out["percentile"] = out["pressure"].rolling(params.rank_window).rank(pct=True)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        day = pd.Series(data.index.day, index=data.index)
        month = pd.Series(data.index.year * 12 + data.index.month, index=data.index)
        out["midmonth"] = ((day >= 15) & (day.shift(1) < 15)
                           & month.eq(month.shift(1))).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        position_size = params.min_size
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        ranks = indicators["percentile"].to_numpy(dtype=float)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        gates = indicators["midmonth"].fillna(False).to_numpy(dtype=bool)
        for i in range(n):
            if direction:
                # Read the fill open only when its bar arrives; ATR stays frozen.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                stopped = (not invalid and
                           direction * (closes[i] - entry_price) < -params.stop_atr * entry_atr)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    direction = 0
                    continue
                signals[i] = direction
                sizes[i] = position_size
            elif (gates[i] and np.isfinite(ranks[i]) and np.isfinite(pressure[i])
                  and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i])):
                if ranks[i] >= params.entry_percentile and pressure[i] > 0:
                    direction = 1
                elif ranks[i] <= 1 - params.entry_percentile and pressure[i] < 0:
                    direction = -1
                if direction:
                    conviction = np.clip(2 * abs(ranks[i] - 0.5), 0.0, 1.0)
                    position_size = float(params.min_size
                                          + conviction * (params.max_size - params.min_size))
                    entry_bar = i + 1
                    entry_price = np.nan
                    entry_atr = atr[i]
                    signals[i] = direction
                    sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({"open": 100.0,
                         "close": [100., 100., 97., 100., 100., 100., 103., 100.]},
                        index=index)
    ind = pd.DataFrame({"percentile": [0.9, 0.5, 0.5, 0.5, 0., 0.5, 0.5, 0.5],
                        "pressure": [1., 0., 0., 0., -1., 0., 0., 0.],
                        "atr": [1., 50., 50., 50., 1., 50., 50., 50.],
                        "midmonth": [True, False, False, False, True, False, False, False]},
                       index=index)
    params = Params(stop_atr=2.0)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert 0 < result["size"].iloc[1] < result["size"].iloc[5] <= params.max_size
    params.hold_bars = 2
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
