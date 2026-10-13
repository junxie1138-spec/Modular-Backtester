from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_span: int = 12
    breakout_window: int = 25
    atr_window: int = 14
    projection_bars: int = 5
    innovation_floor: float = 0.01
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791233482"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(4 * params.trend_span + 2,
                       params.breakout_window + 2, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.trend_span < 2 or params.breakout_window < 2
                or params.atr_window < 2 or params.projection_bars < 1
                or params.max_hold_bars < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not np.isfinite(params.innovation_floor)
                or params.innovation_floor <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([(data["high"] - data["low"]).abs(),
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window,
                         min_periods=params.atr_window).mean().replace(0, np.nan)
        smooth = close.ewm(span=params.trend_span, adjust=False,
                           min_periods=params.trend_span).mean()
        slow = smooth.ewm(span=params.trend_span, adjust=False,
                          min_periods=params.trend_span).mean()
        level = 2 * smooth - slow
        alpha = 2.0 / (params.trend_span + 1)
        slope = alpha / (1 - alpha) * (smooth - slow)
        innovation = slope.diff() / atr
        projection = level + params.projection_bars * slope
        ceiling = close.rolling(params.breakout_window,
                                min_periods=params.breakout_window).max().shift(1)
        breakout = ((close > ceiling) & (projection > ceiling)
                    & (slope > 0) & (innovation > params.innovation_floor))
        out = pd.DataFrame({"true_range": tr, "atr": atr,
                            "smooth": smooth, "slow": slow, "level": level,
                            "slope": slope, "slope_innovation": innovation,
                            "projection": projection, "ceiling": ceiling,
                            "breakout": breakout.fillna(False)}, index=data.index)
        out.loc[out.index[:self.warmup_bars(params)], "breakout"] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = indicators["breakout"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0)
            if active:
                if not valid:
                    active = False
                else:
                    # Entry fills on this bar after the previous decision.
                    high_water = (close[i] if np.isnan(high_water)
                                  else max(high_water, close[i]))
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or i - entry_bar >= params.max_hold_bars:
                        active = False
            elif valid and confirmed[i]:
                active = True
                entry_bar = i
                high_water = np.nan
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 103., 102., 99., 100., 101.]},
                        index=index)
    derived = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 1., 1.],
                            "breakout": [True, True, False, False,
                                         False, True, True]}, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, Params())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
