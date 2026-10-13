from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    graph_window: int = 12
    season_window: int = 504
    visibility_threshold: float = 0.60
    min_samples: int = 8
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = "gen_local_1790976424"

    @classmethod
    def params_type(cls) -> type[VisibilitySeasonParams]:
        return VisibilitySeasonParams

    def warmup_bars(self, params: VisibilitySeasonParams) -> int:
        return int(max(params.atr_window + 1,
                       params.graph_window + params.hold_bars + params.season_window + 2))

    def indicators(self, data: pd.DataFrame, params: VisibilitySeasonParams) -> pd.DataFrame:
        if (params.graph_window < 2 or params.season_window < 1
                or params.min_samples < 1 or params.atr_window < 1
                or params.hold_bars < 1 or params.stop_atr <= 0
                or not 0 <= params.visibility_threshold <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()

        # Natural visibility: a backward edge exists when its slope exceeds
        # every nearer backward slope, placing intermediate nodes below it.
        maximum = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        ascending = pd.Series(0.0, index=data.index)
        for distance in range(1, params.graph_window + 1):
            slope = (out["log_close"].shift(distance) - out["log_close"]) / distance
            visible = slope.notna() & slope.gt(maximum)
            degree += visible.astype(float)
            ascending += (visible & slope.lt(0)).astype(float)
            maximum = pd.Series(np.maximum(maximum.to_numpy(),
                                           slope.fillna(-np.inf).to_numpy()), index=data.index)
        complete = out["log_close"].rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        out["degree"] = degree.where(complete)
        out["ascending_degree"] = ascending.where(complete)
        out["ascent_share"] = out["ascending_degree"] / out["degree"].replace(0, np.nan)
        out["topology_ready"] = out["ascent_share"].ge(params.visibility_threshold) & out["degree"].ge(3)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)

        # A training observation becomes available only after its entire
        # holding period has elapsed, plus one bar of separation.
        lag = params.hold_bars + 1
        out["mature_return"] = (close / close.shift(params.hold_bars) - 1).shift(1).replace([np.inf, -np.inf], np.nan)
        out["mature_topology"] = out["topology_ready"].shift(lag).fillna(False).astype(bool)
        out["mature_weekday"] = out["weekday"].shift(lag)
        seasonal = pd.Series(np.nan, index=data.index)
        for weekday in range(5):
            observed = (out["mature_weekday"].eq(weekday)
                        & out["mature_topology"] & out["mature_return"].notna())
            count = observed.astype(float).rolling(params.season_window).sum()
            total = out["mature_return"].where(observed, 0.0).rolling(params.season_window).sum()
            mean = (total / count.replace(0, np.nan)).where(count.ge(params.min_samples))
            out[f"weekday_{weekday}_samples"] = count
            out[f"weekday_{weekday}_mean"] = mean
            seasonal = seasonal.where(~out["weekday"].eq(weekday), mean)
        out["seasonal_return"] = seasonal
        out["eligible"] = (out["topology_ready"] & seasonal.gt(0)
                           & out["atr"].gt(0) & np.isfinite(out["atr"])
                           & close.gt(0) & np.isfinite(close))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilitySeasonParams) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1).fillna(False).astype(bool)).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The previous close requested entry; this open is the fill.
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        active = False
                        continue
                    fixed_stop = opening[i] - params.stop_atr * entry_atr
                stop_hit = np.isfinite(close[i]) and close[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    active = False
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilitySeasonParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"eligible": [True, True] + [False] * 10,
                             "atr": 1.0}, index=index)
    timed = strategy.generate_signals(data, features, None, params).data
    assert timed["signal"].tolist() == [0, 0] + [1] * 8 + [0, 0]
    data.loc[index[2], "open"] = 110.0
    data.loc[index[2], "close"] = 107.0
    stopped = strategy.generate_signals(data, features, None, params).data
    assert stopped["signal"].tolist() == [0, 0, 1] + [0] * 9
    assert stopped["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["eligible"].any()
    assert strategy.warmup_bars(params) >= params.season_window + params.hold_bars + params.graph_window + 2


if __name__ == "__main__":
    _self_check()
