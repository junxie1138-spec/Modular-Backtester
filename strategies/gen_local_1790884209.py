from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphParams:
    graph_lookback: int = 16
    hub_degree: int = 4
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    stop_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GraphParams]):
    strategy_id = "gen_local_1790884209"

    @classmethod
    def params_type(cls):
        return GraphParams

    def warmup_bars(self, params: GraphParams) -> int:
        return int(max(200, params.graph_lookback + 2,
                       params.vol_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
        if not (2 <= params.graph_lookback <= 100
                and 2 <= params.hub_degree <= params.graph_lookback
                and params.vol_window >= 2 and params.atr_window >= 2
                and 3 <= params.hold_bars <= 5
                and np.isfinite(params.target_vol) and params.target_vol > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid graph, volatility, stop, or holding parameters")
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()

        # Horizontal visibility on -close: both endpoint prices must lie
        # strictly below every intervening close. Only backward edges exist.
        degree = (close.notna() & previous.notna()).astype(float)
        for lag in range(2, params.graph_lookback + 1):
            interior_min = previous.rolling(lag - 1, min_periods=lag - 1).min()
            endpoints_max = pd.concat([close, close.shift(lag)], axis=1).max(
                axis=1, skipna=False)
            degree += (endpoints_max < interior_min).astype(float)
        complete = close.rolling(
            params.graph_lookback + 1, min_periods=params.graph_lookback + 1).count()
        out["degree"] = degree.where(complete == params.graph_lookback + 1)
        out["prior_degree"] = out["degree"].shift(1)
        out["rebound"] = close > previous
        out["exposure"] = (params.target_vol / out["vol"].clip(lower=0.01)).clip(
            lower=0.05, upper=0.95)
        out["entry"] = ((out["prior_degree"] >= params.hub_degree)
                        & (out["degree"] <= 2) & out["rebound"]
                        & (close > out["ma200"]) & (out["atr"] > 0)
                        & out["vol"].notna()).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GraphParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        vol = indicators["vol"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0
                     and np.isfinite(ma[i]) and np.isfinite(vol[i]))
            if active:
                # The entry decision filled on this bar, so the first
                # in-trade close, rather than the decision close, seeds the high.
                age += 1
                if valid:
                    high_water = (max(high_water, close[i])
                                  if np.isfinite(high_water) else close[i])
                    stop = max(stop, high_water - params.stop_atr * atr[i])
                if (not valid or close[i] <= ma[i] or close[i] <= stop
                        or age >= params.hold_bars):
                    active = False
            elif valid and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals}, index=data.index)
        df["size"] = indicators["exposure"].shift(1).replace(
            [np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GraphParams(graph_lookback=6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    close = pd.Series([101., 110., 102., 109., 103., 108., 100.],
                      index=pd.date_range("2020-01-01", periods=7))
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.})
    derived = strategy.indicators(data, params)
    assert derived["degree"].iloc[-1] == 4
    assert not derived["entry"].any()
    assert strategy.warmup_bars(params) >= 200

    params.hold_bars = 5
    path = data.copy()
    path["close"] = [100., 105., 104., 102., 101., 102., 103.]
    fake = pd.DataFrame({"atr": [2., 2., 5., 5., 5., 5., 5.],
                         "ma200": 90., "vol": 0.2,
                         "entry": [True, False, False, False, False, False, False],
                         "exposure": [0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6]},
                        index=path.index)
    result = strategy.generate_signals(path, fake, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].iloc[1] == 0.6
    fake["atr"] = 100.
    params.hold_bars = 4
    result = strategy.generate_signals(path, fake, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
