from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_lookback: int = 12
    bridge_density: float = 0.65
    displacement_atr: float = 0.5
    atr_period: int = 14
    vol_period: int = 20
    target_vol: float = 0.12
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1790860465"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return max(params.graph_lookback + 2, params.atr_period + 1,
                   params.vol_period + 1, 4) + 1

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if (params.graph_lookback < 3 or params.atr_period < 2
                or params.vol_period < 2 or not 0 < params.bridge_density <= 1
                or not np.isfinite(params.displacement_atr) or params.displacement_atr <= 0
                or not np.isfinite(params.target_vol) or params.target_vol <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid topology or risk parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        relative_range = out["tr"] / previous
        out["log_range"] = np.log(relative_range.where(relative_range > 0))
        x = out["log_range"]
        max_slope = pd.Series(-np.inf, index=data.index)
        links = pd.Series(0.0, index=data.index)
        # A past vertex is visible when its backward slope beats every intervening slope.
        for lag in range(1, params.graph_lookback + 1):
            slope = (x.shift(lag) - x) / lag
            visible = slope.gt(max_slope) & slope.notna()
            if lag >= 2:
                links += visible.astype(float)
            max_slope = pd.concat([max_slope, slope], axis=1).max(axis=1)
        full_graph = x.rolling(params.graph_lookback + 1).count().eq(params.graph_lookback + 1)
        out["bridge_density"] = (links / (params.graph_lookback - 1)).where(full_graph)
        out["atr"] = out["tr"].rolling(params.atr_period).mean()
        out["realized_vol"] = close.pct_change(fill_method=None).rolling(params.vol_period).std() * np.sqrt(252.0)
        out["displacement"] = (close - close.shift(3)) / out["atr"].where(out["atr"] > 0)
        out["cooling"] = x.lt(x.shift(1))
        out["target_size"] = (params.target_vol / out["realized_vol"].where(out["realized_vol"] > 0)).clip(0.05, 1.0)
        ready = (
            out["bridge_density"].ge(params.bridge_density)
            & out["cooling"]
            & out["displacement"].abs().ge(params.displacement_atr)
            & out["target_size"].notna()
            & out["atr"].gt(0)
        )
        out["candidate"] = pd.Series(
            np.where(ready, -np.sign(out["displacement"].fillna(0)), 0),
            index=data.index
        ).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        target_size = indicators["target_size"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_atr = 0.0
        stop = 0.0
        frozen_size = 1.0
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Entry fills on this bar's open; ATR was frozen at the prior close.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                invalid = not np.isfinite(close[i]) or not np.isfinite(stop)
                stopped = position * (close[i] - stop) < 0
                expired = i - entry_bar + 1 >= 2
                signals[i] = 0 if invalid or stopped or expired else position
                sizes[i] = frozen_size
                if signals[i] == 0:
                    position = 0
                continue
            if (i >= self.warmup_bars(params) - 1 and candidate[i] != 0
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(target_size[i]) and target_size[i] > 0
                    and np.isfinite(close[i])):
                position = int(candidate[i])
                entry_bar = i + 1
                entry_atr = float(atr[i])
                frozen_size = float(target_size[i])
                signals[i] = position
                sizes[i] = frozen_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = TopologyParams(graph_lookback=3, atr_period=2, vol_period=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "confirmed": 0,
                               "target_size": 0.4}, index=index)
    indicators.loc[index[[7, 12, 17]], "confirmed"] = [1, -1, 1]
    data.loc[index[8], "close"] = 97.0
    data.loc[index[13], "close"] = 103.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[7] == 0
    assert result["signal"].iloc[8] == 1 and result["signal"].iloc[9] == 0
    assert result["signal"].iloc[13] == -1 and result["signal"].iloc[14] == 0
    assert result["signal"].iloc[18:21].tolist() == [1, 1, 0]
    assert result["size"].iloc[18:20].eq(0.4).all()
    assert result["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived["confirmed"].ne(0).sum() == 0
    assert derived["bridge_density"].isna().iloc[:4].all()
    prefix = strategy.indicators(data.iloc[:16], params)
    assert prefix.equals(derived.iloc[:16])
    t = np.arange(60, dtype=float)
    prices = pd.Series(100.0 - 0.5 * t + 0.01 * np.sin(t))
    prior = prices.shift(1).fillna(prices.iloc[0])
    ranges = prior * np.exp(-4.0 + 2.0 / (t + 1.0))
    cooling = pd.DataFrame({"open": prices, "close": prices,
                            "high": prices + ranges / 2,
                            "low": prices - ranges / 2, "volume": 1000.0})
    topology = strategy.indicators(cooling, params)
    assert topology["bridge_density"].iloc[10:].gt(0.99).all()
    assert topology["confirmed"].iloc[10:].eq(1).all()
    assert (topology["confirmed"].ne(0) <= (
        topology["candidate"].ne(0)
        & topology["candidate"].eq(topology["candidate"].shift(1)))).all()


if __name__ == "__main__":
    _self_check()
