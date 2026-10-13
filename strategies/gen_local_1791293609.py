from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_lookback: int = 24
    breakout_lookback: int = 12
    long_span_share: float = 0.60
    atr_lookback: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791293609"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.graph_lookback + 1,
                       params.breakout_lookback + 2,
                       params.atr_lookback + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.graph_lookback < 4 or params.breakout_lookback < 2
                or params.atr_lookback < 2 or params.max_hold < 1
                or not 0.0 <= params.long_span_share <= 1.0
                or params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        tr = tr.where(previous_close.notna())
        atr = tr.rolling(params.atr_lookback,
                         min_periods=params.atr_lookback).mean()
        resistance = data["high"].shift(1).rolling(
            params.breakout_lookback,
            min_periods=params.breakout_lookback).max()

        # An older node is visible when its slope is below every nearer slope.
        nearest_min = np.full(len(data), np.inf)
        total_span = np.zeros(len(data), dtype=float)
        distant_span = np.zeros(len(data), dtype=float)
        distant_cutoff = (params.graph_lookback + 1) // 2
        for distance in range(1, params.graph_lookback + 1):
            slope = ((close - close.shift(distance)) / distance).to_numpy()
            visible = np.isfinite(slope) & (slope < nearest_min)
            total_span += distance * visible
            if distance >= distant_cutoff:
                distant_span += distance * visible
            nearest_min = np.minimum(nearest_min, slope)
        share = np.divide(distant_span, total_span,
                          out=np.zeros(len(data), dtype=float),
                          where=total_span > 0)
        complete = close.rolling(params.graph_lookback + 1).count().eq(
            params.graph_lookback + 1)
        return pd.DataFrame({
            "atr": atr,
            "resistance": resistance,
            "long_span_share": pd.Series(share, index=data.index).where(complete),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        valid = np.isfinite(indicators).all(axis=1)
        setup = (valid
                 & (indicators["atr"] > 0.0)
                 & (data["close"] > indicators["resistance"])
                 & (indicators["long_span_share"] >= params.long_span_share))
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        holding = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if (np.isfinite(opens[i]) and opens[i] > 0.0 and i > 0
                        and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0.0):
                    holding = True
                    armed = False
                    entry = opens[i]
                    peak = entry
                    stop = entry - params.trail_atr * atrs[i - 1]
                    entry_bar = i
            if holding:
                price = closes[i]
                if not np.isfinite(price) or price <= 0.0:
                    raw[i] = 1
                    continue
                peak = max(peak, price)
                if price >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                if price <= stop or i - entry_bar + 1 >= params.max_hold:
                    holding = False
                else:
                    raw[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                pending = True
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = Params()
    x = np.arange(120, dtype=float)
    c = 100.0 + 0.02 * x * x
    sample = pd.DataFrame({"open": c, "high": c + 0.1,
                           "low": c - 0.1, "close": c,
                           "volume": 1000000.0},
                          index=pd.date_range("2020-01-01", periods=len(x)))
    features = GeneratedStrategy.indicators(sample, p)
    result = GeneratedStrategy.generate_signals(sample, features, None, p).data
    assert result.index.equals(sample.index)
    assert result["signal"].isin([0, 1]).all()
    assert result["signal"].iloc[0] == 0
    assert (result["size"] > 0.0).all()
    assert result["signal"].max() == 1
    assert (result["signal"].diff() == -1).any()
