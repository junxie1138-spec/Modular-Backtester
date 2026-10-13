from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CatchupParams:
    ma_window: int = 20
    atr_window: int = 14
    z_min: float = 0.6
    z_max: float = 2.0
    breakeven_pct: float = 0.004
    initial_atr: float = 1.5
    trail_atr: float = 0.8
    max_hold: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[CatchupParams]):
    strategy_id = "gen_local_1790870033"

    @classmethod
    def params_type(cls):
        return CatchupParams

    def warmup_bars(self, params: CatchupParams) -> int:
        return int(max(200, params.ma_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CatchupParams) -> pd.DataFrame:
        values = (params.z_min, params.z_max, params.breakeven_pct,
                  params.initial_atr, params.trail_atr, params.size)
        if (not all(np.isfinite(v) and v > 0 for v in values)
                or params.z_min >= params.z_max
                or params.ma_window < 2 or params.atr_window < 1
                or params.max_hold not in (1, 2)):
            raise ValueError("Invalid catch-up parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"] / out["std"].replace(0.0, np.nan)
        out["price_step"] = close.diff()
        out["distance_step"] = out["distance"].diff()
        out["regime_ma"] = close.rolling(200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["z_step"] = out["z"].diff()
        long = (
            (close > out["regime_ma"])
            & out["z"].between(params.z_min, params.z_max)
            & (out["price_step"] > 0)
            & (out["distance_step"] < 0)
            & (out["z_step"] < 0)
        )
        short = (
            (close < out["regime_ma"])
            & (-out["z"]).between(params.z_min, params.z_max)
            & (out["price_step"] < 0)
            & (out["distance_step"] > 0)
            & (out["z_step"] > 0)
        )
        valid = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        out["entry"] = np.select([long & valid, short & valid], [1, -1], default=0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CatchupParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry"].to_numpy(dtype=int)
        side, age = 0, 0
        entry = stop = best = np.nan
        armed = False
        for i, (opening, high, low, close) in enumerate(prices):
            if not (np.isfinite(prices[i]).all()
                    and np.all(prices[i] > 0)
                    and np.isfinite(atr[i]) and atr[i] > 0):
                side, age = 0, 0
                entry = np.nan
                continue
            if side:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry = opening
                    best = side * entry
                    stop = best - params.initial_atr * atr[i - 1]
                    armed = False
                age += 1
                adverse = side * (low if side == 1 else high)
                # A stop observed today requests an exit at tomorrow's open.
                if adverse <= stop or age >= params.max_hold:
                    side, age = 0, 0
                    entry = np.nan
                    continue
                best = max(best, side * close)
                armed = armed or (side * (close - entry) / entry >= params.breakeven_pct)
                if armed:
                    # Signed stop only rises: long stops rise; short stops fall.
                    stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                decisions[i] = side
            elif candidates[i]:
                side = int(candidates[i])
                entry = np.nan
                age = 0
                decisions[i] = side
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 101, 100, 100, 99, 100],
        "high": [101, 102, 102, 101, 101, 101, 101],
        "low": [99, 99, 99, 99, 98, 98, 99],
        "close": [100, 101, 100, 100, 99, 100, 100],
        "volume": [1000] * 7,
    }, index=index)
    indicators = pd.DataFrame(
        {"atr": 1.0, "entry": [1, 0, 0, -1, 0, 0, 0]}, index=index
    )
    strategy = GeneratedStrategy()
    params = CatchupParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    stopped = data.copy()
    stopped.loc[index[1], "low"] = 98.0
    stopped.loc[index[4], "high"] = 102.0
    result = strategy.generate_signals(stopped, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    flat = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=220),
    )
    derived = strategy.indicators(flat, params)
    assert strategy.warmup_bars(params) >= 200
    assert derived["entry"].eq(0).all()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
