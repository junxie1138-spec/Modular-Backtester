from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapConsensusParams:
    lookback: int = 8
    atr_window: int = 14
    coherence: float = 0.45
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    initial_stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapConsensusParams]):
    strategy_id = "gen_local_1791323862"

    @classmethod
    def params_type(cls):
        return GapConsensusParams

    def warmup_bars(self, params: GapConsensusParams) -> int:
        return int(max(params.lookback, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapConsensusParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError("Lookback, ATR window and holding limit must be positive.")
        if not (0.0 < params.coherence < 1.0 and params.breakeven_pct > 0.0
                and params.trail_atr > 0.0 and params.initial_stop_atr > 0.0):
            raise ValueError("Invalid coherence or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1.0
        out["body"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1).where(out["prior_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        for primitive in ("gap", "body"):
            net = out[primitive].rolling(params.lookback).sum()
            travel = out[primitive].abs().rolling(params.lookback).sum()
            out[primitive + "_strength"] = net / travel.where(travel > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapConsensusParams,
    ) -> SignalFrame:
        gap = indicators["gap_strength"]
        body = indicators["body_strength"]
        candidates = np.select([
            (gap > params.coherence) & (body > params.coherence),
            (gap < -params.coherence) & (body < -params.coherence),
        ], [1, -1], default=0)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = np.isfinite(indicators[["gap_strength", "body_strength", "atr"]]
                            .to_numpy(dtype=float)).all(axis=1) & (atr > 0)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry, best, stop = np.nan, np.nan, np.nan
        armed = False

        for i, (opening, high, low, close) in enumerate(prices):
            if not np.isfinite([opening, high, low, close]).all() or opening <= 0 or not ready[i]:
                position = 0
                entry = np.nan
                continue
            if position:
                # An entry decision on i-1 fills at today's open.
                if np.isnan(entry):
                    entry = opening
                    best = position * entry
                    stop = best - params.initial_stop_atr * atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                worst = low if position == 1 else -high
                favorable = high if position == 1 else -low
                signed_close = position * close
                # Breaches of yesterday's barrier produce a next-open flat signal.
                exit_now = worst <= stop
                best = max(best, favorable)
                if best - position * entry >= params.breakeven_pct * entry:
                    armed = True
                if armed:
                    # Signed-price stops only rise; actual short stops only fall.
                    stop = max(stop, position * entry, best - params.trail_atr * atr[i])
                    # Use the close for the newly raised barrier: no assumed intrabar ordering.
                    exit_now = exit_now or signed_close <= stop
                if exit_now or age >= params.max_hold:
                    position = 0
                    entry = np.nan
                    continue
            elif candidates[i]:
                position = int(candidates[i])
                entry = np.nan
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapConsensusParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 102.0, 101.0, 100.0],
        "high": [100.5, 102.5, 103.0, 102.0, 100.5],
        "low": [99.5, 99.5, 101.5, 99.5, 99.5],
        "close": [100.0, 102.0, 102.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "gap_strength": [1.0, 0.0, 0.0, 0.0, 0.0],
        "body_strength": [1.0, 0.0, 0.0, 0.0, 0.0],
        "atr": 1.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    mismatch = ind.copy()
    mismatch["body_strength"] = -mismatch["body_strength"]
    assert strategy.generate_signals(data, mismatch, ctx, params).data["signal"].eq(0).all()
    mirrored = data.copy()
    mirrored["open"] = 200.0 - data["open"]
    mirrored["close"] = 200.0 - data["close"]
    mirrored["high"] = 200.0 - data["low"]
    mirrored["low"] = 200.0 - data["high"]
    short_ind = ind.copy()
    short_ind[["gap_strength", "body_strength"]] *= -1.0
    assert strategy.generate_signals(mirrored, short_ind, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, 0]
    for length in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:length], ind.iloc[:length], ctx, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:length])
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
