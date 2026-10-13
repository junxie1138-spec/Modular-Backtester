from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReplicationParams:
    lookback: int = 12
    replication_threshold: float = 0.12
    acceptance_threshold: float = 0.65
    atr_period: int = 14
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReplicationParams]):
    strategy_id = "gen_local-w2_1791250946"

    @classmethod
    def params_type(cls):
        return ReplicationParams

    def warmup_bars(self, params: ReplicationParams) -> int:
        return max(params.lookback + 2, params.atr_period + 1, 3)

    def indicators(self, data: pd.DataFrame, params: ReplicationParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; lookback must be at least two.")
        if not (0 <= params.replication_threshold <= 1 and 0 <= params.acceptance_threshold <= 1):
            raise ValueError("Signal thresholds must be between zero and one.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("Stops must be positive and holding time must be one or two sessions.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        # Signed returns following up days measure reproduction of buying shocks.
        out["offspring"] = out["return"].where(out["previous_return"] > 0, 0.0).where(
            out["return"].notna() & out["previous_return"].notna()
        )
        out["return_mass"] = out["return"].abs().rolling(params.lookback).sum()
        out["replication"] = out["offspring"].rolling(params.lookback).sum().div(
            out["return_mass"].replace(0.0, np.nan)
        )
        out["close_location"] = (data["close"] - data["low"]).div(
            (data["high"] - data["low"]).replace(0.0, np.nan)
        )
        out["acceptance"] = out["close_location"].rolling(3).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["setup"] = (
            (out["replication"] > params.replication_threshold)
            & (out["return"] > 0)
            & (out["acceptance"] > params.acceptance_threshold)
            & (out["close_location"] > params.acceptance_threshold)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ReplicationParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        stop = np.nan
        watermark = np.nan
        armed = False
        entry_bar = -1
        entry_atr = np.nan
        setup = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        for i in range(len(data)):
            if active:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's observed open.
                    entry = opens[i]
                    entry_bar = i
                    watermark = entry
                    stop = entry - params.trail_atr * entry_atr
                valid = np.isfinite([entry, highs[i], lows[i], atr[i]]).all()
                # Existing stops apply first; newly raised stops apply next session.
                # ponytail: daily OHLC cannot order intrabar touches; use intraday bars if needed.
                exit_now = not valid or lows[i] <= stop
                if not exit_now:
                    watermark = max(watermark, highs[i])
                    armed = armed or watermark >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, watermark - params.trail_atr * atr[i])
                    exit_now = i - entry_bar + 1 >= params.hold_bars
                if exit_now:
                    active = False
                    entry = np.nan
                    armed = False
                else:
                    signals[i] = 1
                continue
            if setup[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry = np.nan
                entry_atr = atr[i]
                armed = False
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ReplicationParams()
    bars = pd.DataFrame(
        {"open": [100.0] * 4, "high": [100.5] * 4,
         "low": [99.5] * 4, "close": [100.0] * 4, "volume": [1000.0] * 4},
        index=pd.date_range("2020-01-01", periods=4),
    )
    derived = pd.DataFrame({"setup": [True, False, False, False], "atr": 1.0}, index=bars.index)
    assert strategy.generate_signals(bars, derived, None, params).data["signal"].tolist() == [0, 1, 1, 0]
    bars.loc[bars.index[1], "low"] = 98.0
    assert strategy.generate_signals(bars, derived, None, params).data["signal"].tolist() == [0, 1, 0, 0]
    bars.loc[bars.index[1], ["high", "low"]] = [102.0, 100.1]
    bars.loc[bars.index[2], "low"] = 100.4
    params.hold_bars = 1
    assert strategy.generate_signals(bars, derived, None, params).data["signal"].tolist() == [0, 1, 0, 0]
