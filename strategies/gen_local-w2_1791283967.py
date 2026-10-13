from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    z_threshold: float = 1.0
    stop_atr: float = 2.5

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.z_threshold, self.stop_atr)):
            raise ValueError("Parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = "gen_local-w2_1791283967"

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params):
        return 80

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(60, min_periods=60).mean()
        out["std"] = close.rolling(60, min_periods=60).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["recent_z"] = out["z"].rolling(5, min_periods=5).mean()
        out["prior_z"] = out["z"].shift(5).rolling(15, min_periods=15).mean()
        out["prior_load"] = out["z"].abs().ge(params.z_threshold).astype(float).where(out["z"].notna()).shift(5).rolling(15, min_periods=15).sum()
        out["shift_z"] = out["recent_z"] - out["prior_z"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        eligible = out["prior_z"].abs().le(0.5) & out["prior_load"].le(2.0) & out["atr"].gt(0)
        out["up"] = eligible & out["recent_z"].ge(params.z_threshold) & out["shift_z"].ge(params.z_threshold)
        out["down"] = eligible & out["recent_z"].le(-params.z_threshold) & out["shift_z"].le(-params.z_threshold)
        out["event"] = (
            (out["up"] & ~out["up"].shift(1, fill_value=False)).astype(int)
            - (out["down"] & ~out["down"].shift(1, fill_value=False)).astype(int)
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        events = indicators["event"].to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_bar = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # The prior close's decision fills at this open; freeze its ATR.
                if np.isfinite(opening[i]) and opening[i] > 0:
                    position = pending
                    entry_bar = i
                    stop = opening[i] - position * params.stop_atr * atr[i - 1]
                pending = 0
            if position:
                stopped = np.isfinite(close[i]) and position * (close[i] - stop) < 0
                if stopped or i - entry_bar + 1 >= 18:
                    position = 0
                raw[i] = position
            elif events[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(events[i])
                raw[i] = pending
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = CapacityParams()
    index = pd.date_range("2020-01-01", periods=25)
    sample = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "event": 0}, index=index)
    derived.iloc[0, 1] = 1
    signal = strategy.generate_signals(sample, derived, None, params).data["signal"]
    assert signal.iloc[0] == 0 and (signal.iloc[1:19] == 1).all() and signal.iloc[19] == 0
    sample.loc[index[3], "close"] = 90.0
    sample.loc[index[23], "close"] = 110.0
    derived.iloc[20, 1] = -1
    signal = strategy.generate_signals(sample, derived, None, params).data["signal"]
    assert signal.iloc[3] == 1 and signal.iloc[4] == 0 and signal.iloc[21] == -1 and signal.iloc[24] == 0
    flat = pd.concat([sample.iloc[:1]] * 100, ignore_index=True)
    assert strategy.indicators(flat, params)["event"].eq(0).all()
