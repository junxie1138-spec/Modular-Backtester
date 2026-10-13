from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entry_z: float = 1.8
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791335371"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 100

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (np.isfinite(params.entry_z) and params.entry_z > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        # ponytail: 40-tap fractional truncation; extend taps and warmup if tail bias dominates.
        fd = close.copy()
        weight = 1.0
        for lag in range(1, 40):
            weight *= (lag - 1 - 0.45) / lag
            fd = fd + weight * close.shift(lag)
        out["fractional_close"] = fd
        out["fd_mean"] = fd.shift(1).rolling(60, min_periods=60).mean()
        out["fd_std"] = fd.shift(1).rolling(60, min_periods=60).std(ddof=0)
        out["z"] = (fd - out["fd_mean"]) / out["fd_std"].replace(0.0, np.nan)
        out["turn"] = fd.diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volume_mean"] = data["volume"].shift(1).rolling(20, min_periods=20).mean()
        out["volume_ratio"] = data["volume"] / out["volume_mean"].replace(0.0, np.nan)
        out["body_fraction"] = (close - data["open"]).abs() / out["true_range"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        if not (np.isfinite(params.entry_z) and params.entry_z > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        columns = ["z", "turn", "atr", "volume_ratio", "body_fraction"]
        values = indicators[columns].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        ready = np.isfinite(values).all(axis=1) & np.isfinite(closes) & (closes > 0)
        ready &= values[:, 2] > 0
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry_atr, stop = np.nan, np.nan
        for i in range(len(data)):
            if position and age == 0:
                # Next-bar open is the entry reference; ATR was frozen at the decision.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop = opens[i] - position * params.stop_atr * entry_atr
                else:
                    position = 0
            if position:
                age += 1
                stopped = position * (closes[i] - stop) < 0
                if not ready[i] or stopped or age >= 18:
                    position = 0
            elif ready[i]:
                z, turn, atr, volume_ratio, body_fraction = values[i]
                absorbed = volume_ratio >= 1.25 and body_fraction <= 0.35
                if absorbed and z <= -params.entry_z and turn > 0:
                    position = 1
                elif absorbed and z >= params.entry_z and turn < 0:
                    position = -1
                if position:
                    age, entry_atr = 0, atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "z": 0.0, "turn": 0.0, "atr": 1.0,
        "volume_ratio": 2.0, "body_fraction": 0.1,
    }, index=index)
    indicators.loc[index[0], ["z", "turn"]] = [-2.0, 1.0]
    indicators.loc[index[4], ["z", "turn"]] = [2.0, -1.0]
    data.loc[index[2], "close"] = 96.0
    data.loc[index[5], "close"] = 104.0
    signal = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert signal.iloc[:7].tolist() == [0, 1, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    indicators.loc[index[4], ["z", "turn"]] = [0.0, 0.0]
    signal = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert (signal.iloc[1:19] == 1).all() and signal.iloc[19] == 0
    indicators.loc[index[0], "atr"] = np.nan
    assert (strategy.generate_signals(data, indicators, None, params).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
