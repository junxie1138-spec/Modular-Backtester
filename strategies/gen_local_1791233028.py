from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    min_drawdown: float = 0.025
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791233028"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not np.isfinite(params.min_drawdown) or not 0 < params.min_drawdown < 1:
            raise ValueError("min_drawdown must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(20, min_periods=20).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].replace(0.0, np.nan)).clip(lower=0.0)
        out["previous_depth"] = out["depth"].shift(1)
        out["stress_depth"] = out["depth"].shift(1).rolling(5, min_periods=5).max()
        out["recovery_boundary"] = out["stress_depth"] * 0.5
        out["breakout_level"] = data["high"].shift(1).rolling(3, min_periods=3).max()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(5, min_periods=5).mean()
        out["entry"] = (
            (out["stress_depth"] >= params.min_drawdown)
            & (out["previous_depth"] > out["recovery_boundary"])
            & (out["depth"] <= out["recovery_boundary"])
            & (data["close"] > out["breakout_level"])
            & out["atr"].notna()
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        activated = False
        entry_price = 0.0
        stop = 0.0
        highest = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    active = True
                    activated = False
                    entry_price = opens[i]
                    highest = entry_price
                    stop = entry_price - params.trail_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not np.isfinite([highs[i], lows[i], closes[i], atr[i]]).all() or atr[i] <= 0.0:
                    active = False
                    continue
                # The existing stop applies throughout this bar; new ratchets
                # apply at its close, avoiding assumptions about high/low order.
                breached = lows[i] <= stop
                highest = max(highest, highs[i])
                if highest >= entry_price * 1.005:
                    activated = True
                if activated:
                    stop = max(stop, entry_price, highest - params.trail_atr * atr[i])
                if breached or closes[i] <= stop or age >= 2:
                    active = False
                else:
                    raw[i] = 1
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                raw[i] = 1
                pending = True
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.2] * 5,
        "low": [99.9] * 5,
        "close": [100.0] * 5,
        "volume": [1000.0] * 5,
    }, index=index)
    ind = pd.DataFrame({"atr": [1.0] * 5,
                        "entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "high"] = 102.0
    result = strategy.generate_signals(data, ind, None, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    data.loc[index[1], "high"] = 100.2
    data.loc[index[1], "low"] = 98.0
    result = strategy.generate_signals(data, ind, None, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
