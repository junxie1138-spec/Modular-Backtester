from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SteadyTrendParams:
    noise_ceiling: float = 0.55
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[SteadyTrendParams]):
    strategy_id = "gen_local_1791195458"

    @classmethod
    def params_type(cls):
        return SteadyTrendParams

    def warmup_bars(self, params: SteadyTrendParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: SteadyTrendParams) -> pd.DataFrame:
        if not np.isfinite(params.noise_ceiling) or params.noise_ceiling <= 0:
            raise ValueError("noise_ceiling must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["change_std"] = out["change"].rolling(8, min_periods=8).std(ddof=0)
        scale = out["atr"].where(out["atr"] > 0)
        out["noise"] = out["change_std"] / scale
        out["displacement"] = data["close"].diff(8) / scale
        valid = np.isfinite(out["noise"]) & np.isfinite(out["displacement"])
        eligible = valid & (out["noise"] <= params.noise_ceiling) & (out["displacement"].abs() >= 2.0)
        out["direction"] = 0
        out.loc[eligible & (out["displacement"] > 0), "direction"] = 1
        out.loc[eligible & (out["displacement"] < 0), "direction"] = -1
        out["entry"] = out["direction"].where(
            out["direction"].ne(out["direction"].shift(1).fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SteadyTrendParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        position = age = 0
        highest = lowest = stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = age = 0
                continue
            if position:
                if age == 0:
                    # Entry fills at this bar's open after the final signal shift.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    highest = lowest = opens[i]
                    stop = opens[i] - position * params.trail_atr * atrs[i - 1]
                highest = max(highest, close)
                lowest = min(lowest, close)
                age += 1
                if position == 1:
                    stop = max(stop, highest - params.trail_atr * atr)
                    stopped = close < stop
                else:
                    # Mirror the highest-close long trail with a lowest-close short trail.
                    stop = min(stop, lowest + params.trail_atr * atr)
                    stopped = close > stop
                if stopped or age >= 4:
                    position = age = 0
                signal[i] = position
                continue
            if entries[i]:
                position = int(entries[i])
                age = 0
            signal[i] = position
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=11)
    close = [100, 103, 102.5, 101.5, 101, 101, 100, 100, 98, 99.2, 100]
    data = pd.DataFrame({
        "open": close, "high": np.array(close) + 0.5,
        "low": np.array(close) - 0.5, "close": close, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1],
        "entry": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = SteadyTrendParams(trail_atr=1.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    got = strategy.generate_signals(data, ind, ctx, params).data
    assert got["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0]
    ind["atr"] = 10.0
    ind["entry"] = [1] + [0] * 10
    got = strategy.generate_signals(data, ind, ctx, params).data
    assert got["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()
    assert (got["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
