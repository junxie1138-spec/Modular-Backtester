from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    memory: int = 40
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791283593"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(63 + params.memory + 6)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.memory < 10 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("memory must be >= 10 and trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["peak"] = close.rolling(63, min_periods=63).max()
        out["depth"] = (out["peak"] - close) / out["peak"].where(out["peak"] > 0)
        out["depth_atr"] = (out["peak"] - close) / out["atr"].where(out["atr"] > 0)
        out["depth_change"] = out["depth"].diff()
        out["previous_change"] = out["depth_change"].shift(1)
        out["weekly_change"] = out["depth_change"].shift(5)
        out["weekly_correlation"] = out["depth_change"].rolling(
            params.memory, min_periods=params.memory
        ).corr(out["weekly_change"]).replace([np.inf, -np.inf], np.nan)
        # Weekly anticorrelation predicts an unwinding of last week's deepening drawdown.
        out["entry"] = (
            (out["depth_atr"] >= 2.0)
            & (out["weekly_correlation"] < -0.15)
            & (out["weekly_change"] > 0)
            & (out["depth_change"] < 0)
            & (out["previous_change"] >= 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        highest_close = -np.inf
        stop = -np.inf
        for i, (close, atr, enter) in enumerate(zip(closes, atrs, entries)):
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                active = False
            elif active:
                # Entry fills at this bar's open; only in-trade closes set the high-water mark.
                highest_close = max(highest_close, close)
                stop = max(stop, highest_close - params.trail_atr * atr)
                if close <= stop or i - entry_bar + 1 >= 18:
                    active = False
            elif enter:
                active = True
                entry_bar = i + 1
                highest_close = -np.inf
                stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    closes = np.array([100., 100., 110., 106., 105.] + [105.] * 19)
    data = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1,
         "close": closes, "volume": 1000.},
        index=index,
    )
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    inputs = pd.DataFrame(
        {"atr": [1., 1., 1.] + [10.] * 21, "entry": [True] + [False] * 23},
        index=index,
    )
    result = strategy.generate_signals(data, inputs, context, params).data
    # The stop remains 107 after ATR expands; the exit is filled one bar later.
    assert result["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    data.loc[:, ["open", "high", "low", "close"]] = 100.
    inputs["atr"] = 1.
    result = strategy.generate_signals(data, inputs, context, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
