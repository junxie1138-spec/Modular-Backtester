from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    gap_atr: float = 0.35
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = "gen_local_1791273923"

    @classmethod
    def params_type(cls):
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        if not (np.isfinite(params.gap_atr) and params.gap_atr > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["floor"] = data["close"].rolling(5, min_periods=5).min().shift(1)
        out["ceiling"] = data["close"].rolling(5, min_periods=5).max().shift(1)
        out["peak"] = data["close"].rolling(20, min_periods=20).max().shift(1)
        out["trough"] = data["close"].rolling(20, min_periods=20).min().shift(1)
        out["gap_atr"] = (data["open"] - out["previous_close"]) / out["prior_atr"]
        out["drawdown_atr"] = (out["peak"] - out["previous_close"]) / out["prior_atr"]
        out["runup_atr"] = (out["previous_close"] - out["trough"]) / out["prior_atr"]
        long_entry = (
            (out["drawdown_atr"] >= 2.0)
            & (out["gap_atr"] <= -params.gap_atr)
            & (data["open"] < out["floor"])
            & (data["close"] > out["floor"])
        )
        short_entry = (
            (out["runup_atr"] >= 2.0)
            & (out["gap_atr"] >= params.gap_atr)
            & (data["open"] > out["ceiling"])
            & (data["close"] < out["ceiling"])
        )
        ready = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        out["entry"] = (long_entry.astype(int) - short_entry.astype(int)).where(ready, 0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapRecoveryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        held = 0
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if position:
                # An instruction from yesterday fills at today's open.
                if held == 0:
                    stop = opens[i] - position * params.stop_atr * entry_atr
                held += 1
                stopped = position * (closes[i] - stop) < 0
                if not np.isfinite(stop) or not np.isfinite(closes[i]) or stopped or held >= 4:
                    position = 0
                    held = 0
            elif entries[i] in (-1, 1) and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(entries[i])
                entry_atr = atrs[i]
                held = 0
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRecoveryParams()
    index = pd.date_range("2020-01-01", periods=8)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    for direction in (1, -1):
        data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
        indicators = pd.DataFrame({"atr": 2.0, "entry": 0}, index=index)
        indicators.iloc[1, indicators.columns.get_loc("entry")] = direction
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, direction, 0, 0]
        # Favorable movement must not move the frozen stop.
        data.loc[index[3], "close"] = 100.0 + direction * 20.0
        data.loc[index[4], "close"] = 100.0 - direction * 2.0
        assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].equals(result["signal"])
        # Stop uses the fill open, not yesterday's close.
        data.loc[index[2], "open"] = 100.0 + direction * 10.0
        stopped = strategy.generate_signals(data, indicators, ctx, params).data
        assert stopped["signal"].tolist() == [0, 0, direction, 0, 0, 0, 0, 0]
        assert (stopped["size"] > 0).all()
    empty = data.iloc[:0]
    assert strategy.generate_signals(empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
