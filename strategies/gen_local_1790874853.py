from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    box_bars: int = 8
    rank_bars: int = 80
    compression_pct: float = 0.15
    arm_bars: int = 5
    atr_bars: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790874853"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(200, params.box_bars + params.rank_bars,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ma200"] = close.rolling(200).mean()
        out["box_high"] = data["high"].rolling(params.box_bars).max()
        out["box_low"] = data["low"].rolling(params.box_bars).min()
        out["width"] = (out["box_high"] - out["box_low"]) / close.where(close > 0)
        out["compression_rank"] = out["width"].rolling(params.rank_bars).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        values = indicators[["atr", "ma200", "box_high", "box_low",
                             "compression_rank"]].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        armed_left = 0
        can_arm = True
        upper = lower = extreme = stop = np.nan
        warmup = self.warmup_bars(params)
        for i, close in enumerate(closes):
            atr, ma, box_high, box_low, rank = values[i]
            if (i + 1 < warmup or not np.isfinite(close)
                    or not np.isfinite(values[i]).all() or atr <= 0):
                position = 0
                armed_left = 0
                can_arm = True
                continue
            if position:
                # First iteration after entry decision is the actual fill bar.
                age += 1
                if not np.isfinite(extreme):
                    extreme = close
                    stop = close - position * params.trail_atr * atr
                elif position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                stopped = close <= stop if position == 1 else close >= stop
                regime_lost = close <= ma if position == 1 else close >= ma
                if stopped or regime_lost or age >= params.hold_bars:
                    position = 0
                raw[i] = position
                continue
            if rank >= 0.5:
                can_arm = True
            if armed_left:
                direction = 1 if close > upper else (-1 if close < lower else 0)
                armed_left -= 1
                if direction:
                    armed_left = 0
                    if (direction == 1 and close > ma) or (direction == -1 and close < ma):
                        position = direction
                        age = 0
                        extreme = stop = np.nan
                        raw[i] = position
                        continue
            if not armed_left and can_arm and rank <= params.compression_pct:
                # Freeze this box; subsequent rolling boundaries cannot move the trigger.
                upper, lower = box_high, box_low
                armed_left = params.arm_bars
                can_arm = False
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams()
    n = strategy.warmup_bars(params) - 1
    index = pd.date_range("2020-01-01", periods=n + 8)
    ctx = StrategyContext("SPY", "1d", n + 1)
    for direction in (1, -1):
        tail = np.array([100, 103, 104, 108, 107, 104, 104, 104], dtype=float)
        if direction == -1:
            tail = 200 - tail
        close = np.concatenate([np.full(n, 100.0), tail])
        data = pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "ma200": 100.0,
                            "box_high": 101.0, "box_low": 99.0,
                            "compression_rank": 0.8}, index=index)
        ind.loc[index[n], "compression_rank"] = 0.1
        ind.loc[index[n + 4:], "atr"] = 10.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[n:].tolist() == [0, 0, direction, direction,
                                                     direction, direction, 0, 0]
        prefix = strategy.generate_signals(data.iloc[:-2], ind.iloc[:-2], ctx, params).data
        assert prefix.equals(result.iloc[:-2])
        actual = strategy.indicators(data, params)
        assert actual.index.equals(data.index)
        assert strategy.generate_signals(data, actual, ctx, params).data["signal"].iloc[:n].eq(0).all()
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
