from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendFloorParams:
    segment_bars: int = 2
    strength_bars: int = 6
    rank_bars: int = 63
    entry_percentile: float = 0.80
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    initial_stop_atr: float = 2.0
    trail_atr: float = 1.6
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TrendFloorParams]):
    strategy_id = "gen_local-w2_1791289179"

    @classmethod
    def params_type(cls):
        return TrendFloorParams

    def warmup_bars(self, params: TrendFloorParams) -> int:
        return int(max(params.segment_bars + params.strength_bars + params.rank_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: TrendFloorParams) -> pd.DataFrame:
        if min(params.segment_bars, params.strength_bars, params.rank_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not (0.0 < params.entry_percentile <= 1.0):
            raise ValueError("entry_percentile must be in (0, 1].")
        if params.breakeven_pct <= 0 or min(params.initial_stop_atr, params.trail_atr) <= 0:
            raise ValueError("Stop distances must be positive.")
        if not 3 <= params.max_hold_bars <= 5:
            raise ValueError("max_hold_bars must be between 3 and 5.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["segment_return"] = out["log_close"].diff(params.segment_bars)
        out["floor"] = out["segment_return"].rolling(params.strength_bars, min_periods=params.strength_bars).min()
        out["noise"] = out["return"].pow(2).rolling(params.strength_bars, min_periods=params.strength_bars).mean().pow(0.5)
        out["strength"] = out["floor"].div(out["noise"].replace(0.0, np.nan))
        out["percentile"] = out["strength"].rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["qualified"] = (
            out["percentile"].ge(params.entry_percentile)
            & out["floor"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = out["qualified"] & out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TrendFloorParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decision = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    entry = peak = opens[i]
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                if not np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all():
                    active = False
                    continue
                # A stop touch is observed at this close; flatten at the next open.
                # Test the previously established stop before raising it.
                if lows[i] <= stop or age >= params.max_hold_bars:
                    active = False
                    continue
                peak = max(peak, highs[i])
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                decision[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                decision[i] = 1
        df["signal"] = decision
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.0, 100.0, 101.0, 103.0, 101.0, 100.0],
        "low": [99.0, 99.0, 100.0, 101.0, 99.0, 99.0],
        "close": [100.0, 100.0, 101.0, 102.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    qualified = pd.Series([False, True, True, False, False, False], index=index)
    ind = pd.DataFrame({
        "atr": 1.0,
        "confirmed": qualified & qualified.shift(1, fill_value=False),
    }, index=index)
    strategy = GeneratedStrategy()
    params = TrendFloorParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0]
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], ind.iloc[:end], ctx, params).data
        assert prefix.equals(result.iloc[:end])


if __name__ == "__main__":
    _self_check()
