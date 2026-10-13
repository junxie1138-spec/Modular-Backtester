from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 20
    rank_window: int = 60
    extreme_pct: float = 0.25
    profit_target: float = 0.008
    max_hold: int = 2

    def __post_init__(self):
        if self.peak_window < 3 or self.rank_window < 3:
            raise ValueError("Lookbacks must be at least three.")
        if not 0.0 < self.extreme_pct < 0.5:
            raise ValueError("extreme_pct must be between zero and one-half.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")
        if self.max_hold not in (1, 2):
            raise ValueError("max_hold must be one or two.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791256787"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.peak_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        low = data["low"].where(np.isfinite(data["low"]) & (data["low"] > 0))
        high = data["high"].where(np.isfinite(data["high"]) & (data["high"] > 0))
        out["peak"] = high.rolling(params.peak_window).max()
        out["trough"] = low.rolling(params.peak_window).min()
        out["drawdown"] = close / out["peak"] - 1.0
        out["drawup"] = close / out["trough"] - 1.0
        for name in ("drawdown", "drawup"):
            out[name + "_rank"] = out[name].rolling(params.rank_window).rank(pct=True)
        out["low_rank"] = low.rolling(params.rank_window).rank(pct=True)
        out["high_rank"] = high.rolling(params.rank_window).rank(pct=True)
        out["low_rank_change"] = out["low_rank"].diff()
        out["high_rank_change"] = out["high_rank"].diff()
        out["low_change"] = low.diff()
        out["high_change"] = high.diff()
        out["close_change"] = close.diff()
        out["long_setup"] = (
            (out["drawdown_rank"] <= params.extreme_pct)
            & (out["drawdown"] < 0)
            & (out["low_rank_change"] > 0)
            & (out["low_change"] > 0)
            & (out["close_change"] > 0)
        ).fillna(False)
        out["short_setup"] = (
            (out["drawup_rank"] >= 1.0 - params.extreme_pct)
            & (out["drawup"] > 0)
            & (out["high_rank_change"] < 0)
            & (out["high_change"] < 0)
            & (out["close_change"] < 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        long_setup = indicators["long_setup"].fillna(False).astype(bool)
        short_setup = indicators["short_setup"].fillna(False).astype(bool)
        longs = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        shorts = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry = np.nan
        for i in range(len(data)):
            if position:
                # A decision at i-1 fills at this bar's open, never at its close.
                if age == 0:
                    entry = opens[i]
                age += 1
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or age >= params.max_hold:
                    position, age, entry = 0, 0, np.nan
            elif np.isfinite(closes[i]) and closes[i] > 0:
                if longs[i] and not shorts[i]:
                    position = 1
                elif shorts[i] and not longs[i]:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": [100.0, 100.0, 100.0, 101.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    setups = pd.DataFrame({
        "long_setup": [False, True, True, False, False, False, False],
        "short_setup": False,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    def signals(params):
        return strategy.generate_signals(data, setups, ctx, params).data["signal"].tolist()
    assert signals(StrategyParams()) == [0, 0, 0, 1, 0, 0, 0]
    data["close"] = 100.0
    assert signals(StrategyParams()) == [0, 0, 0, 1, 1, 0, 0]
    setups["short_setup"] = setups["long_setup"]
    setups["long_setup"] = False
    data.loc[index[3], "close"] = 99.0
    assert signals(StrategyParams()) == [0, 0, 0, -1, 0, 0, 0]
    setups["short_setup"] = [False, True, False, False, False, False, False]
    assert signals(StrategyParams()) == [0] * 7
    derived = strategy.indicators(data, StrategyParams())
    assert not derived["long_setup"].any() and not derived["short_setup"].any()


if __name__ == "__main__":
    _self_check()
