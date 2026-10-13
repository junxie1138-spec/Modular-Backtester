from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_window: int = 24
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1791247091"

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    @staticmethod
    def warmup_bars(params: OrdinalParams) -> int:
        return int(params.complexity_window + 7)

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if params.complexity_window < 6 or params.holding_bars < 1:
            raise ValueError("complexity_window must be >= 6 and holding_bars >= 1")
        window = params.complexity_window
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna() & (a != b) & (a != c) & (b != c)
        pattern = pd.Series(np.select([
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=-1), index=data.index).where(valid)
        out = pd.DataFrame(index=data.index)
        out["ordinal_pattern"] = pattern
        out["valid_patterns"] = valid.astype(float).rolling(window, min_periods=window).sum()
        entropy = pd.Series(0.0, index=data.index)
        for code in range(6):
            probability = pattern.eq(code).astype(float).rolling(window, min_periods=window).sum().div(out["valid_patterns"].replace(0.0, np.nan))
            out[f"pattern_probability_{code}"] = probability
            entropy -= probability * np.log(probability.where(probability > 0.0, 1.0))
        out["ordinal_entropy"] = (entropy / np.log(6.0)).where(out["valid_patterns"] >= 0.9 * window)
        out["prior_entropy_floor"] = out["ordinal_entropy"].shift(1).rolling(5, min_periods=5).min()
        out["entropy_change"] = out["ordinal_entropy"].diff()
        out["price_mean"] = close.rolling(window, min_periods=window).mean()
        out["window_return"] = close.div(close.shift(window).replace(0.0, np.nan)) - 1.0
        out["daily_return"] = close.div(close.shift(1).replace(0.0, np.nan)) - 1.0
        out["entry_eligible"] = (
            (out["prior_entropy_floor"] <= 0.45)
            & (out["ordinal_entropy"] >= 0.60)
            & (out["entropy_change"] >= 0.08)
            & (close < out["price_mean"])
            & (out["window_return"] < 0.0)
            & (out["daily_return"] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        eligible = indicators["entry_eligible"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = None
        for bar in range(len(data)):
            if entry_bar is not None:
                if bar - entry_bar == params.holding_bars:
                    entry_bar = None
                else:
                    signal[bar] = 1
            elif eligible[bar]:
                entry_bar = bar
                signal[bar] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = OrdinalParams()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_eligible": True}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20:38].eq(1).all()
    assert result["signal"].iloc[38] == 0
    assert result["size"].gt(0.0).all()
