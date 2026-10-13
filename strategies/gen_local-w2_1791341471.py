from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    complexity_window: int = 18
    recovery_window: int = 18
    peak_window: int = 63
    trough_window: int = 21
    min_diversity: float = 4.5
    min_efficiency: float = 0.15
    min_drawdown: float = 0.04


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791341471"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.complexity_window + 3,
                       params.recovery_window + 1,
                       params.peak_window, params.trough_window))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.complexity_window, params.recovery_window,
               params.peak_window, params.trough_window) < 1:
            raise ValueError("Lookback windows must be positive")
        if not (1.0 <= params.min_diversity <= 6.0
                and 0.0 <= params.min_efficiency <= 1.0
                and 0.0 < params.min_drawdown < 1.0):
            raise ValueError("Invalid diversity, efficiency, or drawdown threshold")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns are ordered by observation time, earliest first.
        out["ordinal_pattern"] = pd.Series(np.select([
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index).where(valid)
        probability_columns = []
        for pattern in range(6):
            column = f"pattern_probability_{pattern}"
            occurrence = out["ordinal_pattern"].eq(pattern).astype(float).where(valid)
            out[column] = occurrence.rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
            probability_columns.append(column)
        out["pattern_concentration"] = out[probability_columns].pow(2).sum(axis=1, min_count=6)
        out["ordinal_diversity"] = 1.0 / out["pattern_concentration"].replace(0.0, np.nan)
        out["close_change"] = close.diff()
        out["path_length"] = out["close_change"].abs().rolling(
            params.recovery_window, min_periods=params.recovery_window
        ).sum()
        out["recovery_displacement"] = close.diff(params.recovery_window)
        out["recovery_efficiency"] = (
            out["recovery_displacement"] / out["path_length"].replace(0.0, np.nan)
        )
        out["recent_peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["recent_trough"] = close.rolling(params.trough_window, min_periods=params.trough_window).min()
        out["drawdown_depth"] = out["recent_trough"] / out["recent_peak"] - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        complexity_agrees = indicators["ordinal_diversity"].ge(params.min_diversity)
        recovery_agrees = indicators["recovery_efficiency"].ge(params.min_efficiency)
        drawdown_context = indicators["drawdown_depth"].le(-params.min_drawdown)
        entry_condition = (complexity_agrees & recovery_agrees & drawdown_context).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintain exposure while the complete entry condition holds; exit when it flips.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({
        "ordinal_diversity": [6.0, 6.0, 3.0, 6.0, 6.0, np.nan],
        "recovery_efficiency": [0.3, 0.3, 0.3, -0.1, 0.3, 0.3],
        "drawdown_depth": [-0.1] * 6,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=63)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, 1]
    assert signals["size"].gt(0).all()
    flat_index = pd.date_range("2020-01-01", periods=100)
    flat = pd.DataFrame(100.0, index=flat_index, columns=data.columns)
    flat_indicators = strategy.indicators(flat, params)
    flat_signals = strategy.generate_signals(flat, flat_indicators, ctx, params).data
    assert flat_indicators.index.equals(flat.index)
    assert flat_signals["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.complexity_window + 3


if __name__ == "__main__":
    _self_check()
